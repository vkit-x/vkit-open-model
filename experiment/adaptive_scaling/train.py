from typing import Tuple, Mapping, Optional, List
import itertools
from enum import Enum, unique
import logging
import statistics
import shutil
import math

import attrs
import cattrs
import iolite as io
import torch
from torch.utils.data import DataLoader

from vkit.utility import dyn_structure, PathType
from vkit_open_model.model import (
    AdaptiveScaling,
    AdaptiveScalingSize,
    AdaptiveScalingNeckHeadType,
)
from vkit_open_model.dataset import (
    adaptive_scaling_dataset_collate_fn,
    AdaptiveScalingIterableDataset,
)
from vkit_open_model.loss_function import AdaptiveScalingLossFunction
from vkit_open_model.training import (
    batch_to_device,
    device_is_cuda,
    enable_cudnn_benchmark,
    enable_cudnn_deterministic,
    setup_seeds,
    calculate_iterable_dataset_num_samples,
    generate_iterable_dataset_rng_seeds,
    Metrics,
)

logger = logging.getLogger(__name__)


@attrs.define
class EpochConfig:
    torch_seed: int = 133
    num_epochs: int = 98
    train_num_batches: int = 672
    train_batch_size: int = 7
    train_prefetch_factor: int = 4
    dev_num_batches: int = 68
    dev_batch_size: int = 32
    dev_rng_seed: int = 13
    dev_prefetch_factor: int = 4
    num_workers: int = 8
    avg_num_batches: int = 50
    enable_overfit_testing: bool = False


@attrs.define
class ModelConfig:
    size: AdaptiveScalingSize = AdaptiveScalingSize.SMALL
    neck_head_type: AdaptiveScalingNeckHeadType = AdaptiveScalingNeckHeadType.FPN
    init_scale_output_bias: float = 8.0


@attrs.define
class OptimizerConfig:
    adamw_lr: float = 1E-3
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_weight_decay: float = 0.01
    cosine_annealing_warm_restarts_t0: int = 14
    cosine_annealing_warm_restarts_tmulti: int = 2
    cosine_annealing_warm_restarts_eta_min: float = 1E-5
    clip_grad_norm_max_norm: Optional[float] = None


@attrs.define
class LossConfig:
    negative_ratio: float = 3.0
    bce_factor: float = 2.0
    dice_factor: float = 1.0
    l1_factor: float = 1.0


@unique
class MetricsTag(Enum):
    TRAIN_LOSS = 'train_loss'
    DEV_LOSS = 'dev_loss'


@attrs.define
class RestoreState:
    epoch_idx: int
    model_jit_state_dict: Mapping[str, torch.Tensor]
    optimizer_state_dict: Mapping[str, torch.Tensor]
    optimizer_scheduler_state_dict: Mapping[str, torch.Tensor]


def train(
    adaptive_scaling_dataset_steps_json: str,
    output_folder: str,
    reset_output_folder: bool = False,
    device_value: str = 'cuda',
    epoch_config_json: Optional[str] = None,
    model_config_json: Optional[str] = None,
    optimizer_config_json: Optional[str] = None,
    loss_config_json: Optional[str] = None,
    restore_state_dict_path: Optional[str] = None,
    reset_epoch_idx: bool = False,
):
    out_fd = io.folder(output_folder, reset=reset_output_folder, touch=True)
    logger.info(f'out_fd = {out_fd}')

    # Logging to file.
    logger.addHandler(logging.FileHandler(out_fd / 'log.txt'))
    # Set logging format.
    logger_formatter = logging.Formatter('%(message)s   [%(asctime)s]')
    for logger_handler in itertools.chain(logging.getLogger().handlers, logger.handlers):
        logger_handler.setFormatter(logger_formatter)

    # Config.
    epoch_config = dyn_structure(
        epoch_config_json,
        EpochConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('epoch_config:')
    logger.info(cattrs.unstructure(epoch_config))
    io.write_json(out_fd / 'epoch_config.json', cattrs.unstructure(epoch_config), indent=2)

    model_config = dyn_structure(
        model_config_json,
        ModelConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('model_config:')
    logger.info(cattrs.unstructure(model_config))
    io.write_json(out_fd / 'model_config.json', cattrs.unstructure(model_config), indent=2)

    optimizer_config = dyn_structure(
        optimizer_config_json,
        OptimizerConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('optimizer_config:')
    logger.info(cattrs.unstructure(optimizer_config))
    io.write_json(out_fd / 'optimizer_config.json', cattrs.unstructure(optimizer_config), indent=2)

    loss_config = dyn_structure(
        loss_config_json,
        LossConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('loss_config:')
    logger.info(cattrs.unstructure(loss_config))
    io.write_json(out_fd / 'loss_config.json', cattrs.unstructure(loss_config), indent=2)

    device = torch.device(device_value)
    logger.info(f'device = {device}')

    # Init.
    setup_seeds(torch_seed=epoch_config.torch_seed)
    enable_cudnn_benchmark(device)
    enable_cudnn_deterministic(device)

    # Dataset.
    logger.info('dataset')
    train_num_samples = calculate_iterable_dataset_num_samples(
        num_workers=epoch_config.num_workers,
        batch_size=epoch_config.train_batch_size,
        num_batches=epoch_config.train_num_batches,
    )
    dev_num_samples = calculate_iterable_dataset_num_samples(
        num_workers=epoch_config.num_workers,
        batch_size=epoch_config.dev_batch_size,
        num_batches=epoch_config.dev_num_batches,
    )
    logger.info(f'num_workers={epoch_config.num_workers}')
    logger.info(f'train_num_samples = {train_num_samples}, dev_num_samples={dev_num_samples}')

    shutil.copyfile(
        adaptive_scaling_dataset_steps_json, out_fd / 'adaptive_scaling_dataset_steps.json'
    )
    dev_rng_seed = generate_iterable_dataset_rng_seeds(
        num_samples=dev_num_samples,
        rng_seed=epoch_config.dev_rng_seed,
    )
    dev_adaptive_scaling_dataset = AdaptiveScalingIterableDataset(
        steps_json=adaptive_scaling_dataset_steps_json,
        num_samples=dev_num_samples,
        rng_seed=dev_rng_seed,
    )
    if not epoch_config.enable_overfit_testing:
        train_adaptive_scaling_dataset = AdaptiveScalingIterableDataset(
            steps_json=adaptive_scaling_dataset_steps_json,
            num_samples=train_num_samples,
        )
    else:
        train_rng_seed = dev_rng_seed * math.ceil(train_num_samples / dev_num_samples)
        train_rng_seed = train_rng_seed[:train_num_samples]
        train_adaptive_scaling_dataset = AdaptiveScalingIterableDataset(
            steps_json=adaptive_scaling_dataset_steps_json,
            num_samples=train_num_samples,
            rng_seed=train_rng_seed,
        )

    # Model.
    model = AdaptiveScaling(
        size=model_config.size,
        neck_head_type=model_config.neck_head_type,
        init_scale_output_bias=model_config.init_scale_output_bias,
    )
    model_jit: torch.jit.ScriptModule = torch.jit.script(model)  # type: ignore
    model_jit = model_jit.to(device)
    del model

    # Loss.
    loss_function = AdaptiveScalingLossFunction(
        negative_ratio=loss_config.negative_ratio,
        bce_factor=loss_config.bce_factor,
        dice_factor=loss_config.dice_factor,
        l1_factor=loss_config.l1_factor,
    )

    # Optimizer.
    optimizer = torch.optim.AdamW(
        params=model_jit.parameters(),
        lr=optimizer_config.adamw_lr,
        betas=optimizer_config.adamw_betas,
        weight_decay=optimizer_config.adamw_weight_decay,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=optimizer_config.cosine_annealing_warm_restarts_t0,
        T_mult=optimizer_config.cosine_annealing_warm_restarts_tmulti,
        eta_min=optimizer_config.cosine_annealing_warm_restarts_eta_min,
    )

    # Metircs.
    metrics = Metrics(MetricsTag, avg_num_batches=epoch_config.avg_num_batches)

    # Epoch.
    epoch_idx = 0

    # Restore.
    if restore_state_dict_path:
        restore_state = dyn_structure(
            torch.load(restore_state_dict_path, map_location='cpu'),
            RestoreState,
        )
        if not reset_epoch_idx:
            epoch_idx = restore_state.epoch_idx + 1
        model_jit.load_state_dict(restore_state.model_jit_state_dict)
        optimizer.load_state_dict(restore_state.optimizer_state_dict)  # type: ignore
        optimizer_scheduler.load_state_dict(
            restore_state.optimizer_scheduler_state_dict  # type: ignore
        )

    # Dataloader.
    dev_data_loader = DataLoader(
        dev_adaptive_scaling_dataset,
        collate_fn=adaptive_scaling_dataset_collate_fn,
        batch_size=epoch_config.dev_batch_size,
        num_workers=epoch_config.num_workers,
        prefetch_factor=epoch_config.dev_prefetch_factor,
        pin_memory=device_is_cuda(device),
        pin_memory_device=str(device) if device_is_cuda(device) else '',
    )
    train_data_loader = DataLoader(
        train_adaptive_scaling_dataset,
        collate_fn=adaptive_scaling_dataset_collate_fn,
        batch_size=epoch_config.train_batch_size,
        num_workers=epoch_config.num_workers,
        prefetch_factor=epoch_config.train_prefetch_factor,
        pin_memory=device_is_cuda(device),
        pin_memory_device=str(device) if device_is_cuda(device) else '',
        persistent_workers=True,
    )

    best_dev_loss = float('inf')

    while epoch_idx < epoch_config.num_epochs:
        logger.info('Training...')
        model_jit.train()
        torch.set_grad_enabled(True)

        for batch_idx, batch in enumerate(train_data_loader, start=1):
            batch = batch_to_device(batch, device)
            mask_feature, scale_feature = model_jit(batch['image'])

            loss = loss_function(
                mask_feature=mask_feature,
                scale_feature=scale_feature,
                downsampled_mask=batch['downsampled_mask'],
                downsampled_score_map=batch['downsampled_score_map'],
                downsampled_shape=batch['downsampled_shape'],
                downsampled_core_box=batch['downsampled_core_box'],
            )
            avg_loss = metrics.update(MetricsTag.TRAIN_LOSS, float(loss))
            loss.backward()

            if optimizer_config.clip_grad_norm_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    model_jit.parameters(),
                    optimizer_config.clip_grad_norm_max_norm,
                )

            optimizer.step()
            optimizer_scheduler.step(
                epoch_idx + (batch_idx - 1) / epoch_config.train_num_batches  # type: ignore
            )
            optimizer.zero_grad()

            if batch_idx % 4 == 0 or batch_idx == epoch_config.train_num_batches:
                logger.info(
                    f'E={epoch_idx}, '
                    f'B={batch_idx}/{epoch_config.train_num_batches}, '
                    f'L={avg_loss:.5f}, '
                    f'LR={optimizer_scheduler.get_last_lr()[-1]:.6f}'
                )

        logger.info('Evaluating...')
        model_jit.eval()
        torch.set_grad_enabled(False)

        dev_losses: List[float] = []
        for batch_idx, batch in enumerate(dev_data_loader, start=1):
            batch = batch_to_device(batch, device)
            mask_feature, scale_feature = model_jit(batch['image'])

            loss = loss_function(
                mask_feature=mask_feature,
                scale_feature=scale_feature,
                downsampled_mask=batch['downsampled_mask'],
                downsampled_score_map=batch['downsampled_score_map'],
                downsampled_shape=batch['downsampled_shape'],
                downsampled_core_box=batch['downsampled_core_box'],
            )
            loss = float(loss)
            dev_losses.append(loss)

            avg_loss = metrics.update(MetricsTag.TRAIN_LOSS, loss)
            if batch_idx % 4 == 0 or batch_idx >= epoch_config.dev_num_batches:
                logger.info(
                    f'E={epoch_idx}, '
                    f'B={batch_idx}/{epoch_config.dev_num_batches}, '
                    f'L={avg_loss:.5f}'
                )

        dev_loss = statistics.mean(dev_losses)
        logger.info(f'E={epoch_idx}, dev_loss = {dev_loss}')
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            state_dict_path = out_fd / f'state_dict_{epoch_idx}.pt'
            logger.info(f'E={epoch_idx}, FOR NOW THE BEST, SAVING TO {state_dict_path}')

            restore_state = RestoreState(
                epoch_idx=epoch_idx,
                model_jit_state_dict=model_jit.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                optimizer_scheduler_state_dict=optimizer_scheduler.state_dict(),
            )
            torch.save(cattrs.unstructure(restore_state), state_dict_path)

        epoch_idx += 1


def build_model_jit_from_state_dict_path(
    state_dict_path: PathType,
    model_config_json: Optional[str] = None,
):
    model_config = dyn_structure(
        model_config_json,
        ModelConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('model_config:')
    logger.info(cattrs.unstructure(model_config))

    model = AdaptiveScaling(
        size=model_config.size,
        neck_head_type=model_config.neck_head_type,
        init_scale_output_bias=model_config.init_scale_output_bias,
    )
    model_jit: torch.jit.ScriptModule = torch.jit.script(model)  # type: ignore
    del model

    restore_state = dyn_structure(
        torch.load(state_dict_path, map_location='cpu'),
        RestoreState,
    )
    model_jit.load_state_dict(restore_state.model_jit_state_dict)
    model_jit.eval()

    return model_jit


def build_and_dump_model_jit_from_state_dict_path(
    state_dict_path: PathType,
    output_model_jit: PathType,
    model_config_json: Optional[str] = None,
):
    model_jit = build_model_jit_from_state_dict_path(
        state_dict_path=state_dict_path,
        model_config_json=model_config_json,
    )
    torch.jit.save(model_jit, output_model_jit)  # type: ignore
