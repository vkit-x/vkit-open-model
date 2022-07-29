import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
import iolite as io

from vkit.element import Box, Mask, ScoreMap, Image, Painter
from vkit_open_model.model.adaptive_scaling import (
    AdaptiveScaling,
    AdaptiveScalingSize,
    AdaptiveScalingNeckHeadType,
)
from vkit_open_model.dataset.adaptive_scaling import (
    adaptive_scaling_dataset_collate_fn,
    AdaptiveScalingIterableDataset,
)
from vkit_open_model.loss_function import AdaptiveScalingLossFunction


def test_adaptive_scaling_jit():
    model = AdaptiveScaling(AdaptiveScalingSize.TINY, AdaptiveScalingNeckHeadType.UPERNEXT)
    model_jit = torch.jit.script(model)  # type: ignore

    x = torch.rand((1, 3, 320, 320))
    mask_feature, scale_feature = model_jit(x)  # type: ignore
    assert mask_feature.shape == (1, 1, 160, 160)
    assert scale_feature.shape == (1, 1, 160, 160)


def debug_adaptive_scaling_jit_model_summary():
    print('AdaptiveScaling(BASE, FPN)')
    model = AdaptiveScaling(AdaptiveScalingSize.BASE, AdaptiveScalingNeckHeadType.FPN)
    model.eval()
    print('depth=1')
    summary(model, input_size=(1, 3, 640, 40), depth=1)
    print('depth=3')
    summary(model, input_size=(1, 3, 640, 40), depth=3)
    print()

    print('AdaptiveScaling(BASE, UPERNEXT)')
    model = AdaptiveScaling(AdaptiveScalingSize.BASE, AdaptiveScalingNeckHeadType.UPERNEXT)
    model.eval()
    print('depth=1')
    summary(model, input_size=(1, 3, 640, 40), depth=1)
    print('depth=3')
    summary(model, input_size=(1, 3, 640, 40), depth=3)
    print()


def test_adaptive_scaling_jit_loss_backward():
    model = AdaptiveScaling(AdaptiveScalingSize.TINY)
    model_jit = torch.jit.script(model)  # type: ignore
    del model

    loss_function = AdaptiveScalingLossFunction()

    with torch.autograd.profiler.profile() as prof:
        x = torch.rand((2, 3, 640, 640))
        mask_feature, scale_feature = model_jit(x)  # type: ignore
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))  # type: ignore

    loss = loss_function(
        mask_feature=mask_feature,
        scale_feature=scale_feature,
        downsampled_mask=(torch.rand(2, 300, 300) > 0.5).float(),
        downsampled_score_map=torch.rand(2, 300, 300) + 8.75,
        downsampled_shape=(320, 320),
        downsampled_core_box=Box(up=10, down=309, left=10, right=309),
    )
    loss.backward()


def sample_adaptive_scaling_dataset(
    num_processes: int,
    batch_size: int,
    epoch_size: int,
    output_folder: str,
):
    out_fd = io.folder(output_folder, touch=True)

    num_samples = batch_size * epoch_size
    dataset = AdaptiveScalingIterableDataset(
        steps_json=(
            '$VKIT_ARTIFACT_PACK/pipeline/text_detection/dev_adaptive_scaling_dataset_steps.json'
        ),
        num_samples=num_samples,
        rng_seed=13,
        num_processes=num_processes,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=adaptive_scaling_dataset_collate_fn,
    )

    for batch_idx, batch in enumerate(data_loader):
        print(f'Saving batch_idx={batch_idx} ...')
        batch_image = batch['image']
        batch_downsampled_mask = batch['downsampled_mask']
        batch_downsampled_score_map = batch['downsampled_score_map']
        downsampled_shape = batch['downsampled_shape']
        downsampled_core_box: Box = batch['downsampled_core_box']

        assert batch_image.shape[0] \
            == batch_downsampled_mask.shape[0] \
            == batch_downsampled_score_map.shape[0] \
            == batch_size

        for sample_in_batch_idx in range(batch_size):
            np_image = \
                torch.permute(batch_image[sample_in_batch_idx], (1, 2, 0)).numpy().astype(np.uint8)
            np_downsampled_mask = \
                batch_downsampled_mask[sample_in_batch_idx].numpy().astype(np.uint8)
            np_downsampled_score_map = \
                batch_downsampled_score_map[sample_in_batch_idx].numpy()

            image = Image(mat=np_image)
            downsampled_mask = Mask(
                mat=np_downsampled_mask,
                box=downsampled_core_box,
            )
            downsampled_score_map = ScoreMap(
                mat=np_downsampled_score_map,
                is_prob=False,
                box=downsampled_core_box,
            )

            output_prefix = f'{batch_idx}_{sample_in_batch_idx}'
            image.to_file(out_fd / f'{output_prefix}_image.jpg')

            downsampled_image = image.to_resized_image(
                resized_height=downsampled_shape[0],
                resized_width=downsampled_shape[1],
            )

            painter = Painter.create(downsampled_image)
            painter.paint_mask(downsampled_mask)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_mask.jpg')

            painter = Painter.create(downsampled_image)
            painter.paint_score_map(downsampled_score_map)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_score_map.jpg')

            painter = Painter.create(downsampled_image)
            painter.paint_score_map(downsampled_score_map, alpha=1.0)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_score_map_alpha_1.jpg')


def profile_adaptive_scaling_dataset(num_processes: int, batch_size: int, epoch_size: int):
    from datetime import datetime
    from tqdm import tqdm
    import numpy as np

    num_samples = batch_size * epoch_size

    data_loader = DataLoader(
        dataset=AdaptiveScalingIterableDataset(
            steps_json='$VKIT_ARTIFACT_PACK/pipeline/text_detection/adaptive_scaling.json',
            num_samples=num_samples,
            rng_seed=13370,
            num_processes=num_processes,
        ),
        batch_size=batch_size,
        collate_fn=adaptive_scaling_dataset_collate_fn,
    )

    dt_batches = []
    dt_begin = datetime.now()
    for _ in tqdm(data_loader):
        dt_batches.append(datetime.now())
    dt_end = datetime.now()

    dt_delta = dt_end - dt_begin
    print('total:', dt_delta.seconds)
    print('per_batch:', dt_delta.seconds / batch_size)
    dt_batch_deltas = []
    for idx, dt_batch in enumerate(dt_batches):
        if idx == 0:
            dt_prev = dt_begin
        else:
            dt_prev = dt_batches[idx - 1]
        dt_batch_deltas.append((dt_batch - dt_prev).seconds)
    print('per_batch std:', float(np.std(dt_batch_deltas)))
