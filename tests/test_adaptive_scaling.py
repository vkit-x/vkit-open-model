import torch
from torch.utils.data import DataLoader

from vkit_open_model.model.adaptive_scaling import AdaptiveScaling
from vkit_open_model.dataset.adaptive_scaling import (
    adaptive_scaling_dataset_collate_fn,
    AdaptiveScalingIterableDataset,
)


def test_adaptive_scaling_jit():
    model = AdaptiveScaling.create_tiny()
    model_jit = torch.jit.script(model)  # type: ignore

    x = torch.rand((1, 3, 320, 320))
    mask_feature, scale_feature = model_jit(x)  # type: ignore
    assert mask_feature.shape == (1, 1, 160, 160)
    assert scale_feature.shape == (1, 1, 160, 160)


def profile_adaptive_scaling_dataset(num_workers: int, batch_size: int, epoch_size: int):
    from datetime import datetime
    from tqdm import tqdm
    import numpy as np

    num_steps = batch_size * epoch_size
    rnd_seed = list(range(num_steps))

    data_loader = DataLoader(
        dataset=AdaptiveScalingIterableDataset(
            steps_json='$VKIT_ARTIFACT_PACK/pipeline/text_detection/adaptive_scaling.json',
            num_steps=num_steps,
            rnd_seed=rnd_seed,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=adaptive_scaling_dataset_collate_fn,
        persistent_workers=True,
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
