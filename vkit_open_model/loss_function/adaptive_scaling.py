from typing import Tuple

import torch
from vkit.element import Box

from .weighted_bce_with_logits import WeightedBceWithLogitsLossFunction
from .dice import DiceLossFunction
from .l1 import L1LossFunction


class AdaptiveScalingLossFunction:

    def __init__(
        self,
        negative_ratio: float = 3.0,
        bce_factor: float = 2.0,
        dice_factor: float = 1.0,
        l1_factor: float = 1.0,
    ):
        # Mask.
        self.bce_factor = bce_factor
        self.weighted_bce_with_logits = WeightedBceWithLogitsLossFunction(
            negative_ratio=negative_ratio,
        )

        self.dice_factor = dice_factor
        self.dice = DiceLossFunction()

        # Scale.
        self.l1_factor = l1_factor
        self.l1 = L1LossFunction(smooth=True)

    def __call__(
        self,
        mask_feature: torch.Tensor,
        scale_feature: torch.Tensor,
        downsampled_mask: torch.Tensor,
        downsampled_score_map: torch.Tensor,
        downsampled_shape: Tuple[int, int],
        downsampled_core_box: Box,
    ) -> torch.Tensor:
        # (B, 1, H, W)
        assert mask_feature.shape == scale_feature.shape
        assert mask_feature.shape[1:] == (1, *downsampled_shape)

        # (B, H, W)
        mask_feature = torch.squeeze(mask_feature, dim=1)
        scale_feature = torch.squeeze(scale_feature, dim=1)

        # (B, DH, DW)
        dc_box = downsampled_core_box
        mask_feature = mask_feature[:, dc_box.up:dc_box.down + 1, dc_box.left:dc_box.right + 1]
        scale_feature = scale_feature[:, dc_box.up:dc_box.down + 1, dc_box.left:dc_box.right + 1]

        # Mask.
        loss = self.bce_factor * self.weighted_bce_with_logits(
            pred=mask_feature,
            gt=downsampled_mask,
        )
        loss += self.dice_factor * self.dice(
            pred=torch.sigmoid(mask_feature),
            gt=downsampled_mask,
        )

        # Scale.
        # Clamp min to 1.1 to avoid torch.log nan.
        scale_min = 1.1
        l1_mask = ((scale_feature > scale_min) & (downsampled_score_map > scale_min)
                   & downsampled_mask.bool()).float()
        scale_feature = torch.clamp(scale_feature, min=scale_min)
        downsampled_score_map = torch.clamp(downsampled_score_map, min=scale_min)
        # Log space + smooth L1 to model the relative scale difference.
        loss += self.l1_factor * self.l1(
            pred=torch.log(scale_feature),
            gt=torch.log(downsampled_score_map),
            mask=l1_mask,
        )

        return loss
