# This project (vkit-x/vkit-open-model) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import Tuple, Optional

import torch
from vkit.element import Box
import attrs

from .weighted_bce_with_logits import WeightedBceWithLogitsLossFunction
from .focal_with_logits import FocalWithLogitsLossFunction
from .dice import DiceLossFunction
from .l1 import L1LossFunction
from .l2 import L2LossFunction
from .weight_adaptive_heatmap_regression import WeightAdaptiveHeatmapRegressionLossFunction
from .cross_entropy_with_logits import CrossEntropyWithLogitsLossFunction


@attrs.define
class AdaptiveScalingRoughLossFunctionConifg:
    bce_negative_ratio: float = 3.0
    bce_factor: float = 0.0
    focal_factor: float = 5.0
    dice_factor: float = 1.0
    l1_factor: float = 1.0
    downsampled_score_map_min: float = 1.1
    char_height_feature_min: float = 1.1


class AdaptiveScalingRoughLossFunction:

    def __init__(self, config: AdaptiveScalingRoughLossFunctionConifg):
        self.config = config

        # Mask.
        self.weighted_bce_with_logits = WeightedBceWithLogitsLossFunction(
            negative_ratio=config.bce_negative_ratio,
        )
        self.focal_with_logits = FocalWithLogitsLossFunction()
        self.dice = DiceLossFunction()

        # Scale.
        self.l1 = L1LossFunction(smooth=True)

    def __call__(
        self,
        # Model predictions.
        # (B, 1, H, W)
        rough_char_mask_feature: torch.Tensor,
        rough_char_height_feature: torch.Tensor,
        # Ground truths.
        # (B, CH, CW)
        downsampled_mask: torch.Tensor,
        downsampled_score_map: torch.Tensor,
        downsampled_shape: Tuple[int, int],
        downsampled_core_box: Box,
    ) -> torch.Tensor:
        # (B, 1, H, W)
        assert rough_char_mask_feature.shape == rough_char_height_feature.shape
        assert rough_char_mask_feature.shape[1:] == (1, *downsampled_shape)

        # (B, H, W)
        rough_char_mask_feature = torch.squeeze(rough_char_mask_feature, dim=1)
        rough_char_height_feature = torch.squeeze(rough_char_height_feature, dim=1)

        # (B, CH, CW)
        dc_box = downsampled_core_box

        rough_char_mask_feature = rough_char_mask_feature[
            :,
            dc_box.up:dc_box.down + 1,
            dc_box.left:dc_box.right + 1
        ]  # yapf: disable
        rough_char_height_feature = rough_char_height_feature[
            :,
            dc_box.up:dc_box.down + 1,
            dc_box.left:dc_box.right + 1
        ]  # yapf: disable

        loss = 0.0

        # Mask.
        if self.config.bce_factor > 0.0:
            loss += self.config.bce_factor * self.weighted_bce_with_logits(
                pred=rough_char_mask_feature,
                gt=downsampled_mask,
            )

        if self.config.focal_factor > 0.0:
            loss += self.config.focal_factor * self.focal_with_logits(
                pred=rough_char_mask_feature,
                gt=downsampled_mask,
            )

        if self.config.dice_factor > 0.0:
            loss += self.config.dice_factor * self.dice(
                pred=torch.sigmoid(rough_char_mask_feature),
                gt=downsampled_mask,
            )

        # Scale.
        if self.config.l1_factor > 0.0:
            # NOTE: critical mask!
            l1_mask = ((rough_char_height_feature > self.config.char_height_feature_min)
                       & (downsampled_score_map > self.config.downsampled_score_map_min)
                       & downsampled_mask.bool()).float()
            rough_char_height_feature = torch.clamp(
                rough_char_height_feature,
                min=self.config.char_height_feature_min,
            )
            downsampled_score_map = torch.clamp(
                downsampled_score_map,
                min=self.config.downsampled_score_map_min,
            )
            # Log space + smooth L1 to model the relative scale difference.
            loss += self.config.l1_factor * self.l1(
                pred=torch.log(rough_char_height_feature),
                gt=torch.log(downsampled_score_map),
                mask=l1_mask,
            )

        assert not isinstance(loss, float)
        return loss


@attrs.define
class AdaptiveScalingPreciseLossFunctionConifg:
    char_mask_focal_factor: float = 0.0
    char_prob_l1_factor: float = 0.0
    char_prob_pos_l2_factor: float = 2.0
    char_prob_neg_l2_factor: float = 1.0
    char_prob_wahr_factor: float = 0.0
    char_up_left_offset_l1_factor: float = 1.0
    char_up_left_distance_regulation_l1_factor: float = 1.0
    char_corner_angle_cross_entropy_factor: float = 5.0
    char_corner_distance_l1_factor: float = 1.0
    loss_factor: float = 0.15


class AdaptiveScalingPreciseLossFunction:

    def __init__(self, config: AdaptiveScalingPreciseLossFunctionConifg):
        self.config = config

        # Mask.
        self.char_mask_focal_with_logits = FocalWithLogitsLossFunction()
        # Prob.
        self.char_prob_l1 = L1LossFunction(smooth=True, smooth_beta=0.25)
        self.char_prob_l2 = L2LossFunction()
        self.char_prob_wahr = WeightAdaptiveHeatmapRegressionLossFunction()
        # Up-left corner.
        self.char_up_left_offset_l1 = L1LossFunction(smooth=True, smooth_beta=2.5)
        self.char_up_left_distance_regulation_l1 = L1LossFunction(smooth=True, smooth_beta=2.5)
        # Corner angle.
        self.char_corner_angle_cross_entropy = CrossEntropyWithLogitsLossFunction()
        # Corner distance.
        self.char_corner_distance_l1 = L1LossFunction(smooth=True, smooth_beta=2.5)

    @classmethod
    def get_label_point_feature(
        cls,
        # (B, *, H, W)
        feature: torch.Tensor,
        # (B, P)
        label_point_y: torch.Tensor,
        label_point_x: torch.Tensor,
    ):
        batch_size = feature.shape[0]
        assert batch_size == label_point_y.shape[0] == label_point_x.shape[0]
        # (B, P, *)
        return feature[torch.arange(batch_size)[:, None], :, label_point_y, label_point_x]

    def __call__(
        self,
        # Model predictions.
        # (B, 1, H, W)
        precise_char_mask_feature: Optional[torch.Tensor],
        precise_char_prob_feature: torch.Tensor,
        # (B, 2, H, W)
        precise_char_up_left_corner_offset_feature: torch.Tensor,
        # (B, 4, H, W)
        precise_char_corner_angle_feature: torch.Tensor,
        # (B, 4, H, W)
        precise_char_corner_distance_feature: torch.Tensor,
        # Ground truths.
        # (B, CH, CW)
        downsampled_char_prob_score_map: torch.Tensor,
        downsampled_char_mask: torch.Tensor,
        downsampled_shape: Tuple[int, int],
        downsampled_core_box: Box,
        # Label points.
        # (B, P)
        downsampled_label_point_y: torch.Tensor,
        downsampled_label_point_x: torch.Tensor,
        # (B, P, 2)
        char_up_left_offsets: torch.Tensor,
        # (B, P, 4)
        char_corner_angles: torch.Tensor,
        # (B, P, 3)
        char_corner_distances: torch.Tensor,
    ) -> torch.Tensor:
        # Prob.
        # (B, H, W)
        if precise_char_mask_feature is not None:
            assert precise_char_mask_feature.shape == precise_char_prob_feature.shape
        assert precise_char_prob_feature.shape[1:] == (1, *downsampled_shape)

        if precise_char_mask_feature is not None:
            precise_char_mask_feature = torch.squeeze(precise_char_mask_feature, dim=1)
        precise_char_prob_feature = torch.squeeze(precise_char_prob_feature, dim=1)

        # (B, CH, CW)
        if precise_char_mask_feature is not None:
            precise_char_mask_feature = precise_char_mask_feature[
                :,
                downsampled_core_box.up:downsampled_core_box.down + 1,
                downsampled_core_box.left:downsampled_core_box.right + 1
            ]  # yapf: disable
        precise_char_prob_feature = precise_char_prob_feature[
            :,
            downsampled_core_box.up:downsampled_core_box.down + 1,
            downsampled_core_box.left:downsampled_core_box.right + 1
        ]  # yapf: disable

        # Up-left corner.
        # (B, P, 2)
        precise_char_up_left_corner_offset_label_point_feature = self.get_label_point_feature(
            feature=precise_char_up_left_corner_offset_feature,
            label_point_y=downsampled_label_point_y,
            label_point_x=downsampled_label_point_x,
        )

        # Corner angle.
        # (B, P, 4)
        precise_char_corner_angle_label_point_feature = self.get_label_point_feature(
            feature=precise_char_corner_angle_feature,
            label_point_y=downsampled_label_point_y,
            label_point_x=downsampled_label_point_x,
        )
        # (B, 4, P), required by torch.nn.functional.cross_entropy.
        precise_char_corner_angle_label_point_feature = \
            precise_char_corner_angle_label_point_feature.transpose(1, 2)
        char_corner_angles = char_corner_angles.transpose(1, 2)

        # Corner distance.
        # (B, P, 4)
        precise_char_corner_distance_label_point_feature = self.get_label_point_feature(
            # Trim the up-left corner distance.
            feature=precise_char_corner_distance_feature,
            label_point_y=downsampled_label_point_y,
            label_point_x=downsampled_label_point_x,
        )
        # Up-left trimmed.
        # (B, P, 3)
        precise_char_up_left_trimmed_corner_distance_label_point_feature = \
            precise_char_corner_distance_label_point_feature[:, :, 1:]
        # Up-left only.
        # (B, P, 1)
        precise_char_up_left_only_corner_distance_label_point_feature = \
            precise_char_corner_distance_label_point_feature[:, :, 0:1]

        loss = 0.0

        if self.config.char_mask_focal_factor > 0:
            assert precise_char_mask_feature is not None
            loss += self.config.char_mask_focal_factor * self.char_mask_focal_with_logits(
                pred=precise_char_mask_feature,
                gt=downsampled_char_mask,
            )

        if self.config.char_prob_l1_factor > 0 \
                or self.config.char_prob_pos_l2_factor > 0 \
                or self.config.char_prob_neg_l2_factor > 0 \
                or self.config.char_prob_wahr_factor > 0:
            precise_char_prob_feature_sigmoid = torch.sigmoid(precise_char_prob_feature)
            if self.config.char_prob_l1_factor > 0:
                loss += self.config.char_prob_l1_factor * self.char_prob_l1(
                    pred=precise_char_prob_feature_sigmoid,
                    gt=downsampled_char_prob_score_map,
                    mask=downsampled_char_mask,
                )
            if self.config.char_prob_pos_l2_factor > 0:
                loss += self.config.char_prob_pos_l2_factor * self.char_prob_l2(
                    pred=precise_char_prob_feature_sigmoid,
                    gt=downsampled_char_prob_score_map,
                    mask=downsampled_char_mask,
                )
            if self.config.char_prob_neg_l2_factor > 0:
                loss += self.config.char_prob_neg_l2_factor * self.char_prob_l2(
                    pred=precise_char_prob_feature_sigmoid,
                    gt=downsampled_char_prob_score_map,
                    # NOTE: negative mask here.
                    mask=(1 - downsampled_char_mask),
                )
            if self.config.char_prob_wahr_factor > 0:
                loss += self.config.char_prob_wahr_factor * self.char_prob_wahr(
                    pred=precise_char_prob_feature_sigmoid,
                    gt=downsampled_char_prob_score_map,
                )

        if self.config.char_up_left_offset_l1_factor > 0:
            loss += self.config.char_up_left_offset_l1_factor * self.char_up_left_offset_l1(
                pred=precise_char_up_left_corner_offset_label_point_feature,
                gt=char_up_left_offsets,
            )

        if self.config.char_up_left_distance_regulation_l1_factor > 0:
            factor = self.config.char_up_left_distance_regulation_l1_factor
            loss += factor * self.char_up_left_distance_regulation_l1(
                pred=torch.linalg.norm(
                    precise_char_up_left_corner_offset_label_point_feature,
                    dim=2,
                ),
                gt=torch.squeeze(
                    precise_char_up_left_only_corner_distance_label_point_feature,
                    dim=2,
                )
            )

        if self.config.char_corner_angle_cross_entropy_factor > 0:
            factor = self.config.char_corner_angle_cross_entropy_factor
            loss += factor * self.char_corner_angle_cross_entropy(
                pred=precise_char_corner_angle_label_point_feature,
                gt=char_corner_angles,
            )

        if self.config.char_corner_distance_l1_factor > 0:
            loss += self.config.char_corner_distance_l1_factor * self.char_corner_distance_l1(
                pred=precise_char_up_left_trimmed_corner_distance_label_point_feature,
                gt=char_corner_distances,
            )

        assert not isinstance(loss, float)

        # Balance gradiant.
        loss *= self.config.loss_factor

        return loss
