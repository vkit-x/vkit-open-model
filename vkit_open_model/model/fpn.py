from typing import Sequence, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from . import helper


def build_conv1x1_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        helper.permute_bchw_to_bhwc(),
        helper.conv1x1(in_channels=in_channels, out_channels=out_channels),
        helper.ln(in_channels=out_channels),
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    )


def build_conv3x3_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        helper.conv3x3(in_channels=in_channels, out_channels=out_channels),
        helper.permute_bchw_to_bhwc(),
        helper.ln(in_channels=out_channels),
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    )


class FpnNeck(nn.Module):

    @staticmethod
    def build_step1_conv_blocks(
        in_channels_group: Sequence[int],
        out_channels: int,
    ):
        step1_conv_blocks: List[nn.Module] = []
        for in_channels in in_channels_group:
            step1_conv_blocks.append(
                build_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )
        return nn.ModuleList(step1_conv_blocks)

    @staticmethod
    def build_step2_conv_blocks(
        in_channels_group: Sequence[int],
        out_channels: int,
    ):
        assert out_channels % len(in_channels_group) == 0
        inner_channels = out_channels // len(in_channels_group)
        step2_conv_blocks: List[nn.Module] = []
        for _ in in_channels_group:
            step2_conv_blocks.append(
                build_conv3x3_block(
                    in_channels=out_channels,
                    out_channels=inner_channels,
                )
            )
        return nn.ModuleList(step2_conv_blocks)

    def __init__(
        self,
        in_channels_group: Sequence[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        assert len(in_channels_group) > 1
        self.step1_conv_blocks = self.build_step1_conv_blocks(
            in_channels_group=in_channels_group,
            out_channels=out_channels,
        )
        self.step2_conv_blocks = self.build_step2_conv_blocks(
            in_channels_group=in_channels_group,
            out_channels=out_channels,
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:  # type: ignore
        num_features = len(features)
        assert num_features == len(self.step1_conv_blocks)

        # Step 1.
        outputs = [
            step1_conv_block(features[feature_idx])
            for feature_idx, step1_conv_block in enumerate(self.step1_conv_blocks)
        ]

        # Upsampling & add to the previous layer.
        for feature_idx in range(num_features - 1, 0, -1):
            prev_feature_idx = feature_idx - 1
            prev_feature = outputs[prev_feature_idx]
            prev_shape = (prev_feature.shape[-2], prev_feature.shape[-1])
            outputs[prev_feature_idx] += F.interpolate(
                outputs[feature_idx],
                size=prev_shape,
                mode='nearest',
            )

        # Step 2.
        for feature_idx, step2_conv_block in enumerate(self.step2_conv_blocks):
            outputs[feature_idx] = step2_conv_block(outputs[feature_idx])

        # Final.
        feature0_shape: Tuple[int, int] = (features[0].shape[-2], features[0].shape[-1])
        for feature_idx in range(1, num_features):
            outputs[feature_idx] = F.interpolate(
                outputs[feature_idx],
                size=feature0_shape,
                mode='nearest',
            )
        # (B, out_channels, H, W)
        outputs_cat = torch.cat(outputs, dim=1)

        return outputs_cat


class FpnHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsampling_factor: int = 1,
        init_output_bias: float = 0.0,
    ):
        super().__init__()

        self.upsampling_factor = upsampling_factor

        inner_channels = (in_channels + out_channels) // 2
        self.step1_conv3x3 = build_conv3x3_block(
            in_channels=in_channels,
            out_channels=inner_channels,
        )
        self.step2_conv1x1 = nn.Sequential(
            helper.permute_bchw_to_bhwc(),
            helper.conv1x1(in_channels=inner_channels, out_channels=out_channels),
            helper.permute_bhwc_to_bchw(),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.constant_(self.step2_conv1x1[1].bias, init_output_bias)  # type: ignore

    def forward(self, fpn_neck_feature: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = fpn_neck_feature

        if self.upsampling_factor > 1:
            x = F.interpolate(
                x,
                size=(
                    x.shape[-2] * self.upsampling_factor,
                    x.shape[-1] * self.upsampling_factor,
                ),
                mode='nearest',
            )

        x = self.step1_conv3x3(x)
        x = self.step2_conv1x1(x)
        return x
