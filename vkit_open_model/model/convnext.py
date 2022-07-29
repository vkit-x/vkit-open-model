from typing import Sequence, Tuple, List, Optional

import torch
from torch import nn

from . import helper


class ConvNextBlockLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        prob_bypass: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            helper.dconv7x7(in_channels=in_channels),
            helper.permute_bchw_to_bhwc(),
            helper.ln(in_channels=in_channels),
            helper.conv1x1(in_channels=in_channels, out_channels=4 * in_channels),
            helper.gelu(),
            helper.conv1x1(in_channels=4 * in_channels, out_channels=in_channels),
            helper.permute_bhwc_to_bchw(),
        )
        self.block_scale = nn.parameter.Parameter(torch.ones(in_channels, 1, 1) * 1E-6)
        self.prob_bypass = prob_bypass

    def apply_stochastic_depth(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob_bypass == 0.0:
            return x

        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device)

        prob_keep = 1.0 - self.prob_bypass
        mask.bernoulli_(prob_keep)
        if prob_keep > 0.0:
            mask.div_(prob_keep)

        return mask * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self.block_scale * self.block(x)
        output = self.apply_stochastic_depth(output)
        output += x
        return output


class ConvNextBlock(nn.Module):

    def __init__(
        self,
        layer_idx_begin: int,
        layer_idx_end: int,
        in_channels: int,
        num_layers: int,
        out_channels: Optional[int],
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        for layer_idx in range(num_layers):
            prob_bypass = 0.1 * (layer_idx_begin + layer_idx) / layer_idx_end
            layers.append(ConvNextBlockLayer(
                in_channels=in_channels,
                prob_bypass=prob_bypass,
            ))
        self.layers = nn.Sequential(*layers)

        self.ln = nn.Sequential(
            helper.permute_bchw_to_bhwc(),
            helper.ln(in_channels=in_channels),
            helper.permute_bhwc_to_bchw(),
        )

        self.pconv2x2: Optional[nn.Module] = None
        if out_channels:
            self.pconv2x2 = helper.pconv2x2(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        x = self.layers(x)
        x = self.ln(x)
        feature = x

        if self.pconv2x2 is not None:
            x = self.pconv2x2(x)

        return feature, x


class ConvNext(nn.Module):

    @staticmethod
    def build_stem(
        stem_in_channels: int,
        block_in_channels: int,
        use_pconv2x2: bool,
    ):
        if not use_pconv2x2:
            pconv = helper.pconv4x4(in_channels=stem_in_channels, out_channels=block_in_channels)
        else:
            pconv = helper.pconv2x2(in_channels=stem_in_channels, out_channels=block_in_channels)

        return nn.Sequential(
            pconv,
            helper.permute_bchw_to_bhwc(),
            helper.ln(in_channels=block_in_channels),
            helper.permute_bhwc_to_bchw(),
        )

    @staticmethod
    def build_blocks(block_in_channels_and_num_layers: Sequence[Tuple[int, int]]):
        num_layers_sum = sum(num_layers for _, num_layers in block_in_channels_and_num_layers)
        layer_idx_begin = 0
        layer_idx_end = num_layers_sum - 1

        blocks: List[ConvNextBlock] = []
        in_channels_group: List[int] = []
        for block_idx, (in_channels, num_layers) in enumerate(block_in_channels_and_num_layers):
            if block_idx + 1 < len(block_in_channels_and_num_layers):
                out_channels = block_in_channels_and_num_layers[block_idx + 1][0]
            else:
                out_channels = None
            blocks.append(
                ConvNextBlock(
                    layer_idx_begin=layer_idx_begin,
                    layer_idx_end=layer_idx_end,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    out_channels=out_channels,
                )
            )
            in_channels_group.append(in_channels)
            layer_idx_begin += num_layers
        return nn.ModuleList(blocks), in_channels_group

    def __init__(
        self,
        stem_in_channels: int,
        block_in_channels_and_num_layers: Sequence[Tuple[int, int]],
        stem_use_pconv2x2: bool,
    ):
        super().__init__()

        self.stem = self.build_stem(
            stem_in_channels=stem_in_channels,
            block_in_channels=block_in_channels_and_num_layers[0][0],
            use_pconv2x2=stem_use_pconv2x2,
        )
        self.blocks, self.in_channels_group = self.build_blocks(block_in_channels_and_num_layers)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def create_tiny(stem_use_pconv2x2: bool = False):
        return ConvNext(
            stem_in_channels=3,
            block_in_channels_and_num_layers=(
                (96, 3),
                (192, 3),
                (384, 9),
                (768, 3),
            ),
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    @staticmethod
    def create_small(stem_use_pconv2x2: bool = False):
        return ConvNext(
            stem_in_channels=3,
            block_in_channels_and_num_layers=(
                (96, 3),
                (192, 3),
                (384, 27),
                (768, 3),
            ),
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    @staticmethod
    def create_base(stem_use_pconv2x2: bool = False):
        return ConvNext(
            stem_in_channels=3,
            block_in_channels_and_num_layers=(
                (128, 3),
                (256, 3),
                (512, 27),
                (1024, 3),
            ),
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    @staticmethod
    def create_large(stem_use_pconv2x2: bool = False):
        return ConvNext(
            stem_in_channels=3,
            block_in_channels_and_num_layers=(
                (192, 3),
                (384, 3),
                (768, 27),
                (1536, 3),
            ),
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        features: List[torch.Tensor] = []

        x = self.stem(x)
        for block in self.blocks:
            feature, x = block(x)
            features.append(feature)

        return features
