import torch
from vkit_open_model.model.fpn import FpnNeck, FpnHead


def test_fpn():
    neck = FpnNeck(
        in_channels_group=(96, 192, 384, 768),
        out_channels=400,
    )
    features = [
        torch.rand(1, 96, 80, 80),
        torch.rand(1, 192, 40, 40),
        torch.rand(1, 384, 20, 20),
        torch.rand(1, 768, 10, 10),
    ]
    neck_output = neck(features)
    assert neck_output.shape == (1, 400, 80, 80)

    model_jit = torch.jit.script(neck)  # type: ignore
    assert model_jit

    head = FpnHead(
        in_channels=400,
        out_channels=1,
        upsampling_factor=1,
    )
    head_output = head(neck_output)
    assert head_output.shape == (1, 1, 80, 80)

    head = FpnHead(
        in_channels=400,
        out_channels=1,
        upsampling_factor=2,
    )
    head_output = head(neck_output)
    assert head_output.shape == (1, 1, 160, 160)

    model_jit = torch.jit.script(head)  # type: ignore
    assert model_jit
