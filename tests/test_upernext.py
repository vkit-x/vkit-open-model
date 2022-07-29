import torch
from vkit_open_model.model.upernext import UperNextNeck


def test_upernext():
    model = UperNextNeck(
        in_channels_group=(96, 192, 384, 768),
        out_channels=384,
    )
    features = [
        torch.rand(1, 96, 80, 80),
        torch.rand(1, 192, 40, 40),
        torch.rand(1, 384, 20, 20),
        torch.rand(1, 768, 10, 10),
    ]
    output = model(features)
    assert output.shape == (1, 384, 80, 80)

    model_jit = torch.jit.script(model)  # type: ignore
    assert model_jit
