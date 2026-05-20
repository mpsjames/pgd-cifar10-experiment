from __future__ import annotations

import torch
from torch import nn

from src.models.normalizer import Normalizer


def test_normalize_wrapper_applies_expected_formula() -> None:
    inner = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    inner.weight.data.fill_(1.0)
    model = Normalizer(inner, mean=(0.5, 0.25, 0.0), std=(0.5, 0.25, 1.0))
    x = torch.ones(2, 3, 32, 32)
    out = model(x)
    expected = torch.full((2, 1, 32, 32), 5.0)
    assert torch.allclose(out, expected)


def test_normalize_wrapper_buffers_move_with_device() -> None:
    inner = nn.Identity()
    model = Normalizer(inner, mean=(0.1, 0.2, 0.3), std=(1.0, 1.0, 1.0))
    model = model.to(torch.device("cpu"))
    assert model.mean.device.type == "cpu"
    assert model.std.device.type == "cpu"
