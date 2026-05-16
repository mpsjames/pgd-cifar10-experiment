from __future__ import annotations

import torch

from src.models.builders import ARCH_BUILDERS


def test_all_4_builders_forward_pass() -> None:
    x = torch.rand(2, 3, 32, 32)
    for arch, builder in ARCH_BUILDERS.items():
        model = builder(10).eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10), arch


def test_all_4_builders_param_count_in_range() -> None:
    expected_ranges = {
        "resnet18": (10_000_000, 13_000_000),
        "wrn_34_10": (40_000_000, 50_000_000),
        "resnet50": (20_000_000, 27_000_000),
        "vgg16_bn": (14_000_000, 16_000_000),
    }
    for arch, builder in ARCH_BUILDERS.items():
        model = builder(10)
        n_params = sum(param.numel() for param in model.parameters())
        low, high = expected_ranges[arch]
        assert low <= n_params <= high, f"{arch}: {n_params}"
