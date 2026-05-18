from __future__ import annotations

import torch

from src.experiments.config import ModelConfig, ViTConfig, WRNConfig
from src.models.builders import ARCH_BUILDERS


def _default_model_config(arch: str) -> ModelConfig:
    vit = ViTConfig() if arch == "vit_tiny" else None
    wrn = WRNConfig() if arch == "wrn_34_10" else None
    return ModelConfig(arch=arch, checkpoint_path=None, vit=vit, wrn=wrn)  # type: ignore[arg-type]


def test_all_builders_forward_pass() -> None:
    x = torch.rand(2, 3, 32, 32)
    for arch, builder in ARCH_BUILDERS.items():
        model = builder(_default_model_config(arch)).eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10), arch


def test_all_builders_param_count_in_range() -> None:
    expected_ranges = {
        "resnet18": (10_000_000, 13_000_000),
        "wrn_34_10": (40_000_000, 50_000_000),
        "vit_tiny": (4_000_000, 8_000_000),
    }
    for arch, builder in ARCH_BUILDERS.items():
        model = builder(_default_model_config(arch))
        n_params = sum(param.numel() for param in model.parameters())
        low, high = expected_ranges[arch]
        assert low <= n_params <= high, f"{arch}: {n_params}"
