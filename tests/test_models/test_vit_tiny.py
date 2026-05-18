from __future__ import annotations

import torch

from src.experiments.config_loader import load_experiment_config
from src.models.builders import build_model


def test_vit_tiny_uses_yaml_hyperparameters() -> None:
    config = load_experiment_config(arch="vit_tiny").model
    model = build_model(config).eval()
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, config.num_classes)
    assert config.vit is not None
    assert config.vit.patch_size == 4
    assert config.vit.embed_dim == 192
