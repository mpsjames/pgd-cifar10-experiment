"""Shared script loading helpers for checkpoints and smoke data."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.experiments.checkpoint_paths import variant_checkpoint_path
from src.experiments.config import ModelConfig
from src.models.builders import build_normalized_model, load_model_from_checkpoint


def load_checkpoint_or_smoke(
    *,
    arch: str,
    seed: int,
    variant: str,
    model_config: ModelConfig,
    smoke: bool,
    checkpoint: Path | None = None,
):
    path = checkpoint or variant_checkpoint_path(arch, seed, variant)
    if path.exists():
        return load_model_from_checkpoint(model_config, path)
    if smoke:
        print(f"WARNING: smoke run on random weights; checkpoint not found: {path}")
        return build_normalized_model(model_config)
    raise FileNotFoundError(path)


def cifar_test_loader(batch_size: int, no_download: bool) -> DataLoader:
    from src.data.cifar10 import get_cifar10_loaders

    _, loader = get_cifar10_loaders(batch_size, download=not no_download)
    return loader
