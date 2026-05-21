"""Resolve configured runtime devices for experiment services."""

from __future__ import annotations

import torch

from src.experiments.config import ExperimentConfig


def resolve_configured_device(config: ExperimentConfig) -> torch.device:
    """Return the device requested by config, failing fast when unavailable."""
    requested = config.hardware.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Configured device 'cuda' is unavailable; use --hardware cpu to run on CPU."
            )
        return torch.device("cuda")
    raise ValueError(f"Unsupported hardware.device: {requested!r}")
