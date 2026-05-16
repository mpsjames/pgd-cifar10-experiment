"""Describe small experiment-level policies around adversarial training."""

from __future__ import annotations

from src.experiments.config import TrainingConfig


def requires_single_seed_disclosure(config: TrainingConfig) -> bool:
    """Report whether notebook results must disclose single-seed training.

    Args:
        config: Training configuration under consideration.

    Returns:
        True when the training mode is adversarial, which the project reports
        as `n=1` per plan §6.
    """
    return config.mode == "adversarial"
