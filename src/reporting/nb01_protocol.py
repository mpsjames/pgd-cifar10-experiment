"""Notebook helper for protocol invariants (NB01)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

from src.experiments.config_loader import load_experiment_config
from src.reporting.constants import SEED


def nb01_protocol() -> dict[str, object]:
    """Validate protocol-level invariants and expose them to NB01."""
    config = load_experiment_config()
    try:
        config.seed = 7  # type: ignore[misc]
    except FrozenInstanceError:
        frozen = True
    else:
        frozen = False
    assert frozen, "ExperimentConfig must be frozen"
    return {
        "frozen_configs": frozen,
        "seed": SEED,
        "default_epsilon": config.attack.epsilon if config.attack else None,
    }
