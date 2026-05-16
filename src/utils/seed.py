"""Seed every relevant RNG used by training, evaluation, and notebooks."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs and enable deterministic cudnn mode.

    Args:
        seed: Positive non-zero seed used for every RNG source.

    Raises:
        ValueError: When `seed == 0`, which the project forbids by convention.
    """
    if seed == 0:
        raise ValueError("seed=0 is disallowed by project reproducibility rules")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_generator(seed: int) -> torch.Generator:
    """Return a torch generator seeded for deterministic DataLoader shuffling.

    Args:
        seed: Positive non-zero seed.

    Returns:
        `torch.Generator` with the requested seed applied.

    Raises:
        ValueError: When `seed == 0`.
    """
    if seed == 0:
        raise ValueError("seed=0 is disallowed by project reproducibility rules")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
