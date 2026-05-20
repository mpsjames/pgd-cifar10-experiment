"""Seed every relevant RNG used by training, evaluation, and notebooks."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(
    seed: int,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
) -> None:
    """Seed Python, NumPy, and torch RNGs and configure cudnn for the run.

    The `deterministic`/`benchmark` knobs let hardware presets trade some
    bitwise reproducibility for throughput. The single-seed project workflow
    tolerates the trade-off when paired with a fixed seed and consistent
    hardware: small kernel-selection nondeterminism is acceptable when only
    one run per architecture is reported.

    Args:
        seed: Positive non-zero seed used for every RNG source.
        deterministic: When True, set `torch.backends.cudnn.deterministic`
            and request deterministic algorithms via
            `torch.use_deterministic_algorithms`. When False, allow PyTorch
            to pick faster nondeterministic kernels.
        benchmark: When True, enable the cuDNN autotuner
            (`torch.backends.cudnn.benchmark = True`) so convolution kernels
            are selected to maximize throughput for fixed input sizes.

    Raises:
        ValueError: When `seed == 0`, which the project forbids by convention.
    """
    if seed == 0:
        raise ValueError("seed=0 is disallowed by project reproducibility rules")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


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
