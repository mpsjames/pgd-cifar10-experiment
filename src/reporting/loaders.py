"""Evaluation data helpers for reporting."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.reporting.constants import SEED, SMOKE_SAMPLE_SIZE

LOGGER = logging.getLogger(__name__)


def evaluation_inputs(sample_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed-seed evaluation batch, warning if synthetic fallback is used."""
    try:
        from src.data.cifar10 import get_cifar10_loaders

        _, test = get_cifar10_loaders(
            batch_size=sample_size, num_workers=0, seed=SEED, download=False
        )
        return next(iter(test))
    except (RuntimeError, OSError) as exc:
        LOGGER.warning(
            "CIFAR-10 dataset unavailable (%s: %s); using synthetic smoke batch",
            type(exc).__name__,
            exc,
        )
        return synthetic_batch(min(sample_size, SMOKE_SAMPLE_SIZE))


def synthetic_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,), dtype=torch.long)
    return x, y


def evaluation_loader(sample_size: int | None, seed: int) -> DataLoader:
    """Build a CIFAR-10 evaluation loader, warning if synthetic fallback is used."""
    try:
        from src.data.cifar10 import get_cifar10_loaders

        batch = 128 if torch.cuda.is_available() else 32
        _, test = get_cifar10_loaders(batch_size=batch, num_workers=0, seed=seed, download=False)
        if sample_size is None:
            return test
        subset_x, subset_y = [], []
        seen = 0
        for xb, yb in test:
            take = min(sample_size - seen, xb.size(0))
            subset_x.append(xb[:take])
            subset_y.append(yb[:take])
            seen += take
            if seen >= sample_size:
                break
        x = torch.cat(subset_x)
        y = torch.cat(subset_y)
        return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False, num_workers=0)
    except (RuntimeError, OSError) as exc:
        LOGGER.warning(
            "CIFAR-10 dataset unavailable (%s: %s); using synthetic smoke loader",
            type(exc).__name__,
            exc,
        )
        x, y = synthetic_batch(SMOKE_SAMPLE_SIZE)
        return DataLoader(TensorDataset(x, y), batch_size=SMOKE_SAMPLE_SIZE, num_workers=0)
