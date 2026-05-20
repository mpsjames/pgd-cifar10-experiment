"""Synthetic smoke-test data loader."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_smoke_loader(batch_size: int, num_classes: int = 10) -> DataLoader:
    n = min(batch_size, 8)
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, num_classes, (n,), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=n, num_workers=0)
