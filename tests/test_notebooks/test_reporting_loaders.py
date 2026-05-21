from __future__ import annotations

import sys
from types import ModuleType

import torch

from src.reporting.loaders import evaluation_loader, synthetic_batch


def test_synthetic_batch_is_seed_deterministic() -> None:
    x1, y1 = synthetic_batch(4, seed=7)
    x2, y2 = synthetic_batch(4, seed=7)
    x3, y3 = synthetic_batch(4, seed=8)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert not (torch.equal(x1, x3) and torch.equal(y1, y3))


def test_evaluation_loader_synthetic_fallback_uses_requested_seed(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("missing CIFAR-10")

    fake_cifar10 = ModuleType("src.data.cifar10")
    fake_cifar10.get_cifar10_loaders = unavailable
    monkeypatch.setitem(sys.modules, "src.data.cifar10", fake_cifar10)

    x1, y1 = next(iter(evaluation_loader(sample_size=None, seed=11)))
    x2, y2 = next(iter(evaluation_loader(sample_size=None, seed=11)))
    x3, y3 = next(iter(evaluation_loader(sample_size=None, seed=12)))

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert not (torch.equal(x1, x3) and torch.equal(y1, y3))
