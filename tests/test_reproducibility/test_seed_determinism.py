from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.tracking.env import log_environment
from src.utils.seed import get_generator, set_all_seeds


def test_same_seed_yields_identical_random_numbers() -> None:
    set_all_seeds(42)
    values_a = (random.random(), np.random.rand(), torch.rand(3))
    set_all_seeds(42)
    values_b = (random.random(), np.random.rand(), torch.rand(3))

    assert values_a[0] == values_b[0]
    assert values_a[1] == values_b[1]
    assert torch.equal(values_a[2], values_b[2])


def test_different_seed_yields_different_random_numbers() -> None:
    set_all_seeds(42)
    value_a = torch.rand(3)
    set_all_seeds(123)
    value_b = torch.rand(3)
    assert not torch.equal(value_a, value_b)


def test_generator_is_deterministic_per_seed() -> None:
    a = torch.rand(4, generator=get_generator(42))
    b = torch.rand(4, generator=get_generator(42))
    assert torch.equal(a, b)


def test_seed_zero_rejected() -> None:
    with pytest.raises(ValueError, match="seed=0"):
        set_all_seeds(0)


def test_log_environment_contains_git_metadata(repo_root) -> None:
    env = log_environment(repo_root)
    assert "git_commit" in env
    assert "git_dirty" in env
    assert env["git_available"] is True


def test_amp_does_not_break_determinism_within_run() -> None:
    set_all_seeds(42)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10)
    ).eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits_a = model(x)
        logits_b = model(x)
    assert torch.allclose(logits_a.float(), logits_b.float(), rtol=1e-4, atol=1e-4)
