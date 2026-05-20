"""Core helpers for the Linf Square Attack."""

from __future__ import annotations

import torch
from torch import Tensor


def margin_loss(logits: Tensor, y: Tensor) -> Tensor:
    correct = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y[:, None], float("-inf"))
    runner_up = masked.max(dim=1).values
    return correct - runner_up


def schedule_p(step: int, budget: int, p_init: float) -> float:
    progress = step / max(budget, 1)
    for threshold, factor in (
        (0.001, 1.0),
        (0.05, 0.5),
        (0.20, 0.25),
        (0.40, 0.125),
        (0.60, 0.0625),
        (0.80, 0.03125),
        (1.00, 0.015625),
    ):
        if progress <= threshold:
            return p_init * factor
    return p_init * 0.015625


def bernoulli_sign(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> Tensor:
    bits = torch.randint(0, 2, shape, generator=generator, device=device)
    return bits.float() * 2.0 - 1.0
