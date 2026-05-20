from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.attacks.base import BaseAttack
from src.attacks.fgsm import FGSMAttack
from src.experiments.config import AttackConfig


class _ConcreteAttack(BaseAttack):
    def perturb(self, _model: nn.Module, x: Tensor, _y: Tensor) -> Tensor:
        return x


def _config() -> AttackConfig:
    return AttackConfig(
        "FGSM",
        epsilon=8 / 255,
        alpha=8 / 255,
        num_steps=1,
        random_start=False,
        norm="Linf",
    )


def test_base_attack_rejects_unsupported_norm() -> None:
    cfg = _config()
    object.__setattr__(cfg, "norm", "L2")

    with pytest.raises(ValueError, match="Linf"):
        FGSMAttack(cfg)


def test_project_linf_clips_delta_and_pixel_domain() -> None:
    attack = _ConcreteAttack(_config())
    x = torch.full((1, 1, 2, 2), 0.5)
    candidate = torch.tensor([[[[0.0, 0.4], [0.6, 1.0]]]])

    projected = attack._project_linf(x, candidate, epsilon=0.1)

    assert projected.min() >= 0.0
    assert projected.max() <= 1.0
    assert torch.max(torch.abs(projected - x)).item() <= 0.1 + 1e-7


def test_loss_helper_uses_requested_reduction() -> None:
    attack = _ConcreteAttack(_config())
    model = nn.Linear(2, 2)
    x = torch.ones(3, 2)
    y = torch.tensor([0, 1, 0])

    loss = attack._loss(model, x, y, reduction="none")

    assert loss.shape == (3,)
