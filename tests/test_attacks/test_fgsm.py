from __future__ import annotations

import torch
from torch import nn

from src.attacks.fgsm import FGSMAttack
from src.attacks.verify import verify_perturbation
from src.experiments.config import AttackConfig


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_fgsm_respects_linf_and_pixel_domain() -> None:
    model = TinyClassifier()
    attack = FGSMAttack(
        AttackConfig(
            "FGSM",
            epsilon=8 / 255,
            alpha=8 / 255,
            num_steps=1,
            random_start=False,
            norm="Linf",
        )
    )
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    x_adv = attack.perturb(model, x, y)
    verify_perturbation(x, x_adv, epsilon=8 / 255)
    assert x_adv.dtype == x.dtype
