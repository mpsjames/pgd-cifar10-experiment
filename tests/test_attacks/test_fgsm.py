from __future__ import annotations

import torch

from src.attacks.fgsm import FGSMAttack
from src.attacks.verify import verify_perturbation


def test_fgsm_respects_linf_and_pixel_domain(tiny_classifier, fgsm_config) -> None:
    attack = FGSMAttack(fgsm_config)
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    x_adv = attack.perturb(tiny_classifier, x, y)
    verify_perturbation(x, x_adv, epsilon=8 / 255)
    assert x_adv.dtype == x.dtype
