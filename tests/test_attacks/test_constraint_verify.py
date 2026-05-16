from __future__ import annotations

import pytest
import torch

from src.attacks.verify import verify_perturbation


def test_verify_accepts_valid_linf() -> None:
    x = torch.zeros(2, 3, 32, 32)
    x_adv = torch.full_like(x, 0.01)
    verify_perturbation(x, x_adv, epsilon=0.02)


def test_verify_rejects_linf_violation() -> None:
    x = torch.zeros(2, 3, 32, 32)
    x_adv = torch.full_like(x, 0.03)
    with pytest.raises(AssertionError, match="L_inf violation"):
        verify_perturbation(x, x_adv, epsilon=0.02)


def test_verify_rejects_pixel_domain_violation() -> None:
    x = torch.zeros(2, 3, 32, 32)
    x_adv = torch.full_like(x, -0.01)
    with pytest.raises(AssertionError, match="valid image domain"):
        verify_perturbation(x, x_adv, epsilon=0.02)
