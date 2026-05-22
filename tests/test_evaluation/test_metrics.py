from __future__ import annotations

import pytest
import torch

from src.evaluation.metrics import (
    attack_success_rate,
    l2_norm,
    linf_norm,
    psnr,
    robust_accuracy,
    ssim,
)


def test_asr_is_robust_error_rate() -> None:
    # asr = (adv_pred != labels).mean() — counts all mismatches including
    # samples already wrong under clean model; equals 1 - robust_acc.
    pred = torch.tensor([1, 2, 3, 4])
    labels = torch.tensor([1, 0, 3, 0])
    assert attack_success_rate(pred, labels) == pytest.approx(0.5)
    assert robust_accuracy(pred, labels) == pytest.approx(0.5)
    assert attack_success_rate(pred, labels) + robust_accuracy(pred, labels) == pytest.approx(1.0)


def test_norms_match_definition() -> None:
    x = torch.zeros(2, 3, 32, 32)
    x_adv = x.clone()
    x_adv[0, 0, 0, 0] = 0.5
    assert torch.equal(linf_norm(x_adv, x), torch.tensor([0.5, 0.0]))
    assert torch.equal(l2_norm(x_adv, x), torch.tensor([0.5, 0.0]))


def test_psnr_ssim_sanity() -> None:
    x = torch.zeros(1, 3, 32, 32)
    x_adv = x.clone()
    assert torch.isinf(psnr(x_adv, x)).all()
    assert torch.allclose(ssim(x_adv, x), torch.ones(1))
