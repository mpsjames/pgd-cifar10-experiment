"""Compute attack-quality metrics used by the evaluation pipeline."""

from __future__ import annotations

import torch
from skimage.metrics import structural_similarity
from torch import Tensor


def attack_success_rate(predictions: Tensor, labels: Tensor) -> float:
    """Compute attack success rate: fraction of labels changed by the attack.

    Args:
        predictions: Predicted class ids after attack, shape `(B,)`.
        labels: Ground-truth class ids, shape `(B,)`.

    Returns:
        Scalar fraction in `[0, 1]`.
    """
    return float((predictions != labels).float().mean().item())


def robust_accuracy(predictions: Tensor, labels: Tensor) -> float:
    """Compute robust accuracy: fraction of adversarial predictions still correct.

    Args:
        predictions: Predicted class ids after attack, shape `(B,)`.
        labels: Ground-truth class ids, shape `(B,)`.

    Returns:
        Scalar fraction in `[0, 1]`.
    """
    return float((predictions == labels).float().mean().item())


def linf_norm(x_adv: Tensor, x_orig: Tensor) -> Tensor:
    """Compute per-sample L-infinity perturbation magnitude.

    Args:
        x_adv: Adversarial samples, shape `(B, C, H, W)`.
        x_orig: Clean samples, same shape and dtype as `x_adv`.

    Returns:
        Tensor of shape `(B,)` containing the max absolute deviation for each
        sample.
    """
    return (x_adv - x_orig).abs().flatten(1).max(dim=1).values


def l2_norm(x_adv: Tensor, x_orig: Tensor) -> Tensor:
    """Compute per-sample L2 perturbation magnitude.

    Args:
        x_adv: Adversarial samples, shape `(B, C, H, W)`.
        x_orig: Clean samples, same shape and dtype as `x_adv`.

    Returns:
        Tensor of shape `(B,)` containing Euclidean perturbation norms.
    """
    return (x_adv - x_orig).flatten(1).norm(p=2, dim=1)


def psnr(x_adv: Tensor, x_orig: Tensor) -> Tensor:
    """Compute per-sample PSNR between clean and adversarial images.

    Args:
        x_adv: Adversarial samples, shape `(B, C, H, W)`, range `[0, 1]`.
        x_orig: Clean samples, same shape and range as `x_adv`.

    Returns:
        Tensor of shape `(B,)` in decibels. Exact matches yield `inf`.
    """
    mse = (x_adv - x_orig).pow(2).flatten(1).mean(dim=1)
    inf = torch.full_like(mse, float("inf"))
    return torch.where(mse == 0, inf, 10.0 * torch.log10(1.0 / mse))


def ssim(x_adv: Tensor, x_orig: Tensor) -> Tensor:
    """Compute per-sample SSIM between clean and adversarial images.

    Args:
        x_adv: Adversarial samples, shape `(B, 3, H, W)`, range `[0, 1]`.
        x_orig: Clean samples, same shape and range as `x_adv`.

    Returns:
        Tensor of shape `(B,)`, dtype `float32`, on the same device as
        `x_adv`.

    Notes:
        `skimage` runs on NumPy arrays, so each sample is evaluated on CPU and
        then packed back into a torch tensor.
    """
    values: list[float] = []
    for adv, orig in zip(x_adv.detach().cpu(), x_orig.detach().cpu(), strict=True):
        adv_np = adv.permute(1, 2, 0).numpy()
        orig_np = orig.permute(1, 2, 0).numpy()
        values.append(float(structural_similarity(orig_np, adv_np, channel_axis=2, data_range=1.0)))
    return torch.tensor(values, dtype=torch.float32, device=x_adv.device)
