"""Render side-by-side clean/adversarial/perturbation comparison panels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def make_perturbation_panel(x: Tensor, x_adv: Tensor, title: str = "Perturbation"):
    """Render clean, adversarial, and amplified perturbation views.

    Args:
        x: Clean image or batch, shape `(3, H, W)` or `(B, 3, H, W)`.
        x_adv: Adversarial image or batch with the same shape as `x`.
        title: Figure-level title/caption string.

    Returns:
        Matplotlib figure containing three panels for the first sample when a
        batch is provided.
    """
    if x.ndim == 4:
        x = x[0]
    if x_adv.ndim == 4:
        x_adv = x_adv[0]
    delta = (x_adv - x).detach().cpu()
    images = [
        x.detach().cpu().clamp(0, 1),
        x_adv.detach().cpu().clamp(0, 1),
        (delta * 10 + 0.5).clamp(0, 1),
    ]
    labels = ["clean", "adversarial", "10x perturbation"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    fig.suptitle(title)
    for ax, image, label in zip(axes, images, labels, strict=True):
        ax.imshow(torch.permute(image, (1, 2, 0)).numpy())
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    return fig
