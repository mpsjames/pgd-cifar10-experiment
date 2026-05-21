"""Mixup and CutMix batch augmentations for image classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MixupConfig:
    """Hyperparameters controlling the Mixup/CutMix scheduler."""

    mixup_alpha: float
    cutmix_alpha: float
    switch_prob: float
    label_smoothing: float

    @property
    def enabled(self) -> bool:
        return self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    if h <= 0 or w <= 0:
        return 0, 0, 0, 0
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h = min(h, max(1, int(round(h * cut_ratio))))
    cut_w = min(w, max(1, int(round(w * cut_ratio))))
    cy = int(rng.integers(0, h))
    cx = int(rng.integers(0, w))
    y1 = min(max(0, cy - cut_h // 2), h - cut_h)
    x1 = min(max(0, cx - cut_w // 2), w - cut_w)
    y2 = y1 + cut_h
    x2 = x1 + cut_w
    return y1, y2, x1, x2


def apply_mix(
    x: Tensor,
    y: Tensor,
    config: MixupConfig,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Apply Mixup or CutMix to a `(x, y)` batch.

    Returns `(x_mixed, y_a, y_b, lam)`. The caller computes
    `lam * loss(logits, y_a) + (1 - lam) * loss(logits, y_b)`.
    When mixing is disabled, returns `(x, y, y, 1.0)`.
    """
    if not config.enabled or x.size(0) < 2:
        return x, y, y, 1.0
    use_cutmix = config.cutmix_alpha > 0.0 and (
        config.mixup_alpha <= 0.0 or rng.random() >= config.switch_prob
    )
    alpha = config.cutmix_alpha if use_cutmix else config.mixup_alpha
    lam = float(rng.beta(alpha, alpha)) if alpha > 0.0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    y_b = y[perm]
    if use_cutmix:
        y1, y2, x1, x2 = _rand_bbox(x.size(2), x.size(3), lam, rng)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        area = max((y2 - y1) * (x2 - x1), 1)
        lam = 1.0 - area / float(x.size(2) * x.size(3))
    else:
        x = x.mul(lam).add_(x[perm], alpha=1.0 - lam)
    return x, y, y_b, lam


def mixed_cross_entropy(
    logits: Tensor,
    y_a: Tensor,
    y_b: Tensor,
    lam: float,
    label_smoothing: float,
) -> Tensor:
    """Cross-entropy with optional label smoothing for a mixed `(y_a, y_b)` pair."""
    if lam >= 1.0:
        return F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1.0 - lam) * loss_b
