"""Normalize raw CIFAR-10 images inside the model forward pass."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Normalizer(nn.Module):
    """Wrap a classifier so attacks can operate on raw `[0, 1]` inputs.

    Attributes:
        model: Underlying classifier that expects normalized inputs.
        mean: Registered `(1, 3, 1, 1)` buffer containing channel means.
        std: Registered `(1, 3, 1, 1)` buffer containing channel standard
            deviations.
    """

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the input batch and delegate to the wrapped model.

        Args:
            x: Raw CIFAR-10 images, shape `(B, 3, 32, 32)`, dtype float, range
                `[0, 1]`.

        Returns:
            Model logits of shape `(B, num_classes)`.
        """
        return self.model((x - self.mean) / self.std)
