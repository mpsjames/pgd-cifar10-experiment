"""Validate tensor-level invariants shared by attacks and training loops."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_input_batch(x: Tensor, y: Tensor) -> None:
    """Validate the project's CIFAR-10 batch contract.

    Args:
        x: Input batch expected to have shape `(B, 3, 32, 32)`, dtype
            `float32`, and range `[0, 1]`.
        y: Label batch expected to have shape `(B,)` and dtype `long`.

    Raises:
        AssertionError: When dtype, shape, batch size alignment, or pixel range
            does not match the documented contract.
    """
    if x.dtype != torch.float32:
        raise AssertionError(f"Expected float32, got {x.dtype}")
    if y.dtype != torch.long:
        raise AssertionError(f"Expected long labels, got {y.dtype}")
    if x.ndim != 4 or x.shape[1:] != (3, 32, 32):
        raise AssertionError(f"Expected input shape (B, 3, 32, 32), got {tuple(x.shape)}")
    if y.ndim != 1:
        raise AssertionError(f"Expected labels shape (B,), got {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise AssertionError(f"Batch size mismatch: x={x.shape[0]}, y={y.shape[0]}")
    if x.numel() > 0:
        x_min = float(x.min().item())
        x_max = float(x.max().item())
        if x_min < 0.0 or x_max > 1.0:
            raise AssertionError(f"Input must be in [0, 1], got [{x_min:.4f}, {x_max:.4f}]")
