"""Capture and restore resume-checkpoint state for training loops."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import Optimizer


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch, and CUDA RNG state.

    Returns:
        Dictionary suitable for `restore_rng_state`.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore RNG state captured by `capture_rng_state`.

    Args:
        rng_state: Dictionary containing Python, NumPy, torch CPU, and
            optional CUDA RNG state.
    """
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def save_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    epoch: int,
    rng_state: dict[str, Any],
) -> None:
    """Persist a resume checkpoint with optimizer, scaler, scheduler, and RNG state.

    Args:
        path: Output checkpoint path.
        model: Model whose weights should be resumed later.
        optimizer: Optimizer state to persist.
        scaler: AMP gradient scaler state to persist.
        scheduler: Scheduler whose state dict should be saved when present.
        epoch: Last completed epoch index.
        rng_state: RNG-state snapshot from `capture_rng_state`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "rng_state": rng_state,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: Any,
) -> tuple[int, dict[str, Any]]:
    """Load a resume checkpoint and restore model/optimizer/RNG state.

    Args:
        path: Resume-checkpoint path.
        model: Model to receive restored weights.
        optimizer: Optimizer to receive restored state.
        scaler: AMP scaler to receive restored state.
        scheduler: Optional scheduler to receive restored state.

    Returns:
        `(next_epoch, rng_state)` where `next_epoch` is the first epoch still
        to run.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    rng_state = checkpoint["rng_state"]
    restore_rng_state(rng_state)
    return int(checkpoint["epoch"]) + 1, rng_state
