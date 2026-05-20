"""Canonical checkpoint path helpers — single source of truth for both schemas."""

from __future__ import annotations

from pathlib import Path


def clean_checkpoint_path(arch: str, seed: int) -> Path:
    """Return the canonical path for a clean-trained checkpoint."""
    return Path("checkpoints/clean") / f"{arch}_seed{seed}.pt"


def adv_checkpoint_path(arch: str, seed: int) -> Path:
    """Return the canonical path for an adversarially-trained checkpoint."""
    return Path("checkpoints/adv") / f"{arch}_apgd_at_seed{seed}.pt"


def variant_checkpoint_path(arch: str, seed: int, variant: str) -> Path:
    """Dispatch to clean or adv checkpoint path based on variant string."""
    if variant == "adv":
        return adv_checkpoint_path(arch, seed)
    if variant == "clean":
        return clean_checkpoint_path(arch, seed)
    raise ValueError(f"Unknown checkpoint variant: {variant!r}")
