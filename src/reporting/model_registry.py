"""Reporting model registry organized around clean/adversarial checkpoint pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.experiments.checkpoint_paths import adv_checkpoint_path, clean_checkpoint_path
from src.experiments.config_loader import load_experiment_config
from src.models.builders import build_model, load_model_from_checkpoint, wrap_with_normalization

CheckpointVariant = Literal["clean", "adv"]


@dataclass(frozen=True)
class ReportingCheckpoint:
    """One checkpoint for a reporting model variant."""

    arch: str
    seed: int
    variant: CheckpointVariant
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def load(self):
        if not self.exists:
            raise FileNotFoundError(f"Missing {self.variant} checkpoint: {self.path}")
        return load_model_from_checkpoint_path(self.arch, self.path)

    def load_or_none(self):
        return self.load() if self.exists else None


@dataclass(frozen=True)
class ReportingModelPair:
    """Clean and adversarial checkpoints for the same architecture and seed."""

    arch: str
    seed: int
    clean: ReportingCheckpoint
    adversarial: ReportingCheckpoint

    @property
    def has_clean(self) -> bool:
        return self.clean.exists

    @property
    def has_adversarial(self) -> bool:
        return self.adversarial.exists

    @property
    def has_complete_pair(self) -> bool:
        return self.has_clean and self.has_adversarial


def reporting_model_pair(arch: str, seed: int) -> ReportingModelPair:
    """Return clean/adversarial checkpoint refs for the same architecture and seed."""
    return ReportingModelPair(
        arch=arch,
        seed=seed,
        clean=ReportingCheckpoint(
            arch=arch,
            seed=seed,
            variant="clean",
            path=clean_checkpoint_path(arch, seed),
        ),
        adversarial=ReportingCheckpoint(
            arch=arch,
            seed=seed,
            variant="adv",
            path=adv_checkpoint_path(arch, seed),
        ),
    )


def load_model_from_checkpoint_path(arch: str, path: Path):
    config = load_experiment_config(arch=arch).model
    return load_model_from_checkpoint(config, path)


def build_fresh_model(arch: str):
    config = load_experiment_config(arch=arch).model
    return wrap_with_normalization(build_model(config), config).eval()
