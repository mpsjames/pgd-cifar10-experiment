"""Top-level experiment orchestration service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from src.attacks.base import BaseAttack
from src.cli.loader import load_checkpoint_or_smoke
from src.data.cifar10 import get_cifar10_loaders
from src.data.smoke import make_smoke_loader
from src.evaluation.attack_evaluator import AttackEvaluator, EvaluationResult
from src.experiments.config import ExperimentConfig
from src.experiments.config_loader import load_experiment_config
from src.experiments.device import resolve_configured_device
from src.models.builders import build_normalized_model
from src.tracking.protocols import TrackerProtocol
from src.training.adversarial import AdversarialTrainer
from src.training.base import TrainingResult
from src.training.clean import CleanTrainer


class ExperimentRunner:
    """Orchestrate model/data construction, training, evaluation, and logging."""

    def __init__(
        self,
        config: ExperimentConfig,
        tracker: TrackerProtocol,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.tracker = tracker
        self.device = device or resolve_configured_device(config)

    def train_clean(
        self,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        smoke: bool = False,
        no_download: bool = False,
    ) -> TrainingResult:
        config = self._training_config(epochs=epochs, batch_size=batch_size)
        model = build_normalized_model(config.model)
        train_loader, val_loader = self._loaders(
            config.training.batch_size, smoke=smoke, no_download=no_download
        )
        return CleanTrainer(
            config, model, train_loader, val_loader, self.tracker, device=self.device
        ).fit()

    def train_adversarial(
        self,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        smoke: bool = False,
        no_download: bool = False,
    ) -> TrainingResult:
        config = self._training_config(epochs=epochs, batch_size=batch_size)
        model = build_normalized_model(config.model)
        train_loader, val_loader = self._loaders(
            config.training.batch_size, smoke=smoke, no_download=no_download
        )
        return AdversarialTrainer(
            config, model, train_loader, val_loader, self.tracker, device=self.device
        ).fit()

    def evaluate_attack(
        self,
        attack: BaseAttack,
        *,
        checkpoint: Path | None = None,
        variant: str = "clean",
        batch_size: int = 128,
        smoke: bool = False,
        no_download: bool = False,
    ) -> EvaluationResult:
        model = load_checkpoint_or_smoke(
            arch=self.config.model.arch,
            seed=self.config.seed,
            variant=variant,
            model_config=self.config.model,
            smoke=smoke,
            checkpoint=checkpoint,
        ).to(self.device)
        _, loader = self._loaders(batch_size, smoke=smoke, no_download=no_download)
        result = AttackEvaluator(model, attack, loader, self.device).run()
        self._log_evaluation_result(result)
        return result

    def evaluate_transfer(
        self,
        attack: BaseAttack,
        *,
        surrogate_arch: str,
        victim_arch: str,
        surrogate_seed: int,
        victim_seed: int,
        surrogate_variant: str = "clean",
        victim_variant: str = "clean",
        batch_size: int = 64,
        smoke: bool = False,
        no_download: bool = False,
    ) -> EvaluationResult:
        surrogate_config = load_experiment_config(arch=surrogate_arch, attack=None).model
        victim_config = load_experiment_config(arch=victim_arch, attack=None).model
        surrogate = (
            load_checkpoint_or_smoke(
                arch=surrogate_arch,
                seed=surrogate_seed,
                variant=surrogate_variant,
                model_config=surrogate_config,
                smoke=smoke,
            )
            .to(self.device)
            .eval()
        )
        victim = (
            load_checkpoint_or_smoke(
                arch=victim_arch,
                seed=victim_seed,
                variant=victim_variant,
                model_config=victim_config,
                smoke=smoke,
            )
            .to(self.device)
            .eval()
        )
        _, loader = self._loaders(batch_size, smoke=smoke, no_download=no_download)
        result = AttackEvaluator(victim, attack, loader, self.device, perturb_model=surrogate).run()
        self._log_evaluation_result(result)
        return result

    def _training_config(
        self,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> ExperimentConfig:
        if self.config.training is None:
            raise ValueError("training config is required")
        training = self.config.training
        updates: dict[str, object] = {}
        if epochs is not None:
            updates["epochs"] = epochs
        if batch_size is not None:
            updates["batch_size"] = batch_size
        if self.config.hardware.use_amp_override is not None:
            updates["use_amp"] = self.config.hardware.use_amp_override
        if updates:
            training = replace(training, **updates)
        return cast(ExperimentConfig, replace(self.config, training=training))

    def _loaders(
        self, batch_size: int, *, smoke: bool, no_download: bool
    ) -> tuple[DataLoader, DataLoader]:
        if smoke:
            loader = make_smoke_loader(batch_size, self.config.model.num_classes)
            return loader, loader
        hw = self.config.hardware
        return get_cifar10_loaders(
            batch_size,
            num_workers=hw.num_workers,
            seed=self.config.seed,
            download=not no_download,
            pin_memory=hw.pin_memory,
            persistent_workers=hw.persistent_workers,
            prefetch_factor=hw.prefetch_factor,
        )

    def _log_evaluation_result(self, result: EvaluationResult) -> None:
        self.tracker.log_metrics(
            {
                "asr": result.asr,
                "conditional_asr": result.conditional_asr,
                "robust_acc": result.robust_acc,
                "linf_mean": result.linf_mean,
                "l2_mean": result.l2_mean,
                "psnr_mean": result.psnr_mean,
                "ssim_mean": result.ssim_mean,
                "confidence_drop_mean": result.confidence_drop_mean,
                "time_per_image_ms": result.time_per_image_ms,
                "n_samples": float(result.n_samples),
            }
        )
