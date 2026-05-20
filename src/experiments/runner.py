"""Top-level experiment orchestration service."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from src.attacks.base import BaseAttack
from src.attacks.verify import verify_perturbation
from src.cli.loader import load_checkpoint_or_smoke
from src.data.cifar10 import get_cifar10_loaders
from src.data.smoke import make_smoke_loader
from src.evaluation.metrics import (
    attack_success_rate,
    l2_norm,
    linf_norm,
    psnr,
    robust_accuracy,
    ssim,
)
from src.evaluation.attack_evaluator import AttackEvaluator, EvaluationResult
from src.experiments.config import ExperimentConfig
from src.experiments.config_loader import load_experiment_config
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        result = self._run_transfer_evaluation(surrogate, victim, attack, loader)
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
        if updates:
            training = replace(training, **updates)
        return cast(ExperimentConfig, replace(self.config, training=training))

    def _loaders(
        self, batch_size: int, *, smoke: bool, no_download: bool
    ) -> tuple[DataLoader, DataLoader]:
        if smoke:
            loader = make_smoke_loader(batch_size, self.config.model.num_classes)
            return loader, loader
        return get_cifar10_loaders(batch_size, seed=self.config.seed, download=not no_download)

    def _run_transfer_evaluation(
        self,
        surrogate,
        victim,
        attack: BaseAttack,
        loader: DataLoader,
    ) -> EvaluationResult:
        predictions: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        linfs: list[torch.Tensor] = []
        l2s: list[torch.Tensor] = []
        psnrs: list[torch.Tensor] = []
        ssims: list[torch.Tensor] = []
        confidence_drops: list[torch.Tensor] = []
        start = time.perf_counter()
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                clean_probs = torch.softmax(victim(x), dim=1)
                clean_conf = clean_probs.gather(1, y[:, None]).squeeze(1)
            x_adv = attack.perturb(surrogate, x, y)
            verify_perturbation(x, x_adv, attack.config.epsilon, attack.config.norm)
            with torch.no_grad():
                logits = victim(x_adv)
                adv_probs = torch.softmax(logits, dim=1)
                adv_conf = adv_probs.gather(1, y[:, None]).squeeze(1)
                predictions.append(logits.argmax(dim=1).detach().cpu())
                labels.append(y.detach().cpu())
                confidence_drops.append((clean_conf - adv_conf).detach().cpu())
            linfs.append(linf_norm(x_adv, x).detach().cpu())
            l2s.append(l2_norm(x_adv, x).detach().cpu())
            psnrs.append(psnr(x_adv, x).detach().cpu())
            ssims.append(ssim(x_adv, x).detach().cpu())

        elapsed = time.perf_counter() - start
        pred = torch.cat(predictions)
        lab = torch.cat(labels)
        linf_all = torch.cat(linfs)
        l2_all = torch.cat(l2s)
        psnr_all = torch.cat(psnrs)
        ssim_all = torch.cat(ssims)
        confidence_drop_all = torch.cat(confidence_drops)
        n_samples = int(lab.numel())
        return EvaluationResult(
            asr=attack_success_rate(pred, lab),
            robust_acc=robust_accuracy(pred, lab),
            linf_mean=float(linf_all.mean().item()),
            l2_mean=float(l2_all.mean().item()),
            psnr_mean=float(psnr_all.mean().item()),
            ssim_mean=float(ssim_all.mean().item()),
            time_per_image_ms=1000.0 * elapsed / max(n_samples, 1),
            confidence_drop_mean=float(confidence_drop_all.mean().item()),
            n_samples=n_samples,
        )

    def _log_evaluation_result(self, result: EvaluationResult) -> None:
        self.tracker.log_metrics(
            {
                "asr": result.asr,
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

