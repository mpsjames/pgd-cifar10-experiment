"""Run attacks over loaders and package the resulting evaluation metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.attacks.base import BaseAttack
from src.attacks.verify import verify_perturbation
from src.evaluation.metrics import (
    attack_success_rate,
    l2_norm,
    linf_norm,
    psnr,
    robust_accuracy,
    ssim,
)
from src.models.normalizer import Normalizer


@dataclass(frozen=True)
class EvaluationResult:
    """Store aggregate metrics from one attack-evaluation run.

    Fields:
        asr: Attack success rate in `[0, 1]`.
        robust_acc: Accuracy under attack in `[0, 1]`.
        linf_mean: Mean per-sample Linf perturbation magnitude.
        l2_mean: Mean per-sample L2 perturbation magnitude.
        psnr_mean: Mean peak signal-to-noise ratio in dB.
        ssim_mean: Mean structural similarity index.
        time_per_image_ms: Wall-clock evaluation time per image in
            milliseconds, including attack generation.
        confidence_drop_mean: Mean reduction in true-class confidence.
        n_samples: Number of evaluated samples.
        per_sample_linf: Optional NumPy array of Linf values when
            `keep_per_sample=True`.
        per_sample_l2: Optional NumPy array of L2 values when
            `keep_per_sample=True`.
        per_sample_confidence_drop: Optional NumPy array of confidence-drop
            values when `keep_per_sample=True`.
        per_sample_correct: Optional NumPy array of booleans indicating
            whether each adversarial prediction remained correct.
    """

    asr: float
    robust_acc: float
    linf_mean: float
    l2_mean: float
    psnr_mean: float
    ssim_mean: float
    time_per_image_ms: float
    confidence_drop_mean: float
    n_samples: int
    per_sample_linf: np.ndarray | None = None
    per_sample_l2: np.ndarray | None = None
    per_sample_confidence_drop: np.ndarray | None = None
    per_sample_correct: np.ndarray | None = None


class AttackEvaluator:
    """Run one attack over a loader and aggregate batch-level metrics.

    Attributes:
        model: Victim model. Moved to `device` and switched to eval mode by
            `run`.
        attack: Attack to apply to each batch.
        test_loader: Evaluation loader yielding `(x, y)` batches.
        device: Target device for attack generation and inference.
        keep_per_sample: When True, retain per-sample arrays in the returned
            `EvaluationResult`.
        perturb_model: Optional surrogate model used for attack generation.
            When set, `attack.perturb` receives this model instead of `model`.
            Clean and adversarial predictions always use `model`.
    """

    def __init__(
        self,
        model: Normalizer,
        attack: BaseAttack,
        test_loader: DataLoader,
        device: torch.device,
        keep_per_sample: bool = False,
        perturb_model: "Normalizer | None" = None,
    ) -> None:
        self.model = model
        self.attack = attack
        self.test_loader = test_loader
        self.device = device
        self.keep_per_sample = keep_per_sample
        self.perturb_model = perturb_model

    def run(self) -> EvaluationResult:
        """Evaluate the configured attack over the full loader.

        Returns:
            Aggregate `EvaluationResult` with means over all processed
            samples.

        Notes:
            `verify_perturbation` is invoked after every attacked batch so
            constraint violations fail fast instead of silently polluting
            downstream tables.
        """
        self.model.to(self.device).eval()
        if self.perturb_model is not None:
            self.perturb_model.to(self.device).eval()
        attack_model = self.perturb_model if self.perturb_model is not None else self.model
        predictions: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        linfs: list[torch.Tensor] = []
        l2s: list[torch.Tensor] = []
        psnrs: list[torch.Tensor] = []
        ssims: list[torch.Tensor] = []
        confidence_drops: list[torch.Tensor] = []
        start = time.perf_counter()

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                clean_probs = torch.softmax(self.model(x), dim=1)
                clean_conf = clean_probs.gather(1, y[:, None]).squeeze(1)
            x_adv = self.attack.perturb(attack_model, x, y)
            verify_perturbation(x, x_adv, self.attack.config.epsilon, self.attack.config.norm)
            with torch.no_grad():
                logits = self.model(x_adv)
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
        correct = pred == lab
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
            per_sample_linf=linf_all.numpy() if self.keep_per_sample else None,
            per_sample_l2=l2_all.numpy() if self.keep_per_sample else None,
            per_sample_confidence_drop=(
                confidence_drop_all.numpy() if self.keep_per_sample else None
            ),
            per_sample_correct=correct.numpy() if self.keep_per_sample else None,
        )
