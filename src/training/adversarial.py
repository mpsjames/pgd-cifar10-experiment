"""Adversarial-training service object."""

from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import Optimizer

from src.attacks.base import BaseAttack
from src.attacks.factory import AttackFactory
from src.attacks.verify import verify_perturbation
from src.experiments.checkpoint_paths import adv_checkpoint_path
from src.training.base import BaseTrainer, TrainingResult
from src.training.early_stopping import EarlyStopping


class AdversarialTrainer(BaseTrainer):
    """Madry-style adversarial training with a configured inner attack."""

    def __init__(
        self,
        *args,
        inner_attack: BaseAttack | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.training_config.inner_attack is None and inner_attack is None:
            raise ValueError("Adversarial training requires config.inner_attack")
        self.inner_attack = inner_attack or AttackFactory.build(self.training_config.inner_attack)

    def fit(self) -> TrainingResult:
        cfg = self.training_config
        if cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0:
            warnings.warn(
                "mixup_alpha/cutmix_alpha in TrainingConfig are only applied by CleanTrainer; "
                "AdversarialTrainer trains on adversarial examples and ignores these fields.",
                stacklevel=2,
            )
        self.model.to(self.device)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        scaler = GradScaler(
            "cuda", enabled=self.training_config.use_amp and self.device.type == "cuda"
        )
        history: list[dict[str, float]] = []
        best_metric = 0.0
        start = time.perf_counter()

        total_epochs = self.training_config.epochs
        val_every = max(1, self.training_config.val_every_n_epochs)
        early_stopper = EarlyStopping(
            patience=self.training_config.early_stopping_patience,
            min_delta=self.training_config.early_stopping_min_delta,
        )
        epochs_completed = 0
        for epoch in range(total_epochs):
            metrics = self._train_epoch(optimizer, scaler)
            scheduler.step()
            epochs_completed = epoch + 1
            do_val = (epoch + 1) % val_every == 0 or epoch == total_epochs - 1
            if do_val:
                val_acc = self._val_epoch()
                metrics["val_acc"] = val_acc
                best_metric = max(best_metric, val_acc)
            history.append(metrics)
            self.tracker.log_metrics(metrics, step=epoch)
            if do_val and early_stopper.update(metrics["val_acc"]):
                break

        final_path = adv_checkpoint_path(self.arch, self.seed)
        self._save_checkpoint(final_path, self.model, epochs_completed)
        return TrainingResult(
            final_checkpoint=final_path,
            best_metric=best_metric,
            history=history,
            elapsed_seconds=time.perf_counter() - start,
            epochs_completed=epochs_completed,
        )

    def _val_epoch(self) -> float:
        self.model.eval()
        total_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total += y.size(0)
        return total_correct / max(total, 1)

    def _train_epoch(self, optimizer: Optimizer, scaler: GradScaler) -> dict[str, float]:
        total_loss = 0.0
        total_correct = 0
        total = 0
        cfg = self.training_config
        amp_active = cfg.use_amp and self.device.type == "cuda"
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.eval()
            x_adv = self.inner_attack.perturb(self.model, x, y)
            verify_perturbation(
                x, x_adv, self.inner_attack.config.epsilon, self.inner_attack.config.norm
            )

            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=amp_active):
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            if amp_active:
                scaler.scale(loss).backward()
                if cfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item()) * y.size(0)
            total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total += y.size(0)
        return {
            "loss": total_loss / max(total, 1),
            "acc_on_adv": total_correct / max(total, 1),
        }
