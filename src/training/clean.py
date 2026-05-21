"""Clean-training service object."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import Optimizer

from src.experiments.checkpoint_paths import clean_checkpoint_path
from src.training.base import BaseTrainer, TrainingResult
from src.training.early_stopping import EarlyStopping
from src.training.mixup import MixupConfig, apply_mix, mixed_cross_entropy


class CleanTrainer(BaseTrainer):
    """Standard ERM training on clean CIFAR-10 batches."""

    def fit(self) -> TrainingResult:
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
        rng = np.random.default_rng(self.seed)
        for epoch in range(total_epochs):
            metrics = self._train_epoch(optimizer, scaler, rng)
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

        final_path = clean_checkpoint_path(self.arch, self.seed)
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

    def _train_epoch(
        self,
        optimizer: Optimizer,
        scaler: GradScaler,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        cfg = self.training_config
        amp_active = cfg.use_amp and self.device.type == "cuda"
        mix_cfg = MixupConfig(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            switch_prob=cfg.mixup_switch_prob,
            label_smoothing=cfg.label_smoothing,
        )
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            x_in, y_a, y_b, lam = apply_mix(x, y, mix_cfg, rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=amp_active):
                logits = self.model(x_in)
                loss = mixed_cross_entropy(logits, y_a, y_b, lam, cfg.label_smoothing)
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
            # `acc` is reported against the original (un-mixed) labels — the
            # standard Mixup/CutMix convention, kept comparable to non-mixed
            # runs.
            total_loss += float(loss.detach().item()) * y.size(0)
            total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total += y.size(0)
        return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}
