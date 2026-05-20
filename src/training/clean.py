"""Clean-training service object."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import Optimizer

from src.experiments.checkpoint_paths import clean_checkpoint_path
from src.training.base import BaseTrainer, TrainingResult


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

        for epoch in range(self.training_config.epochs):
            metrics = self._train_epoch(optimizer, scaler)
            scheduler.step()
            history.append(metrics)
            best_metric = max(best_metric, metrics["acc"])
            self.tracker.log_metrics(metrics, step=epoch)

        final_path = clean_checkpoint_path(self.arch, self.seed)
        self._save_checkpoint(final_path, self.model, self.training_config.epochs)
        return TrainingResult(
            final_checkpoint=final_path,
            best_metric=best_metric,
            history=history,
            elapsed_seconds=time.perf_counter() - start,
            epochs_completed=self.training_config.epochs,
        )

    def _train_epoch(self, optimizer: Optimizer, scaler: GradScaler) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        amp_active = self.training_config.use_amp and self.device.type == "cuda"
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=amp_active):
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            if amp_active:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * y.size(0)
            total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total += y.size(0)
        return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}
