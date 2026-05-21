"""Shared service-object infrastructure for training runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from src.experiments.config import ExperimentConfig, TrainingConfig
from src.experiments.device import resolve_configured_device
from src.tracking.protocols import TrackerProtocol

# Parameter names matched as "no weight decay" for AdamW. Biases, all
# normalization layers, and ViT-specific learned tokens follow the standard
# transformer recipe of skipping weight decay.
_NO_DECAY_KEYWORDS: tuple[str, ...] = ("bias", "norm", "pos_embed", "cls_token")


@dataclass
class TrainingResult:
    """Result summary returned by trainer service objects."""

    final_checkpoint: Path
    best_metric: float
    history: list[dict[str, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    epochs_completed: int = 0


class BaseTrainer(ABC):
    """Service object for a complete training run."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tracker: TrackerProtocol,
        *,
        device: torch.device | None = None,
    ) -> None:
        if config.training is None:
            raise ValueError("training config is required")
        self.config = config
        self.training_config: TrainingConfig = config.training
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tracker = tracker
        self.device = device or resolve_configured_device(config)
        self.arch = config.model.arch
        self.seed = config.seed

    @abstractmethod
    def fit(self) -> TrainingResult:
        """Run training and return a result summary."""

    def _build_optimizer(self) -> torch.optim.Optimizer:
        config = self.training_config
        if config.optimizer == "SGD":
            return SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        if config.optimizer == "AdamW":
            decay, no_decay = self._split_decay_params()
            param_groups: list[dict] = [
                {"params": decay, "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            return AdamW(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                weight_decay=config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    def _split_decay_params(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(k in name for k in _NO_DECAY_KEYWORDS):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler:
        config = self.training_config
        main = self._build_main_scheduler(optimizer)
        if config.warmup_epochs <= 0:
            return main
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, main],
            milestones=[config.warmup_epochs],
        )

    def _build_main_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler:
        config = self.training_config
        if config.scheduler == "cosine":
            t_max = max(1, config.epochs - config.warmup_epochs)
            return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.min_lr)
        if config.scheduler == "multistep":
            if config.lr_milestones is None:
                raise ValueError("multistep scheduler requires config.lr_milestones to be set")
            if config.warmup_epochs > 0 and min(config.lr_milestones) <= config.warmup_epochs:
                raise ValueError(
                    f"All lr_milestones {config.lr_milestones} must be greater than "
                    f"warmup_epochs={config.warmup_epochs}"
                )
            milestones = [m - config.warmup_epochs for m in config.lr_milestones]
            return MultiStepLR(optimizer, milestones=milestones, gamma=config.lr_gamma)
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    @staticmethod
    def _save_checkpoint(path: Path, model: nn.Module, epochs: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epochs": epochs}, path)
