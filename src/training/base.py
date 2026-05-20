"""Shared service-object infrastructure for training runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from src.experiments.config import ExperimentConfig, TrainingConfig
from src.tracking.protocols import TrackerProtocol


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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = config.model.arch
        self.seed = config.seed

    @abstractmethod
    def fit(self) -> TrainingResult:
        """Run training and return a result summary."""

    def _build_optimizer(self) -> torch.optim.Optimizer:
        config = self.training_config
        if config.optimizer != "SGD":
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        return SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        config = self.training_config
        if config.scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "multistep":
            if config.lr_milestones is None:
                raise ValueError("multistep scheduler requires config.lr_milestones to be set")
            return MultiStepLR(
                optimizer, milestones=list(config.lr_milestones), gamma=config.lr_gamma
            )
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    @staticmethod
    def _save_checkpoint(path: Path, model: nn.Module, epochs: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epochs": epochs}, path)
