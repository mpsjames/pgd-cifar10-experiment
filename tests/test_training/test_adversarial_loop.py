from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.base import BaseAttack
from src.experiments.config import AttackConfig, ExperimentConfig, TrainingConfig
from src.experiments.config_loader import load_experiment_config
from src.models.normalizer import Normalizer
from src.training.adversarial import AdversarialTrainer


class ModeRecordingAttack(BaseAttack):
    def __init__(self) -> None:
        super().__init__(
            AttackConfig(
                "PGD",
                epsilon=0.0,
                alpha=0.0,
                num_steps=0,
                random_start=False,
                norm="Linf",
            )
        )
        self.model_training_modes: list[bool] = []

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model_training_modes.append(model.training)
        return x.detach().clone()


class Tracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.tags: dict[str, str] = {}
        self.params: dict[str, object] = {}

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_artifact(self, *_args, **_kwargs):
        return None


def _model() -> Normalizer:
    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    return Normalizer(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _loader(n: int = 4) -> DataLoader:
    x = torch.rand(n, 3, 32, 32)
    y = torch.arange(n, dtype=torch.long) % 10
    return DataLoader(TensorDataset(x, y), batch_size=2)


def _config(training: TrainingConfig) -> ExperimentConfig:
    return replace(load_experiment_config(arch="resnet18", training="apgd_at"), training=training)


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        mode="adversarial",
        epochs=1,
        batch_size=2,
        lr=0.01,
        weight_decay=0.0,
        optimizer="SGD",
        scheduler="cosine",
        use_amp=False,
        inner_attack=AttackConfig(
            "PGD", epsilon=0.0, alpha=0.0, num_steps=0, random_start=False, norm="Linf"
        ),
    )


def test_inner_attack_runs_in_eval_mode() -> None:
    attack = ModeRecordingAttack()
    trainer = AdversarialTrainer(
        _config(_training_config()),
        _model(),
        _loader(),
        _loader(),
        Tracker(),
        inner_attack=attack,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    scaler = GradScaler("cuda", enabled=False)

    trainer._train_epoch(optimizer, scaler)

    assert attack.model_training_modes == [False, False]


def test_outer_update_runs_in_train_mode_and_records_metrics() -> None:
    attack = ModeRecordingAttack()
    trainer = AdversarialTrainer(
        _config(_training_config()),
        _model(),
        _loader(),
        _loader(),
        Tracker(),
        inner_attack=attack,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    scaler = GradScaler("cuda", enabled=False)

    metrics = trainer._train_epoch(optimizer, scaler)

    assert trainer.model.training is True
    assert set(metrics) == {"loss", "acc_on_adv"}


def test_gradient_flow_in_outer_update() -> None:
    trainer = AdversarialTrainer(
        _config(_training_config()),
        _model(),
        _loader(2),
        _loader(2),
        Tracker(),
        inner_attack=ModeRecordingAttack(),
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    scaler = GradScaler("cuda", enabled=False)

    trainer._train_epoch(optimizer, scaler)

    assert all(
        param.grad is not None for param in trainer.model.parameters() if param.requires_grad
    )
