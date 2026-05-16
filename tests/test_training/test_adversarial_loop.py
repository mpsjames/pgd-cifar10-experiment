from __future__ import annotations

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.base import BaseAttack
from src.experiments.config import AttackConfig
from src.models.normalize_wrapper import NormalizedModel
from src.experiments.config import TrainingConfig
from src.training.adversarial import adversarial_train_epoch


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

    def perturb(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        self.model_training_modes.append(model.training)
        return x.detach().clone()


def test_eval_mode_during_inner_attack_and_train_mode_outer_update() -> None:
    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    model = NormalizedModel(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    attack = ModeRecordingAttack()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler("cuda", enabled=False)
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    metrics = adversarial_train_epoch(
        model, loader, optimizer, scaler, attack, torch.device("cpu"), use_amp=False
    )
    assert attack.model_training_modes == [False, False]
    assert model.training is True
    assert set(metrics) == {"loss", "acc_on_adv"}


def test_no_grad_leak_from_inner_to_outer() -> None:
    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    model = NormalizedModel(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    attack = ModeRecordingAttack()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler("cuda", enabled=False)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    adversarial_train_epoch(
        model,
        DataLoader(TensorDataset(x, y), batch_size=2),
        optimizer,
        scaler,
        attack,
        torch.device("cpu"),
        use_amp=False,
    )
    assert all(
        param.grad is not None for param in model.parameters() if param.requires_grad
    )


def test_adversarial_train_non_oom_runtime_error_propagates_without_checkpoint(
    monkeypatch,
) -> None:
    from src.training import adversarial as module

    class Tracker:
        def log_metrics(self, metrics, step=None):
            raise AssertionError("log_metrics should not be reached")

        def set_tags(self, tags):
            raise AssertionError("set_tags should not be reached")

        def log_params(self, params):
            raise AssertionError("log_params should not be reached")

    def boom(*_args, **_kwargs):
        raise RuntimeError("shape mismatch")

    saved = {"called": False}
    monkeypatch.setattr(module, "adversarial_train_epoch", boom)
    monkeypatch.setattr(
        module,
        "save_resume_checkpoint",
        lambda *_args, **_kwargs: saved.__setitem__("called", True),
    )

    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    model = NormalizedModel(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    loader = DataLoader(
        TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1], dtype=torch.long)),
        batch_size=2,
    )
    config = TrainingConfig(
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
    with torch.random.fork_rng():
        try:
            module.adversarial_train(
                model,
                loader,
                loader,
                config,
                Tracker(),
                torch.device("cpu"),
                arch="resnet18",
                seed=42,
            )
        except RuntimeError as exc:
            assert "shape mismatch" in str(exc)
        else:
            raise AssertionError("RuntimeError was not propagated")
    assert saved["called"] is False
