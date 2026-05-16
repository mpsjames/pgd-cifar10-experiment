from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler

from src.training.checkpoint import (
    capture_rng_state,
    load_resume_checkpoint,
    save_resume_checkpoint,
)
from src.utils.seed import set_all_seeds


def test_save_load_roundtrip_preserves_weights_and_epoch(tmp_path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cuda", enabled=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "resume.pt"
    save_resume_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        scheduler,
        epoch=3,
        rng_state=capture_rng_state(),
    )

    restored = nn.Linear(4, 2)
    restored_optim = torch.optim.SGD(restored.parameters(), lr=0.1)
    restored_scaler = GradScaler("cuda", enabled=False)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optim, step_size=1)
    next_epoch, _ = load_resume_checkpoint(
        path, restored, restored_optim, restored_scaler, restored_scheduler
    )
    assert next_epoch == 4
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)


def test_rng_state_preserved_across_resume(tmp_path) -> None:
    set_all_seeds(42)
    path = tmp_path / "resume.pt"
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cuda", enabled=False)
    state = capture_rng_state()
    save_resume_checkpoint(
        path, model, optimizer, scaler, None, epoch=0, rng_state=state
    )

    expected_random = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    load_resume_checkpoint(path, model, optimizer, scaler, None)
    assert random.random() == expected_random
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_resume_continues_from_epoch_n(tmp_path, monkeypatch) -> None:
    """End-to-end: pre-seed a resume checkpoint and verify adversarial_train picks up after it."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.experiments.config import AttackConfig, TrainingConfig
    from src.models.normalize_wrapper import NormalizedModel
    from src.training import adversarial as module

    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    model = NormalizedModel(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler("cuda", enabled=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1], gamma=0.1
    )
    resume_path = tmp_path / "resume_epoch1.pt"
    save_resume_checkpoint(
        resume_path,
        model,
        optimizer,
        scaler,
        scheduler,
        epoch=1,
        rng_state=capture_rng_state(),
    )

    monkeypatch.chdir(tmp_path)

    class _Tracker:
        def __init__(self) -> None:
            self.tags: dict[str, str] = {}
            self.params: dict[str, object] = {}
            self.metrics_steps: list[int] = []

        def log_metrics(self, metrics, step=None):
            self.metrics_steps.append(step)

        def set_tags(self, tags):
            self.tags.update(tags)

        def log_params(self, params):
            self.params.update(params)

    epochs_run: list[int] = []

    def fake_epoch(*_args, **_kwargs):
        epochs_run.append(len(epochs_run))
        return {"loss": 0.0, "acc_on_adv": 0.0}

    monkeypatch.setattr(module, "adversarial_train_epoch", fake_epoch)

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    config = TrainingConfig(
        mode="adversarial",
        epochs=3,
        batch_size=2,
        lr=0.01,
        weight_decay=0.0,
        optimizer="SGD",
        scheduler="cosine",
        use_amp=False,
        inner_attack=AttackConfig(
            "PGD", epsilon=0.0, alpha=0.0, num_steps=0, random_start=False, norm="Linf"
        ),
        resume_from=resume_path,
        save_every_epochs=1,
    )
    tracker = _Tracker()
    module.adversarial_train(
        model,
        loader,
        loader,
        config,
        tracker,
        torch.device("cpu"),
        arch="resnet18",
        seed=42,
    )

    # Resume snapshot saved epoch=1 ⇒ next_epoch=2 ⇒ only epoch 2 runs (range(2, 3)).
    assert len(epochs_run) == 1
    assert tracker.params["resumed_from_epoch"] == 2
    # Plan §3 resume-path naming: {arch}_seed{S}_epoch{N}.pt
    assert (tmp_path / "checkpoints/adv/_resume/resnet18_seed42_epoch2.pt").exists()
    # Final checkpoint also names arch + seed.
    assert (tmp_path / "checkpoints/adv/resnet18_madry_seed42.pt").exists()
