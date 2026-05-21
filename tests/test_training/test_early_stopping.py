from __future__ import annotations

from dataclasses import replace

import torch

from src.experiments.config_loader import load_experiment_config
from src.models.normalizer import Normalizer
from src.training.clean import CleanTrainer
from src.training.early_stopping import EarlyStopping


def test_disabled_when_patience_zero() -> None:
    es = EarlyStopping(patience=0)
    assert not es.enabled
    assert es.update(0.1) is False
    assert es.update(0.05) is False
    assert es.update(0.0) is False
    assert es.should_stop is False


def test_triggers_after_patience_runs_out() -> None:
    es = EarlyStopping(patience=2)
    assert es.update(0.5) is False
    assert es.update(0.5) is False
    assert es.update(0.5) is True
    assert es.should_stop is True


def test_improvement_resets_counter() -> None:
    es = EarlyStopping(patience=2)
    es.update(0.5)
    es.update(0.5)
    assert es.update(0.6) is False
    assert es.stale_ticks == 0
    assert es.best == 0.6


def test_min_delta_is_required_improvement() -> None:
    es = EarlyStopping(patience=1, min_delta=0.05)
    es.update(0.5)
    assert es.update(0.52) is True


def test_clean_trainer_stops_early_when_val_acc_flat(
    tmp_path, monkeypatch, tiny_classifier, tiny_loader, dummy_tracker
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_experiment_config(arch="resnet18", training="clean", seed=42)
    training = replace(
        config.training,
        epochs=10,
        batch_size=2,
        val_every_n_epochs=1,
        early_stopping_patience=2,
    )
    config = replace(config, training=training)

    result = CleanTrainer(
        config,
        Normalizer(tiny_classifier, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        tiny_loader,
        tiny_loader,
        dummy_tracker,
        device=torch.device("cpu"),
    ).fit()

    assert result.epochs_completed < 10
    assert result.epochs_completed == len(result.history)
