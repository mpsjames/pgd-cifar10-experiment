from __future__ import annotations

from dataclasses import replace

import torch

from src.experiments.config_loader import load_experiment_config
from src.models.normalizer import Normalizer
from src.training.adversarial import AdversarialTrainer
from src.training.base import TrainingResult
from src.training.clean import CleanTrainer


def _normalized(model) -> Normalizer:
    return Normalizer(model, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_clean_trainer_fit_returns_training_result(
    tmp_path, monkeypatch, tiny_classifier, tiny_loader, dummy_tracker
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_experiment_config(arch="resnet18", training="clean", seed=42)
    config = replace(config, training=replace(config.training, epochs=1, batch_size=2))

    result = CleanTrainer(
        config,
        _normalized(tiny_classifier),
        tiny_loader,
        tiny_loader,
        dummy_tracker,
        device=torch.device("cpu"),
    ).fit()

    assert isinstance(result, TrainingResult)
    assert result.final_checkpoint.exists()
    assert result.epochs_completed == 1
    assert "acc" in result.history[0]


def test_adversarial_trainer_uses_inner_attack(
    tmp_path,
    monkeypatch,
    tiny_classifier,
    tiny_loader,
    dummy_tracker,
    identity_attack_factory,
    pgd_config,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_experiment_config(arch="resnet18", training="apgd_at", seed=42)
    training = replace(
        config.training,
        epochs=1,
        batch_size=2,
        inner_attack=pgd_config,
    )
    attack = identity_attack_factory(pgd_config)

    result = AdversarialTrainer(
        replace(config, training=training),
        _normalized(tiny_classifier),
        tiny_loader,
        tiny_loader,
        dummy_tracker,
        inner_attack=attack,
        device=torch.device("cpu"),
    ).fit()

    assert result.final_checkpoint.exists()
    assert result.epochs_completed == 1
    assert "acc_on_adv" in result.history[0]
