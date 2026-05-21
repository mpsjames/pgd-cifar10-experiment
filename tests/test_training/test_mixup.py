from __future__ import annotations

from dataclasses import replace

import torch

from src.experiments.config_loader import load_experiment_config
from src.models.normalizer import Normalizer
from src.training import clean as clean_module
from src.training.clean import CleanTrainer
from src.training.mixup import _rand_bbox


def test_cutmix_bbox_is_non_empty_for_near_identity_lambda() -> None:
    y1, y2, x1, x2 = _rand_bbox(32, 32, 1.0, clean_module.np.random.default_rng(42))
    assert y2 > y1
    assert x2 > x1
    assert (y2 - y1) * (x2 - x1) >= 1


def test_clean_trainer_mix_rng_continues_across_epochs(
    tmp_path, monkeypatch, tiny_classifier, tiny_loader, dummy_tracker
) -> None:
    monkeypatch.chdir(tmp_path)
    draws: list[float] = []

    def record_rng(x, y, _config, rng):
        draws.append(float(rng.random()))
        return x, y, y, 1.0

    monkeypatch.setattr(clean_module, "apply_mix", record_rng)
    config = load_experiment_config(arch="resnet18", training="clean", seed=42)
    config = replace(config, training=replace(config.training, epochs=3, batch_size=2))

    CleanTrainer(
        config,
        Normalizer(tiny_classifier, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        tiny_loader,
        tiny_loader,
        dummy_tracker,
        device=torch.device("cpu"),
    ).fit()

    assert len(draws) == 3
    assert len(set(draws)) == 3
