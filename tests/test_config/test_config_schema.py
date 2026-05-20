from __future__ import annotations

import pytest
from yaml.parser import ParserError

from src.experiments.config import ExperimentConfig
from src.experiments.config_loader import load_experiment_config, load_training_config


def test_malformed_training_yaml_raises(tmp_path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "bad.yaml").write_text("mode: [", encoding="utf-8")

    with pytest.raises(ParserError):
        load_training_config("bad", tmp_path)


def test_adversarial_training_requires_inner_attack(tmp_path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "bad_adv.yaml").write_text(
        "\n".join(
            [
                "mode: adversarial",
                "epochs: 1",
                "batch_size: 2",
                "lr: 0.1",
                "weight_decay: 0.0",
                "optimizer: SGD",
                "scheduler: cosine",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inner_attack"):
        load_training_config("bad_adv", tmp_path)


def test_cross_arch_config_compositions_load(repo_root) -> None:
    cases = [
        ("resnet18", "pgd_10", "clean"),
        ("vit_tiny", "pgd_10", "apgd_at"),
        ("wrn_34_10", "apgd_ce_10", "apgd_at"),
    ]
    for arch, attack, training in cases:
        config = load_experiment_config(
            repo_root / "configs", arch=arch, attack=attack, training=training
        )
        assert isinstance(config, ExperimentConfig)
        assert config.model.arch == arch
        assert config.attack is not None
        assert config.training is not None
