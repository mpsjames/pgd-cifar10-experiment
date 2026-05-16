from __future__ import annotations

from pathlib import Path

from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config, load_experiment_config


def test_every_attack_yaml_loads_as_attack_config(repo_root: Path) -> None:
    attack_dir = repo_root / "configs" / "attack"
    for path in attack_dir.glob("*.yaml"):
        config = load_attack_config(path.stem, repo_root / "configs")
        assert isinstance(config, AttackConfig), path.name


def test_root_config_loads_frozen_dataclasses(repo_root: Path) -> None:
    config = load_experiment_config(
        repo_root / "configs", arch="resnet18", attack="pgd_10", training="clean"
    )
    assert config.model.arch == "resnet18"
    assert config.attack is not None
    assert config.training is not None
    assert config.attack.epsilon == 0.03137254901960784
