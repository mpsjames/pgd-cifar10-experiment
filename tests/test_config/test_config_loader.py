from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.config import AttackConfig
from src.experiments.config_loader import (
    load_attack_config,
    load_experiment_config,
    load_hardware_config,
)


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
    assert config.tracking.enable is True
    assert config.tracking.tracking_uri == "http://127.0.0.1:5000"
    assert config.attack.epsilon == 0.03137254901960784


def test_apgd_attack_yaml_loads_optional_fields(repo_root: Path) -> None:
    config = load_attack_config("apgd_ce_10", repo_root / "configs")
    assert config.name == "APGD-CE"
    assert config.rho == 0.75
    assert config.n_restarts == 1


@pytest.mark.parametrize("seed", [0, -1])
def test_seed_must_be_positive_at_config_load(repo_root: Path, seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be a positive non-zero integer"):
        load_experiment_config(repo_root / "configs", arch="resnet18", seed=seed)


def test_hardware_device_literal_is_validated(tmp_path: Path) -> None:
    hardware_dir = tmp_path / "hardware"
    hardware_dir.mkdir()
    (hardware_dir / "bad.yaml").write_text("device: tpu\n", encoding="utf-8")

    with pytest.raises(ValueError, match="device must be 'cuda' or 'cpu'"):
        load_hardware_config("bad", tmp_path)
