from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path


def test_script_modules_import() -> None:
    for module in [
        "scripts.train_clean",
        "scripts.train_adversarial",
        "scripts.run_white_box",
        "scripts.run_transfer",
        "scripts.run_epsilon_sweep",
        "scripts.run_black_box_square",
    ]:
        importlib.import_module(module)


def test_white_box_missing_checkpoint_smoke_falls_back(tmp_path: Path, monkeypatch, capsys) -> None:
    from src.cli import loader as cli_loader
    from src.experiments.config_loader import load_experiment_config

    monkeypatch.chdir(tmp_path)
    sentinel = object()
    monkeypatch.setattr(cli_loader, "build_normalized_model", lambda *args: sentinel)

    exp_config = load_experiment_config(arch="resnet18", attack="pgd_10", seed=42)
    model = cli_loader.load_checkpoint_or_smoke(
        arch="resnet18",
        seed=42,
        variant="clean",
        checkpoint=None,
        smoke=True,
        model_config=exp_config.model,
    )

    assert model is sentinel
    assert "WARNING: smoke run on random weights" in capsys.readouterr().out


def test_white_box_missing_checkpoint_non_smoke_raises(tmp_path: Path, monkeypatch) -> None:
    from src.cli import loader as cli_loader
    from src.experiments.config_loader import load_experiment_config

    monkeypatch.chdir(tmp_path)
    exp_config = load_experiment_config(arch="resnet18", attack="pgd_10", seed=42)

    try:
        cli_loader.load_checkpoint_or_smoke(
            arch="resnet18",
            seed=42,
            variant="clean",
            checkpoint=None,
            smoke=False,
            model_config=exp_config.model,
        )
    except FileNotFoundError as exc:
        assert "checkpoints/clean/resnet18_seed42.pt" in str(exc)
    else:
        raise AssertionError("missing non-smoke checkpoint should raise")
