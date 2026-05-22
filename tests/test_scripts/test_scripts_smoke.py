from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    import pytest

    from src.cli import loader as cli_loader
    from src.experiments.config_loader import load_experiment_config

    monkeypatch.chdir(tmp_path)
    exp_config = load_experiment_config(arch="resnet18", attack="pgd_10", seed=42)

    with pytest.raises(FileNotFoundError, match="checkpoints/clean/resnet18_seed42.pt"):
        cli_loader.load_checkpoint_or_smoke(
            arch="resnet18",
            seed=42,
            variant="clean",
            checkpoint=None,
            smoke=False,
            model_config=exp_config.model,
        )


def test_square_context_tracks_square_overrides(monkeypatch, square_config) -> None:
    from scripts import run_black_box_square as script
    from src.cli.runner import ScriptContext
    from src.experiments.config_loader import load_experiment_config

    args = Namespace(arch="resnet18", seed=42, hardware="cpu")

    def fake_bootstrap(args, *, arch, attack):
        assert attack == "square_5000"
        config = load_experiment_config(arch=arch, attack=attack, seed=args.seed, hardware="cpu")
        return ScriptContext(args=args, config=config)

    monkeypatch.setattr(script, "bootstrap", fake_bootstrap)

    ctx = script.square_context(args, square_config)

    assert ctx.config.attack == square_config
    assert ctx.config.attack.name == "Square"


def _make_sweep_fakes(monkeypatch, seen: dict):
    """Patch ExperimentTracker and ExperimentRunner for sweep tests."""
    from src.cli import sweep

    class FakeTracker:
        def __init__(self, _exp_name, run_name, tags, config, **_kwargs) -> None:
            seen["tracker_epsilon"] = config.attack.epsilon
            seen["run_name"] = run_name
            seen["tags"] = tags

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    class FakeRunner:
        def __init__(self, config, _tracker) -> None:
            seen["runner_epsilon"] = config.attack.epsilon

        def evaluate_attack(self, *_args, **_kwargs):
            return SimpleNamespace(asr=0.0, conditional_asr=0.0, robust_acc=1.0)

    monkeypatch.setattr(sweep, "ExperimentTracker", FakeTracker)
    monkeypatch.setattr(sweep, "ExperimentRunner", FakeRunner)


def test_epsilon_sweep_tracker_config_uses_replaced_epsilon(tmp_path: Path, monkeypatch) -> None:
    from src.cli import sweep
    from src.experiments.config_loader import load_attack_config, load_experiment_config

    seen: dict = {}
    _make_sweep_fakes(monkeypatch, seen)

    args = Namespace(
        tracking_uri=None,
        json_dir=tmp_path,
        no_mlflow=True,
        variant="clean",
        batch_size=2,
        smoke=True,
        no_download=True,
    )
    exp_config = load_experiment_config(arch="resnet18", attack="pgd_10", seed=42, hardware="cpu")
    sweep.run_sweep_point(
        args,
        exp_config,
        "resnet18",
        42,
        "pgd_10",
        load_attack_config("pgd_10"),
        0.0,
    )

    assert seen["tracker_epsilon"] == pytest.approx(0.0)
    assert seen["runner_epsilon"] == pytest.approx(0.0)


def test_epsilon_sweep_run_name_and_tags_include_variant(tmp_path: Path, monkeypatch) -> None:
    """Regression: clean and adv sweeps must produce distinct run identities."""
    from src.cli import sweep
    from src.experiments.config_loader import load_attack_config, load_experiment_config

    for variant in ("clean", "adv"):
        seen: dict = {}
        _make_sweep_fakes(monkeypatch, seen)
        args = Namespace(
            tracking_uri=None,
            json_dir=tmp_path,
            no_mlflow=True,
            variant=variant,
            batch_size=2,
            smoke=True,
            no_download=True,
        )
        exp_config = load_experiment_config(
            arch="resnet18", attack="pgd_10", seed=42, hardware="cpu"
        )
        sweep.run_sweep_point(
            args,
            exp_config,
            "resnet18",
            42,
            "pgd_10",
            load_attack_config("pgd_10"),
            8 / 255,
        )
        assert variant in seen["run_name"], f"variant missing from run_name for {variant!r}"
        assert seen["tags"].get("variant") == variant, f"variant tag missing for {variant!r}"
