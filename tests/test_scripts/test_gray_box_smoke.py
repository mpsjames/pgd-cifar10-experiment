"""Smoke test for the gray-box transfer scenario in `scripts/run_transfer.py`."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def test_gray_box_loads_correct_variant_paths(tmp_path: Path, monkeypatch) -> None:
    from src.cli.transfer import pair_spec

    monkeypatch.chdir(tmp_path)
    args = Namespace(
        mode="gray_box",
        seed=42,
        attack="pgd_10",
        batch_size=2,
        smoke=False,
        no_download=True,
        tracking_uri=None,
        json_dir=tmp_path / "logs",
        no_mlflow=True,
    )
    spec = pair_spec(
        {
            "arch": "resnet18",
            "surrogate_seed": 42,
            "victim_seed": 42,
            "surrogate_variant": "clean",
            "victim_variant": "adv",
        },
        args,
    )

    assert spec["surrogate_seed"] == 42
    assert spec["victim_seed"] == 42
    assert spec["surrogate_variant"] == "clean"
    assert spec["victim_variant"] == "adv"


def test_gray_box_pair_file_is_registered() -> None:
    from scripts import run_transfer

    assert run_transfer._PAIR_FILES["gray_box"] == "configs/transfer/gray_box_pairs.yaml"
    assert Path(run_transfer._PAIR_FILES["gray_box"]).exists()


def test_gray_box_pairs_load_and_have_variant_keys() -> None:
    from scripts.run_transfer import _load_pairs

    pairs = _load_pairs("gray_box")
    assert pairs, "gray-box pair registry must not be empty"
    assert all({"arch", "surrogate_seed", "victim_seed"} <= set(p) for p in pairs)
    variants = {(p.get("surrogate_variant"), p.get("victim_variant")) for p in pairs}
    assert ("clean", "adv") in variants, "gray-box must include an adv-victim pair"
