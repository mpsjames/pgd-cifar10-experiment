from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def test_nb08_requires_matching_seed42_clean_checkpoint(tmp_path: Path, monkeypatch) -> None:
    from src.reporting import nb08_defense_synthesis as reports

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reports, "ARCHES", ["resnet18"])
    (tmp_path / "checkpoints/clean").mkdir(parents=True)
    (tmp_path / "checkpoints/adv").mkdir(parents=True)
    (tmp_path / "checkpoints/clean/resnet18_seed1024.pt").touch()
    (tmp_path / "checkpoints/adv/resnet18_apgd_at_seed42.pt").touch()

    path = reports.nb08_defense_synthesis()

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "arch": "resnet18",
            "attack": "PGD-10",
            "clean_robust_acc": "",
            "at_robust_acc": "",
            "delta": "",
            "status": "full-campaign-pending",
        }
    ]


def test_transfer_mlflow_rows_carry_gray_box_variants(tmp_path: Path, monkeypatch) -> None:
    from src.reporting import mlflow_queries as reports

    runs = [
        SimpleNamespace(
            data=SimpleNamespace(
                tags={
                    "mode": "gray_box",
                    "arch": "resnet18",
                    "surrogate_seed": "42",
                    "victim_seed": "42",
                    "surrogate_variant": "clean",
                    "victim_variant": "adv",
                },
                metrics={"asr": 0.31},
            )
        )
    ]

    class FakeClient:
        def __init__(self, tracking_uri: str) -> None:
            self.tracking_uri = tracking_uri

        def get_experiment_by_name(self, _name: str):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, **_kwargs):
            return runs

    fake_mlflow = SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=FakeClient),
        exceptions=SimpleNamespace(MlflowException=Exception),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    rows = reports.read_transfer_mlflow_runs()

    assert rows[0]["mode"] == "gray_box"
    assert rows[0]["surrogate_variant"] == "clean"
    assert rows[0]["victim_variant"] == "adv"


def test_nb04_main_results_schema_includes_new_columns(tmp_path: Path, monkeypatch) -> None:
    from pathlib import Path as _Path

    from src.evaluation.attack_evaluator import EvaluationResult
    from src.reporting import nb04_main_results as reports

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reports, "ARCHES", ["resnet18"])

    eps = 8 / 255
    fake_result = EvaluationResult(
        asr=0.9,
        robust_acc=0.1,
        linf_mean=eps,
        l2_mean=eps,
        psnr_mean=20.0,
        ssim_mean=0.9,
        time_per_image_ms=1.5,
        confidence_drop_mean=0.5,
        n_samples=10,
    )
    monkeypatch.setattr(
        reports,
        "reporting_model_pair",
        lambda *_args: SimpleNamespace(clean=SimpleNamespace(load_or_none=lambda: object())),
    )
    monkeypatch.setattr(reports, "evaluate_attack", lambda *_args, **_kw: fake_result)
    monkeypatch.setattr(
        reports, "_render_main_figure", lambda _rows: _Path("results/figures/04.png")
    )
    monkeypatch.setattr(reports, "_render_time_vs_asr", lambda *_args: None)

    csv_path, _ = reports.nb04_main_results()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_columns = {
        "arch",
        "attack",
        "seed",
        "num_steps",
        "asr",
        "robust_acc",
        "linf_actual",
        "epsilon_actual_ratio",
        "time_per_image_ms",
    }
    assert expected_columns <= set(rows[0].keys())
    # principles §4.8: Linf actual should track ε for iterative full-budget attacks.
    for row in rows:
        if row["epsilon_actual_ratio"]:
            ratio = float(row["epsilon_actual_ratio"])
            assert 0.95 <= ratio <= 1.05, row


def test_nb03_invariants_are_computed_from_attack_outputs(tmp_path: Path, monkeypatch) -> None:
    from src.reporting import nb03_attack_validation as reports

    class SafeAttack:
        def __init__(self, _config=None) -> None:
            self.config = SimpleNamespace(epsilon=0.25, norm="Linf")

        def perturb(self, _model, x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
            return (x + 0.01).clamp(0.0, 1.0)

    class UnsafeAttack:
        def __init__(self, _config=None) -> None:
            self.config = SimpleNamespace(epsilon=0.25, norm="Linf")

        def perturb(self, _model, x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, 1.5)

    attacks = [SafeAttack(), UnsafeAttack(), SafeAttack()]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        reports,
        "reporting_model_pair",
        lambda *_args: SimpleNamespace(clean=SimpleNamespace(load_or_none=lambda: None)),
    )
    monkeypatch.setattr(reports, "build_fresh_model", lambda _arch: object())
    monkeypatch.setattr(
        reports,
        "evaluation_inputs",
        lambda _n: (torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)),
    )
    monkeypatch.setattr(reports, "FGSMAttack", lambda _config: attacks[0])
    monkeypatch.setattr(
        reports,
        "PGDAttack",
        lambda config: (
            SafeAttack(config)
            if getattr(config, "epsilon", 0.25) == 0.0
            else attacks.pop(1 if len(attacks) > 1 else 0)
        ),
    )
    monkeypatch.setattr(reports, "verify_perturbation", lambda *_args: None)
    monkeypatch.setattr(
        reports,
        "make_perturbation_panel",
        lambda *_args, **_kwargs: SimpleNamespace(
            savefig=lambda _path: None,
        ),
    )
    monkeypatch.setattr(reports.plt, "close", lambda _fig: None)

    reports.nb03_attack_validation()

    data = (tmp_path / "results/tables/03_invariants.json").read_text(encoding="utf-8")
    assert '"linf_holds": false' in data
    assert '"pixel_domain_holds": false' in data
