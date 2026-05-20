"""Notebook helper for defense evaluation synthesis (NB08)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.reporting.attack import build_attack_for_report
from src.reporting.constants import ARCHES, SEED
from src.reporting.evaluators import evaluate_attack
from src.reporting.io import ensure_result_dirs, write_csv
from src.reporting.mlflow_queries import read_square_mlflow_runs
from src.reporting.model_registry import reporting_model_pair


def nb08_defense_synthesis() -> Path:
    """Compare clean and adversarially trained robust accuracy under PGD-10."""
    ensure_result_dirs()
    rows: list[dict[str, object]] = []
    for arch in ARCHES:
        models = reporting_model_pair(arch, SEED)
        if not models.has_complete_pair:
            rows.append(
                {
                    "arch": arch,
                    "attack": "PGD-10",
                    "clean_robust_acc": "",
                    "at_robust_acc": "",
                    "delta": "",
                    "status": "full-campaign-pending",
                }
            )
            continue
        pgd = build_attack_for_report("pgd_10")
        clean_eval = evaluate_attack(models.clean.load(), pgd, sample_size=None, seed=SEED)
        at_eval = evaluate_attack(models.adversarial.load(), pgd, sample_size=None, seed=SEED)
        rows.append(
            {
                "arch": arch,
                "attack": "PGD-10",
                "clean_robust_acc": f"{clean_eval.robust_acc:.4f}",
                "at_robust_acc": f"{at_eval.robust_acc:.4f}",
                "delta": f"{at_eval.robust_acc - clean_eval.robust_acc:.4f}",
                "status": "ok",
            }
        )
    path = Path("results/tables/defense_synthesis.csv")
    write_csv(path, rows)
    return path


def nb08_query_black_box_table() -> Path:
    """Aggregate Square Attack runs from MLflow into a query-black-box table."""
    ensure_result_dirs()
    rows = read_square_mlflow_runs()
    csv_path = Path("results/tables/08_query_black_box.csv")
    if not rows:
        write_csv(
            csv_path,
            [
                {
                    "arch": "",
                    "variant": "",
                    "num_queries": "",
                    "asr_mean": "",
                    "asr_std": "",
                    "robust_acc_mean": "",
                    "time_per_image_ms_mean": "",
                    "n_runs": 0,
                    "status": "full-campaign-pending",
                }
            ],
        )
        return csv_path

    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for row in rows:
        key = (
            str(row.get("arch", "")),
            str(row.get("variant", "")),
            str(row.get("num_queries", "")),
        )
        grouped.setdefault(key, []).append(row)
    write_csv(csv_path, [_square_summary_row(key, runs) for key, runs in sorted(grouped.items())])
    return csv_path


def _square_summary_row(
    key: tuple[str, str, str], runs: list[dict[str, float]]
) -> dict[str, object]:
    arch, variant, num_queries = key
    asrs = np.asarray([float(r["asr"]) for r in runs if r.get("asr") not in (None, "")])
    robust = np.asarray(
        [float(r["robust_acc"]) for r in runs if r.get("robust_acc") not in (None, "")]
    )
    times = np.asarray(
        [
            float(r["time_per_image_ms"])
            for r in runs
            if r.get("time_per_image_ms") not in (None, "")
        ]
    )
    return {
        "arch": arch,
        "variant": variant,
        "num_queries": num_queries,
        "asr_mean": f"{float(asrs.mean()):.4f}" if asrs.size else "",
        "asr_std": f"{float(asrs.std(ddof=1)):.4f}" if asrs.size >= 2 else "",
        "robust_acc_mean": f"{float(robust.mean()):.4f}" if robust.size else "",
        "time_per_image_ms_mean": f"{float(times.mean()):.3f}" if times.size else "",
        "n_runs": int(asrs.size),
        "status": "ok",
    }
