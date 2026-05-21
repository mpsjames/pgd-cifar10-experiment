"""Notebook helper for defense evaluation synthesis (NB08)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reporting.attack import build_attack_for_report
from src.reporting.constants import ARCHES, SEED
from src.reporting.evaluators import evaluate_attack
from src.reporting.io import ensure_result_dirs, read_csv, write_csv
from src.reporting.queries import read_square_mlflow_runs
from src.reporting.registry import reporting_model_pair


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
    """Aggregate Square Attack runs from MLflow and render time-vs-ASR figure."""
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
    else:
        grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
        for row in rows:
            key = (
                str(row.get("arch", "")),
                str(row.get("variant", "")),
                str(row.get("num_queries", "")),
            )
            grouped.setdefault(key, []).append(row)
        write_csv(
            csv_path, [_square_summary_row(key, runs) for key, runs in sorted(grouped.items())]
        )

    nb04_rows = read_csv(Path("results/tables/main_results.csv"))
    square_for_fig = [
        r for r in read_csv(csv_path) if r.get("asr_mean") and r.get("time_per_image_ms_mean")
    ]
    _render_time_vs_asr(nb04_rows, square_for_fig)
    return csv_path


def _render_time_vs_asr(
    rows: list[dict[str, object]],
    square_rows: list[dict[str, object]] | None = None,
) -> Path:
    """Render attack cost vs effectiveness scatter (white-box + Square)."""
    fig_path = Path("results/figures/04_time_vs_asr.png")
    visible = [
        r for r in rows if str(r.get("asr", "")) != "" and str(r.get("time_per_image_ms", "")) != ""
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    if not visible:
        ax.text(0.5, 0.5, "full-campaign-pending", ha="center", va="center")
        ax.set_title("Time-vs-ASR (awaiting checkpoints)")
        fig.savefig(fig_path)
        plt.close(fig)
        return fig_path

    archs = sorted({str(r["arch"]) for r in visible})
    for arch in archs:
        arch_rows = [r for r in visible if r["arch"] == arch]
        xs = [float(r["time_per_image_ms"]) for r in arch_rows]
        ys = [float(r["asr"]) for r in arch_rows]
        labels = [str(r["attack"]) for r in arch_rows]
        ax.scatter(xs, ys, label=arch)
        for x, y, label in zip(xs, ys, labels, strict=True):
            ax.annotate(label, (x, y), fontsize=7, alpha=0.7)

    for row in square_rows or []:
        ax.scatter(
            float(row["time_per_image_ms_mean"]),
            float(row["asr_mean"]),
            marker="x",
            color="black",
            label=f"square ({row.get('arch', '')}/{row.get('variant', '')})",
        )

    ax.set_xlabel("Mean time per image (ms)")
    ax.set_ylabel("ASR")
    ax.set_title("Attack cost vs effectiveness - CIFAR-10 test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


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
