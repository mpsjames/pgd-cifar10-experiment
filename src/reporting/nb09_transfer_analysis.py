"""Notebook helper for transfer attack analysis (NB09)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reporting.io import ensure_result_dirs, write_csv
from src.reporting.mlflow_queries import read_transfer_mlflow_runs


def nb09_transfer_analysis() -> Path:
    """Read MLflow transfer runs written by scripts/run_transfer.py."""
    ensure_result_dirs()
    rows = read_transfer_mlflow_runs()
    if not rows:
        rows = [
            {"mode": "cross_arch", "status": "full-campaign-pending"},
            {"mode": "gray_box", "status": "full-campaign-pending"},
        ]
    path = Path("results/tables/transfer_analysis.csv")
    write_csv(path, rows)
    return path


def nb09_gray_box_summary() -> tuple[Path, Path]:
    """Summarize gray-box transfer runs with a CSV and grouped bar chart."""
    ensure_result_dirs()
    gray_rows = [
        row for row in read_transfer_mlflow_runs() if str(row.get("mode", "")) == "gray_box"
    ]
    csv_path = Path("results/tables/09_gray_box.csv")
    fig_path = Path("results/figures/09_gray_box.png")
    if not gray_rows:
        write_csv(
            csv_path,
            [{"arch": "", "victim_variant": "", "status": "full-campaign-pending"}],
        )
        _render_pending(fig_path, "Gray-box transfer ASR (awaiting checkpoints)")
        return csv_path, fig_path

    summary_rows = _gray_box_summary_rows(gray_rows)
    write_csv(csv_path, summary_rows)
    _render_gray_box(summary_rows, fig_path)
    return csv_path, fig_path


def _render_gray_box(rows: list[dict[str, object]], fig_path: Path) -> None:
    archs = sorted({str(row["arch"]) for row in rows})
    variants = ["clean", "adv"]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(archs)), 3.5))
    width = 0.35
    indices = np.arange(len(archs))
    for offset, variant in enumerate(variants):
        means = [
            float(np.mean([float(r["asr_mean"]) for r in rows if r["arch"] == arch and r["victim_variant"] == variant]) or 0.0)
            for arch in archs
        ]
        ax.bar(indices + (offset - 0.5) * width, means, width, label=f"victim={variant}")
    ax.set_xticks(indices)
    ax.set_xticklabels(archs)
    ax.set_ylabel("ASR")
    ax.set_title(
        "Gray-box transfer ASR (same arch, different weights)\n"
        "surrogate=clean | attack=PGD-10 | CIFAR-10 test"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _render_pending(fig_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "full-campaign-pending", ha="center", va="center")
    ax.set_title(title)
    fig.savefig(fig_path)
    plt.close(fig)


def _gray_box_summary_rows(gray_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in gray_rows:
        key = (
            str(row.get("arch", "")),
            str(row.get("surrogate_seed", "")),
            str(row.get("victim_seed", "")),
            str(row.get("victim_variant", "clean")),
        )
        asr = row.get("asr")
        if asr in (None, ""):
            continue
        grouped.setdefault(key, []).append(float(asr))

    summary_rows: list[dict[str, object]] = []
    for (arch, s_seed, v_seed, victim_variant), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        summary_rows.append(
            {
                "arch": arch,
                "surrogate_seed": s_seed,
                "victim_seed": v_seed,
                "victim_variant": victim_variant,
                "asr_mean": f"{float(arr.mean()):.4f}",
                "asr_std": f"{float(arr.std(ddof=1)):.4f}" if arr.size >= 2 else "",
                "n_runs": int(arr.size),
            }
        )
    return summary_rows
