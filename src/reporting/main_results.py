"""Notebook helper for main white-box quantitative results (NB04)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reporting.attack import build_attack_for_report
from src.reporting.constants import ARCHES, NB04_ATTACK_NAMES, SEED
from src.reporting.evaluators import evaluate_attack
from src.reporting.io import ensure_result_dirs, write_csv
from src.reporting.registry import reporting_model_pair


def nb04_main_results() -> tuple[Path, Path]:
    """Evaluate every architecture × attack combination for the single experiment seed."""
    ensure_result_dirs()
    rows: list[dict[str, object]] = []
    for arch in ARCHES:
        model = reporting_model_pair(arch, SEED).clean.load_or_none()
        for attack_name in NB04_ATTACK_NAMES:
            attack = build_attack_for_report(attack_name)
            if model is not None:
                result = evaluate_attack(model, attack, sample_size=None, seed=SEED)
                rows.append(
                    {
                        "arch": arch,
                        "attack": attack_name,
                        "seed": SEED,
                        "num_steps": attack.config.num_steps,
                        "asr": f"{result.asr:.4f}",
                        "robust_acc": f"{result.robust_acc:.4f}",
                        "linf_actual": f"{result.linf_mean:.6f}",
                        "epsilon_actual_ratio": (
                            f"{result.linf_mean / attack.config.epsilon:.4f}"
                            if attack.config.epsilon > 0
                            else ""
                        ),
                        "time_per_image_ms": f"{result.time_per_image_ms:.3f}",
                    }
                )
            else:
                rows.append(_pending_row(arch, attack_name, attack.config.num_steps))

    table_path = Path("results/tables/main_results.csv")
    write_csv(table_path, rows)
    fig_path = _render_main_figure(rows)
    return table_path, fig_path


def _pending_row(arch: str, attack_name: str, num_steps: int) -> dict[str, object]:
    return {
        "arch": arch,
        "attack": attack_name,
        "seed": SEED,
        "num_steps": num_steps,
        "asr": "",
        "robust_acc": "",
        "linf_actual": "",
        "epsilon_actual_ratio": "",
        "time_per_image_ms": "",
    }


def _render_main_figure(rows: list[dict[str, object]]) -> Path:
    fig_path = Path("results/figures/04_main.png")
    visible = [r for r in rows if str(r.get("asr", "")) != ""]
    fig, ax = plt.subplots(figsize=(7, 4))
    if not visible:
        ax.text(0.5, 0.5, "full-campaign-pending", ha="center", va="center")
        ax.set_title("White-box ASR on CIFAR-10 test (awaiting checkpoints)")
        ax.set_ylabel("ASR")
        fig.savefig(fig_path)
        plt.close(fig)
        return fig_path

    archs = sorted({str(r["arch"]) for r in visible})
    indices = np.arange(len(NB04_ATTACK_NAMES))
    width = 0.8 / max(len(archs), 1)
    for offset, arch in enumerate(archs):
        means = []
        for attack_name in NB04_ATTACK_NAMES:
            match = [r for r in visible if r["arch"] == arch and r["attack"] == attack_name]
            means.append(float(match[0]["asr"]) if match else 0.0)
        ax.bar(
            indices + (offset - (len(archs) - 1) / 2.0) * width,
            means,
            width,
            label=arch,
        )
    ax.set_xticks(indices)
    ax.set_xticklabels(NB04_ATTACK_NAMES)
    ax.set_ylabel("ASR")
    ax.set_title(
        f"White-box ASR on CIFAR-10 test (n=10000)\n"
        f"attacks={{{','.join(NB04_ATTACK_NAMES)}}} | seed=42"
    )
    ax.legend(title="architecture", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path
