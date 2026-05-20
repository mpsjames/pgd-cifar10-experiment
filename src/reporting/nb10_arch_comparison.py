"""Notebook helper for architecture robustness comparison (NB10)."""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting.attack import build_attack_for_report
from src.reporting.constants import ARCHES, SEED
from src.reporting.evaluators import evaluate_attack
from src.reporting.io import ensure_result_dirs
from src.reporting.model_registry import reporting_model_pair


def nb10_architecture_comparison() -> Path:
    """Compare robust accuracy of all supported architectures under PGD-10 (single seed)."""
    ensure_result_dirs()
    attack = build_attack_for_report("pgd_10")
    arch_results: dict[str, float] = {}
    for arch in ARCHES:
        model = reporting_model_pair(arch, SEED).clean.load_or_none()
        if model is not None:
            arch_results[arch] = evaluate_attack(
                model, attack, sample_size=None, seed=SEED
            ).robust_acc

    payload: dict[str, object] = {"seed": SEED}
    for arch in ARCHES:
        payload[f"{arch}_robust_acc"] = (
            arch_results[arch] if arch in arch_results else "full-campaign-pending"
        )

    path = Path("results/tables/architecture_comparison.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
