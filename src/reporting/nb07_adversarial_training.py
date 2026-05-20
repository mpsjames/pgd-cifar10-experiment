"""Notebook helper for adversarial training (NB07)."""

from __future__ import annotations

from pathlib import Path

from src.reporting.attack import build_attack_for_report
from src.reporting.constants import SEED
from src.reporting.evaluators import clean_accuracy, evaluate_attack
from src.reporting.gates import AT_GATES
from src.reporting.io import ensure_result_dirs, write_csv
from src.reporting.model_registry import reporting_model_pair


def nb07_adversarial_training(arch: str) -> Path:
    """Write the single-seed adversarial-training summary for one architecture."""
    ensure_result_dirs()
    models = reporting_model_pair(arch, SEED)
    rows: list[dict[str, object]] = []
    if models.has_adversarial:
        model = models.adversarial.load()
        clean_acc = clean_accuracy(model)
        pgd_result = evaluate_attack(
            model, build_attack_for_report("pgd_10"), sample_size=None, seed=SEED
        )
        clean_gate, robust_gate = AT_GATES[arch]
        disclosure = "single-seed APGD AT (plan section 6)"
        if arch == "wrn_34_10":
            disclosure += "; check MLflow tag wrn_at_source=robustbench for fallback status"
        rows.append(
            {
                "arch": arch,
                "seed": SEED,
                "clean_acc": f"{clean_acc:.4f}",
                "robust_acc": f"{pgd_result.robust_acc:.4f}",
                "gate_clean": f">={clean_gate:.2f}",
                "gate_robust": f">={robust_gate:.2f}",
                "gate_status": _gate_status(clean_acc, pgd_result.robust_acc, arch),
                "disclosure": disclosure,
            }
        )
    else:
        disclosure = "single-seed APGD AT; full training pending"
        if arch == "wrn_34_10":
            disclosure = (
                "single-seed APGD AT; RobustBench fallback documented if fallback_triggered=true"
            )
        rows.append(
            {
                "arch": arch,
                "seed": SEED,
                "clean_acc": "",
                "robust_acc": "",
                "gate_clean": f">={AT_GATES[arch][0]:.2f}",
                "gate_robust": f">={AT_GATES[arch][1]:.2f}",
                "gate_status": "full-campaign-pending",
                "disclosure": disclosure,
            }
        )
    path = Path(f"results/tables/adv_training_{arch}.csv")
    write_csv(path, rows)
    return path


def _gate_status(clean_acc: float, robust_acc: float, arch: str) -> str:
    clean_gate, robust_gate = AT_GATES[arch]
    return "pass" if clean_acc >= clean_gate and robust_acc >= robust_gate else "fail"
