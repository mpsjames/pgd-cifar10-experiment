"""Notebook helper for clean-model summary reporting (NB02)."""

from __future__ import annotations

from pathlib import Path

from src.reporting.constants import ARCHES, SEED
from src.reporting.evaluators import clean_accuracy
from src.reporting.gates import CLEAN_ACC_GATES
from src.reporting.io import ensure_result_dirs, write_csv
from src.reporting.model_registry import reporting_model_pair


def nb02_clean_models() -> Path:
    """Write the clean-accuracy summary table for all supported architectures."""
    ensure_result_dirs()
    rows: list[dict[str, object]] = []
    for arch in ARCHES:
        models = reporting_model_pair(arch, SEED)
        gate = CLEAN_ACC_GATES[arch]
        if models.has_clean:
            acc = clean_accuracy(models.clean.load())
            gate_status = "pass" if acc >= gate else "fail"
            rows.append(
                {
                    "arch": arch,
                    "seed": SEED,
                    "clean_acc": f"{acc:.4f}",
                    "gate": f">={gate:.2f}",
                    "gate_status": gate_status,
                }
            )
        else:
            rows.append(
                {
                    "arch": arch,
                    "seed": SEED,
                    "clean_acc": "",
                    "gate": f">={gate:.2f}",
                    "gate_status": "full-campaign-pending",
                }
            )
    out = Path("results/tables/clean_acc.csv")
    write_csv(out, rows)
    return out
