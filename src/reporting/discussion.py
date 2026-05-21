"""Notebook helper for discussion and limitations (NB11)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.reporting.io import ensure_result_dirs


def nb11_discussion() -> Path:
    """Write the limitations payload and package report assets."""
    ensure_result_dirs()
    limitations = [
        "single-seed workflow — set seed in config or pass --seed for reproducibility",
        "no AutoAttack baseline (plan section 16; could be added if NB11 compels)",
        "no L2 / L1 / L0 attacks (Linf only, plan section 16)",
        "no targeted attacks (plan section 4.2)",
        "no TRADES, MART, or other defense beyond APGD AT (plan section 16)",
        "no ImageNet - CIFAR-10 only (plan section 16)",
        "epsilon sweep limited to ResNet-18 (plan section 7.3)",
        "no distributed / multi-GPU training (plan section 16)",
        "no AT hyperparameter tuning beyond APGD AT defaults (plan section 16)",
    ]
    assert len(limitations) >= 8
    Path("results/tables/limitations.json").write_text(
        json.dumps(limitations, indent=2), encoding="utf-8"
    )
    out = Path("results/report_assets.zip")
    with zipfile.ZipFile(out, "w") as zf:
        for folder in [Path("results/figures"), Path("results/tables")]:
            for file in folder.glob("*"):
                zf.write(file, file.relative_to("results"))
    return out
