#!/usr/bin/env python
"""Download and persist the RobustBench WRN fallback checkpoint.

This command materializes the documented fallback artifact used when
WRN-34-10 adversarial training does not fit on the target 4 GB GPU.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.robustbench_loader import load_robustbench_wrn


def main() -> None:
    """Export the RobustBench WRN fallback checkpoint to the project layout."""
    model = load_robustbench_wrn()
    out = Path("checkpoints/adv/wrn_34_10_madry_seed42.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "source": "robustbench"}, out)
    print(f"Saved RobustBench fallback artifact to {out}")


if __name__ == "__main__":
    main()
