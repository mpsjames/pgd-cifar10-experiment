"""Apply the shared plotting style for report figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_style() -> None:
    """Apply the lightweight matplotlib style used by notebook figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
