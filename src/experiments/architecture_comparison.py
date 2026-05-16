"""Wrap architecture-comparison statistics used by the notebooks."""

from __future__ import annotations

from src.evaluation.statistics import one_sided_t_test


def compare_wrn_vs_resnet18(
    wrn_scores: list[float], resnet18_scores: list[float]
) -> dict[str, float]:
    """Compare WRN-34-10 against ResNet-18 with a one-sided Welch t-test.

    Args:
        wrn_scores: Robust-accuracy measurements for WRN-34-10.
        resnet18_scores: Robust-accuracy measurements for ResNet-18.

    Returns:
        Welch-test payload from `one_sided_t_test`, interpreted as
        `wrn_scores > resnet18_scores`.
    """
    return one_sided_t_test(wrn_scores, resnet18_scores)
