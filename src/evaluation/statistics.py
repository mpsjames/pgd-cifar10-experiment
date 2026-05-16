"""Aggregate seed-level experiment results and run lightweight inference tests."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import numpy as np
from scipy import stats


def _metric_dict(result: object) -> dict[str, object]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported result type: {type(result)!r}")


def aggregate_seeds(
    results: list[object], single_seed_ok: bool = False
) -> dict[str, dict[str, float | int | str | None]]:
    """Aggregate numeric metrics across repeated runs.

    Args:
        results: Seed-level results, each either a dataclass instance or a
            mapping of metric names to scalar values.
        single_seed_ok: When True, accept exactly one result and annotate the
            output with `"note": "single-seed"`.

    Returns:
        Mapping keyed by metric name. Each metric contains `mean`, `std`,
        `min`, `max`, and `n`; single-seed output also includes `note`.

    Raises:
        ValueError: When fewer than three results are provided and the
            single-seed exception does not apply.
        TypeError: When a result is neither a dataclass instance nor a dict.
    """
    if len(results) < 3 and not (single_seed_ok and len(results) == 1):
        raise ValueError(
            "Need at least 3 seed results unless single_seed_ok=True with exactly n=1"
        )

    rows = [_metric_dict(result) for result in results]
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not key.startswith("per_sample")
    ]
    output: dict[str, dict[str, float | int | str | None]] = {}
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        if len(values) == 1:
            output[key] = {
                "mean": float(values[0]),
                "std": None,
                "min": float(values[0]),
                "max": float(values[0]),
                "n": 1,
                "note": "single-seed",
            }
        else:
            output[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": int(len(values)),
            }
    return output


def one_sided_t_test(asrs_a: list[float], asrs_b: list[float]) -> dict[str, float]:
    """Run Welch's one-sided t-test with alternative `a > b`.

    Args:
        asrs_a: Sample values for group A. Must contain at least 2 values.
        asrs_b: Sample values for group B. Must contain at least 2 values.

    Returns:
        Dictionary with `t_stat`, `p_value`, and Welch degrees of freedom
        `df`.

    Raises:
        ValueError: When either sample has fewer than 2 observations.
    """
    if len(asrs_a) < 2 or len(asrs_b) < 2:
        raise ValueError("Welch t-test requires n_a >= 2 and n_b >= 2")
    result = stats.ttest_ind(asrs_a, asrs_b, equal_var=False, alternative="greater")
    df = _welch_df(np.array(asrs_a, dtype=float), np.array(asrs_b, dtype=float))
    return {
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": float(df),
    }


def confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute a two-sided Student-t confidence interval for a sample mean.

    Args:
        values: Sample values. Must contain at least 2 observations.
        confidence: Coverage level in `(0, 1)`.

    Returns:
        `(lower, upper)` confidence bounds for the sample mean.

    Raises:
        ValueError: When fewer than 2 values are provided.
    """
    if len(values) < 2:
        raise ValueError("confidence_interval requires at least 2 values")
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    sem = stats.sem(arr)
    margin = float(sem * stats.t.ppf((1.0 + confidence) / 2.0, len(arr) - 1))
    return mean - margin, mean + margin


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    n_a = len(a)
    n_b = len(b)
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (var_a**2 / (n_a**2 * (n_a - 1))) + (var_b**2 / (n_b**2 * (n_b - 1)))
    return float(numerator / denominator)
