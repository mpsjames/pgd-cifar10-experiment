from __future__ import annotations

import pytest

from src.evaluation.runner import EvaluationResult
from src.evaluation.statistics import (
    aggregate_seeds,
    confidence_interval,
    one_sided_t_test,
)


def _result(asr: float) -> EvaluationResult:
    return EvaluationResult(asr, 1.0 - asr, 0.1, 0.2, 30.0, 0.9, 1.0, 0.1, 10)


def test_aggregate_seeds_rejects_n1_by_default() -> None:
    with pytest.raises(ValueError):
        aggregate_seeds([_result(0.5)])


def test_aggregate_seeds_accepts_n1_with_flag() -> None:
    stats = aggregate_seeds([_result(0.5)], single_seed_ok=True)
    assert stats["asr"]["mean"] == 0.5
    assert stats["asr"]["std"] is None
    assert stats["asr"]["note"] == "single-seed"


def test_aggregate_seeds_multiple_results() -> None:
    stats = aggregate_seeds([_result(0.4), _result(0.5), _result(0.6)])
    assert stats["asr"]["n"] == 3
    assert stats["asr"]["std"] is not None


def test_t_test_and_ci_require_enough_values() -> None:
    with pytest.raises(ValueError):
        one_sided_t_test([0.1], [0.2, 0.3])
    with pytest.raises(ValueError):
        confidence_interval([0.1])
