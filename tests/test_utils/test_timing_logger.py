from __future__ import annotations

from src.utils.logger import get_logger
from src.utils.timing import timing_context


def test_timing_context_no_crash() -> None:
    logger = get_logger("unit")
    with timing_context("unit", logger):
        pass
