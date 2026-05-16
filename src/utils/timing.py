"""Measure wall-clock durations and log them through a shared logger."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging import Logger


@contextmanager
def timing_context(name: str, logger: Logger) -> Iterator[None]:
    """Log the elapsed time of the wrapped block.

    Args:
        name: Human-readable operation label.
        logger: Logger used to emit the timing line.

    Yields:
        Control back to the wrapped block.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("[TIMING] %s: %.3fs", name, time.perf_counter() - start)
