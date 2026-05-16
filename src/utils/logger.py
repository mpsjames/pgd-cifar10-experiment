"""Provide a small shared logger configuration for scripts and helpers."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the project's default format.

    Args:
        name: Logger name, typically `__name__`.

    Returns:
        Standard-library `logging.Logger` instance.

    Notes:
        This helper calls `logging.basicConfig`, so the first caller controls
        the root logging configuration for the current process.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return logging.getLogger(name)
