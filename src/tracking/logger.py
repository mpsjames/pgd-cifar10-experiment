"""Configure experiment logging to a single rotating file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RunNameFilter(logging.Filter):
    def __init__(self, run_name: str) -> None:
        super().__init__()
        self.run_name = run_name

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_name"):
            record.run_name = self.run_name
        return True


def configure_run_logger(
    run_name: str,
    log_dir: Path = Path("results/logs"),
    level: int = logging.INFO,
    global_log_max_bytes: int = 10 * 1024 * 1024,
    global_log_backups: int = 5,
) -> logging.Logger:
    """Configure structured logging for one experiment run.

    Side effects:
        - Attaches a `RotatingFileHandler` to `log_dir/experiment.log`.
        - Returns the project-scoped `pgd_cifar10` logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pgd_cifar10")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):  # list() copy — handler removal during iteration would skip entries
        if getattr(handler, "_pgd_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s [%(run_name)s] %(message)s")
    run_filter = RunNameFilter(run_name)

    global_log = RotatingFileHandler(
        log_dir / "experiment.log",
        maxBytes=global_log_max_bytes,
        backupCount=global_log_backups,
    )
    global_log._pgd_managed = True  # type: ignore[attr-defined]  # sentinel to identify handlers we own
    global_log.setFormatter(formatter)
    global_log.addFilter(run_filter)
    global_log.setLevel(level)

    logger.addHandler(global_log)
    return logger
