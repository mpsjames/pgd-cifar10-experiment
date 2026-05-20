"""Structural tracker protocol used by orchestration and training services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrackerProtocol(Protocol):
    """Minimum tracker surface needed by services."""

    def __enter__(self) -> TrackerProtocol: ...

    def __exit__(self, exc_type, exc, tb) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def set_tags(self, tags: dict[str, str]) -> None: ...

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None: ...
