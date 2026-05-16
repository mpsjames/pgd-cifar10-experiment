from __future__ import annotations

from pathlib import Path

import mlflow

from src.tracking.mlflow_logger import ExperimentTracker


def test_json_sink_failure_tags_mlflow(monkeypatch, tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        "unit",
        "json-fail",
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
        json_dir=tmp_path / "logs",
    )
    tracker.start_run()
    tags: dict[str, str] = {}
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        mlflow, "set_tag", lambda key, value: tags.__setitem__(key, value)
    )
    tracker.end_run()
    assert tags["json_sink_failed"] == "true"
