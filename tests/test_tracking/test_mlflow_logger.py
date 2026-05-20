from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tracking.tracker import ExperimentTracker


def test_json_sink_failure_tags_mlflow(monkeypatch, tmp_path: Path) -> None:
    from src.tracking import tracker

    tags: dict[str, str] = {}

    class _Client:
        def __init__(self, tracking_uri: str) -> None:
            self.tracking_uri = tracking_uri

        def search_experiments(self, max_results: int):
            return []

    fake_mlflow = SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=_Client),
        set_tracking_uri=lambda _uri: None,
        set_experiment=lambda _name: None,
        start_run=lambda run_name, tags: SimpleNamespace(info=SimpleNamespace(run_id="run-1")),
        log_params=lambda _params: None,
        set_tag=lambda key, value: tags.__setitem__(key, value),
        end_run=lambda status: None,
    )
    monkeypatch.setattr(tracker, "_mlflow", lambda: fake_mlflow)

    tracker = ExperimentTracker(
        "unit",
        "json-fail",
        tracking_uri="http://127.0.0.1:5000",
        json_dir=tmp_path / "logs",
    )
    tracker.start_run()
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    tracker.end_run()
    assert tags["json_sink_failed"] == "true"


def test_tracker_rejects_non_http_uri(tmp_path: Path) -> None:
    try:
        ExperimentTracker("unit", "bad-uri", tracking_uri="file:" + "./runs")
    except ValueError as exc:
        assert "http" in str(exc)
    else:
        raise AssertionError("ExperimentTracker must reject non-HTTP MLflow URIs")


def test_tracker_logs_run_through_http_server(mlflow_server: str, tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        "unit",
        "http-run",
        tracking_uri=mlflow_server,
        json_dir=tmp_path / "logs",
        enable=True,
    )
    with tracker:
        tracker.log_metrics({"asr": 0.1})

    assert tracker.run_id is not None
    payload = json.loads((tmp_path / "logs" / tracker.run_id).with_suffix(".json").read_text())
    assert payload["metrics"]["asr"] == pytest.approx(0.1)


def test_logger_writes_experiment_log(tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        "unit",
        "log-test",
        json_dir=tmp_path / "logs",
        enable=False,
    )
    tracker.start_run()
    tracker.log_metrics({"asr": 0.5})
    tracker.end_run()

    assert (tmp_path / "logs" / "experiment.log").exists()


def test_global_log_rotates(tmp_path: Path) -> None:
    # Write two runs so the global log is appended to twice; confirm it stays on disk.
    for run_name in ("rotate-a", "rotate-b"):
        tracker = ExperimentTracker(
            "unit",
            run_name,
            json_dir=tmp_path / "logs",
            enable=False,
        )
        tracker.start_run()
        tracker.log_metrics({"asr": 0.0})
        tracker.end_run()

    global_log = tmp_path / "logs" / "experiment.log"
    assert global_log.exists()
    content = global_log.read_text()
    assert "rotate-a" in content
    assert "rotate-b" in content


def test_tracker_logs_full_config_when_used_as_context_manager(tmp_path: Path) -> None:
    from src.experiments.config_loader import load_experiment_config

    exp_config = load_experiment_config(arch="resnet18")
    with ExperimentTracker(
        exp_config.tracking.experiment_name,
        "config-cm-test",
        json_dir=tmp_path / "logs",
        enable=False,
        config=exp_config,
    ) as tracker:
        assert tracker.run_id is None  # enable=False → no MLflow run

    json_path = tmp_path / "logs" / "config-cm-test.json"
    payload = json.loads(json_path.read_text())
    assert payload["config"], "config block must be non-empty"
    assert payload["config"]["model"]["arch"] == "resnet18"
