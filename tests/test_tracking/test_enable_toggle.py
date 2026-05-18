from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tracking import mlflow_logger
from src.tracking.mlflow_logger import ExperimentTracker


def test_tracker_disabled_skips_mlflow_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        mlflow_logger,
        "_mlflow",
        lambda: (_ for _ in ()).throw(AssertionError("mlflow should not import")),
    )
    tracker = ExperimentTracker(
        "unit",
        "disabled-run",
        tracking_uri="http://127.0.0.1:5000",
        json_dir=tmp_path / "logs",
        enable=False,
    )

    assert tracker.start_run() is None
    tracker.log_params({"p": 1})
    tracker.log_metrics({"asr": 0.25})
    tracker.set_tags({"phase": "unit"})
    tracker.end_run()

    assert tracker.run_id is None
    payload = json.loads((tmp_path / "logs" / "disabled-run.json").read_text())
    assert payload["params"]["p"] == 1
    assert payload["metrics"]["asr"] == 0.25
    assert payload["tags"]["phase"] == "unit"


def test_tracker_disabled_still_runs_file_logger(tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        "unit",
        "file-log-run",
        json_dir=tmp_path / "logs",
        enable=False,
    )

    tracker.start_run()
    tracker.log_metrics({"asr": 0.5})
    tracker.end_run()

    per_run = tmp_path / "logs" / "file-log-run.log"
    global_log = tmp_path / "logs" / "experiment.log"
    assert per_run.exists()
    assert global_log.exists()
    assert "mlflow_enabled=False" in per_run.read_text()
    assert "asr=0.5" in global_log.read_text()


def test_tracker_disabled_still_writes_json_sink(tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        "unit",
        "disabled-json",
        json_dir=tmp_path / "logs",
        enable=False,
    )
    with tracker:
        tracker.log_metrics({"robust_acc": 0.42})

    json_path = tmp_path / "logs" / "disabled-json.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["metrics"]["robust_acc"] == pytest.approx(0.42)
    assert payload["status"] == "FINISHED"


def test_no_mlflow_cli_flag_overrides_yaml_default(tmp_path: Path) -> None:
    from src.experiments.config_loader import load_experiment_config

    exp_config = load_experiment_config(arch="resnet18")
    assert exp_config.tracking.enable, "YAML default should have tracking enabled"

    no_mlflow = True  # simulates passing --no-mlflow on the CLI
    tracker = ExperimentTracker(
        exp_config.tracking.experiment_name,
        "no-mlflow-test",
        json_dir=tmp_path / "logs",
        enable=exp_config.tracking.enable and not no_mlflow,
    )
    assert not tracker.enable

    tracker.start_run()
    tracker.end_run()
    assert (tmp_path / "logs" / "no-mlflow-test.json").exists()


def test_reader_returns_empty_when_mlflow_disabled(tmp_path: Path) -> None:
    from src.experiments import notebook_reports

    with patch.object(notebook_reports, "_resolve_tracking_uri", return_value=None):
        result = notebook_reports._read_transfer_mlflow_runs()

    assert result == []


def test_reader_returns_empty_when_server_unreachable(tmp_path: Path, caplog) -> None:
    from src.experiments import notebook_reports
    from src.experiments.config import TrackingConfig

    unreachable_uri = "http://127.0.0.1:1"

    with patch.object(
        notebook_reports,
        "_resolve_tracking_uri",
        return_value=unreachable_uri,
    ), caplog.at_level(logging.WARNING, logger="src.experiments.notebook_reports"):
        result = notebook_reports._read_transfer_mlflow_runs()

    assert result == []
    assert any("transfer" in r.message.lower() for r in caplog.records)
