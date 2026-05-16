"""Mirror experiment metadata to MLflow and a local JSON sink."""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow

from src.experiments.config import ExperimentConfig
from src.tracking.env import log_environment


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


class ExperimentTracker:
    """Track one experiment run in MLflow plus a structured JSON mirror.

    Attributes:
        experiment_name: MLflow experiment name.
        run_name: Human-readable run identifier.
        tags: MLflow tags accumulated across the run.
        tracking_uri: MLflow backend URI.
        json_dir: Directory used for JSON mirror artifacts.
        run_id: Active MLflow run id after `start_run`.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tags: dict[str, str] | None = None,
        tracking_uri: str = "file:./mlruns",
        json_dir: Path = Path("results/logs"),
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.tracking_uri = tracking_uri
        self.json_dir = json_dir
        self.run_id: str | None = None
        self._config: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._params: dict[str, Any] = {}

    def start_run(self, config: ExperimentConfig | None = None) -> str:
        """Start the MLflow run and seed it with config/environment metadata.

        Args:
            config: Optional frozen `ExperimentConfig` to flatten into params.

        Returns:
            Active MLflow run id.

        Notes:
            A dirty git tree is surfaced to stderr because reproducibility
            claims in the README assume a clean commit.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The filesystem tracking backend.*",
                category=FutureWarning,
            )
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            active = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self.run_id = active.info.run_id
        if config is not None:
            self._config = asdict(config)
            self.log_params(_flatten_dict(asdict(config)))
        env = log_environment()
        self.log_params(env)
        if env.get("git_dirty"):
            print(
                "WARNING: Running on dirty working tree; results are not exactly reproducible from commit.",
                file=sys.stderr,
            )
        return self.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameter values to both the in-memory mirror and MLflow."""
        self._params.update(params)
        mlflow.log_params(
            {key: _safe_param_value(value) for key, value in params.items()}
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics to both the in-memory mirror and MLflow."""
        self._metrics.update(metrics)
        mlflow.log_metrics(metrics, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Update MLflow tags and the in-memory mirror."""
        self.tags.update(tags)
        mlflow.set_tags(tags)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Upload an artifact file to MLflow."""
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_figure(self, fig, name: str) -> None:
        """Persist a matplotlib figure locally and register it with MLflow."""
        output = self.json_dir.parent / "figures" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        self.log_artifact(output, artifact_path="figures")

    def end_run(self, status: Literal["FINISHED", "FAILED"] = "FINISHED") -> None:
        """Flush the JSON mirror and close the MLflow run.

        Args:
            status: MLflow terminal status to record.

        Notes:
            JSON sink failures do not abort the MLflow run; instead the run is
            tagged with `json_sink_failed=true` so the primary tracking record
            survives.
        """
        try:
            self._write_json(status)
        except Exception:
            print(
                "JSON sink failed; tagging MLflow run with json_sink_failed=true.",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            mlflow.set_tag("json_sink_failed", "true")
        finally:
            mlflow.end_run(status=status)

    def __enter__(self) -> "ExperimentTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type, *_args) -> None:
        self.end_run("FAILED" if exc_type else "FINISHED")

    def _write_json(self, status: str) -> None:
        self.json_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.run_id or "unstarted"
        payload = {
            "run_id": run_id,
            "experiment": self.experiment_name,
            "run_name": self.run_name,
            "status": status,
            "tags": self.tags,
            "config": self._config,
            "params": self._params,
            "metrics": self._metrics,
            "environment": log_environment(),
        }
        (self.json_dir / f"{run_id}.json").write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _safe_param_value(value: Any) -> str | int | float | bool:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
