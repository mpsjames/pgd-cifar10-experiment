"""MLflow read helpers for report tables, with JSON mirror fallback."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from src.experiments.config_loader import load_experiment_config

LOGGER = logging.getLogger(__name__)

_JSON_LOG_DIR = Path("results/logs")


def _read_json_runs(phase: str) -> list[dict[str, object]]:
    """Read run records from the local JSON mirror for a given phase tag.

    Reads all *.json files under ``results/logs/`` and returns those whose
    ``tags.phase`` matches *phase*.  Falls back gracefully to an empty list
    when the directory doesn't exist or files are unreadable.
    """
    if not _JSON_LOG_DIR.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in _JSON_LOG_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.debug("Skipping unreadable JSON log %s: %s", path, exc)
            continue
        if payload.get("tags", {}).get("phase") == phase:
            rows.append(payload)
    return rows


def read_square_mlflow_runs() -> list[dict[str, object]]:
    """Pull Square Attack rows from MLflow runs, falling back to JSON mirror.

    Tries MLflow first; if MLflow is disabled, unreachable, or returns no
    results, reads ``results/logs/*.json`` for runs with
    ``tags.phase = 'black_box_query'``.
    """
    mlflow_rows = _read_square_from_mlflow()
    if mlflow_rows:
        return mlflow_rows
    return _read_square_from_json()


def _read_square_from_mlflow() -> list[dict[str, object]]:
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("mlflow not installed; skipping Square Attack MLflow read")
        return []
    uri = resolve_tracking_uri()
    if uri is None:
        return []
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(mlflow_experiment_name())
        if experiment is None:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.phase = 'black_box_query'",
        )
    except mlflow.exceptions.MlflowException as exc:
        LOGGER.warning("MLflow query_black_box read failed: %s", exc)
        return []
    rows: list[dict[str, object]] = []
    for run in runs:
        if str(run_tag(run, "attack")) != "square":
            continue
        rows.append(
            {
                "arch": run_tag(run, "arch"),
                "variant": run_tag(run, "variant"),
                "num_queries": run_tag(run, "num_queries"),
                "asr": run_metric(run, "asr"),
                "robust_acc": run_metric(run, "robust_acc"),
                "time_per_image_ms": run_metric(run, "time_per_image_ms"),
            }
        )
    return rows


def _read_square_from_json() -> list[dict[str, object]]:
    records = _read_json_runs("black_box_query")
    rows: list[dict[str, object]] = []
    for rec in records:
        tags = rec.get("tags", {})
        if str(tags.get("attack", "")) != "square":
            continue
        metrics = rec.get("metrics", {})
        rows.append(
            {
                "arch": tags.get("arch", ""),
                "variant": tags.get("variant", ""),
                "num_queries": tags.get("num_queries", ""),
                "asr": metrics.get("asr", ""),
                "robust_acc": metrics.get("robust_acc", ""),
                "time_per_image_ms": metrics.get("time_per_image_ms", ""),
            }
        )
    if rows:
        LOGGER.info("Square Attack results read from %d JSON log file(s)", len(rows))
    return rows


def read_transfer_mlflow_runs() -> list[dict[str, object]]:
    """Pull transfer-attack ASR rows from MLflow runs, falling back to JSON mirror.

    Tries MLflow first; if MLflow is disabled, unreachable, or returns no
    results, reads ``results/logs/*.json`` for runs with
    ``tags.phase = 'transfer'``.
    """
    mlflow_rows = _read_transfer_from_mlflow()
    if mlflow_rows:
        return mlflow_rows
    return _read_transfer_from_json()


def _read_transfer_from_mlflow() -> list[dict[str, object]]:
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("mlflow not installed; skipping transfer attack MLflow read")
        return []
    uri = resolve_tracking_uri()
    if uri is None:
        return []
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(mlflow_experiment_name())
        if experiment is None:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.phase = 'transfer'",
        )
    except mlflow.exceptions.MlflowException as exc:
        LOGGER.warning("MLflow transfer read failed: %s", exc)
        return []
    rows: list[dict[str, object]] = []
    for run in runs:
        row = _transfer_row_from_mlflow_run(run)
        if row is not None:
            rows.append(row)
    return rows


def _transfer_row_from_mlflow_run(run) -> dict[str, object] | None:
    mode = run_tag(run, "mode")
    if mode == "cross_seed":
        return None  # cross_seed mode removed; skip historical runs
    row: dict[str, object] = {
        "mode": mode,
        "surrogate": "",
        "victim": "",
        "arch": "",
        "surrogate_seed": "",
        "victim_seed": "",
        "surrogate_variant": "",
        "victim_variant": "",
        "asr": run_metric(run, "asr"),
    }
    if mode == "cross_arch":
        row.update({"surrogate": run_tag(run, "surrogate"), "victim": run_tag(run, "victim")})
    elif mode == "gray_box":
        row.update(
            {
                "arch": run_tag(run, "arch"),
                "surrogate_seed": run_tag(run, "surrogate_seed"),
                "victim_seed": run_tag(run, "victim_seed"),
                "surrogate_variant": run_tag(run, "surrogate_variant"),
                "victim_variant": run_tag(run, "victim_variant"),
            }
        )
    return row


def _read_transfer_from_json() -> list[dict[str, object]]:
    records = _read_json_runs("transfer")
    rows: list[dict[str, object]] = []
    for rec in records:
        tags = rec.get("tags", {})
        metrics = rec.get("metrics", {})
        mode = tags.get("mode", "")
        if mode == "cross_seed":
            continue
        row: dict[str, object] = {
            "mode": mode,
            "surrogate": "",
            "victim": "",
            "arch": "",
            "surrogate_seed": "",
            "victim_seed": "",
            "surrogate_variant": "",
            "victim_variant": "",
            "asr": metrics.get("asr", ""),
        }
        if mode == "cross_arch":
            row.update({"surrogate": tags.get("surrogate", ""), "victim": tags.get("victim", "")})
        elif mode == "gray_box":
            row.update(
                {
                    "arch": tags.get("arch", ""),
                    "surrogate_seed": tags.get("surrogate_seed", ""),
                    "victim_seed": tags.get("victim_seed", ""),
                    "surrogate_variant": tags.get("surrogate_variant", ""),
                    "victim_variant": tags.get("victim_variant", ""),
                }
            )
        rows.append(row)
    if rows:
        LOGGER.info("Transfer attack results read from %d JSON log file(s)", len(rows))
    return rows


@lru_cache(maxsize=1)
def tracking_config():
    return load_experiment_config().tracking


def resolve_tracking_uri() -> str | None:
    tracking = tracking_config()
    if not tracking.enable:
        return None
    return tracking.tracking_uri


def mlflow_experiment_name() -> str:
    return tracking_config().experiment_name


def run_tag(run, key: str) -> object:
    return getattr(run.data, "tags", {}).get(key, "")


def run_metric(run, key: str) -> object:
    return getattr(run.data, "metrics", {}).get(key, "")
