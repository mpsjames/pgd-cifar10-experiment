"""MLflow read helpers for report tables."""

from __future__ import annotations

import logging
from functools import lru_cache

from src.experiments.config_loader import load_experiment_config

LOGGER = logging.getLogger(__name__)


def read_square_mlflow_runs() -> list[dict[str, object]]:
    """Pull Square Attack rows from MLflow runs written by run_black_box_square.py."""
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


def read_transfer_mlflow_runs() -> list[dict[str, object]]:
    """Pull transfer-attack ASR rows from MLflow runs written by run_transfer.py."""
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
        mode = run_tag(run, "mode")
        row = {
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
        if mode == "cross_seed":
            continue  # cross_seed mode removed; skip historical runs
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
        rows.append(row)
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
