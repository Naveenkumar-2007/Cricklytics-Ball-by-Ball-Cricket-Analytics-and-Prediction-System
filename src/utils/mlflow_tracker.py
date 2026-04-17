from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

from src.logger import get_logger


logger = get_logger(__name__)


def _extract_dual_summary_values(summary_path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    if not summary_path.exists():
        return {}, {}

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    params: Dict[str, str] = {}
    metrics: Dict[str, float] = {}

    first = summary.get("first_innings", {})
    second = summary.get("second_innings", {})
    second_score = summary.get("second_innings_score", {})

    if first.get("best_model"):
        params["first_best_model"] = str(first["best_model"])
    if second.get("best_model"):
        params["second_best_model"] = str(second["best_model"])
    if second_score.get("best_model"):
        params["second_score_best_model"] = str(second_score["best_model"])

    if "best_test_rmse" in first:
        metrics["first_best_test_rmse"] = float(first["best_test_rmse"])
    if "best_calibrated_test_roc_auc" in second:
        metrics["second_best_calibrated_test_roc_auc"] = float(second["best_calibrated_test_roc_auc"])
    if "best_calibrated_test_brier" in second:
        metrics["second_best_calibrated_test_brier"] = float(second["best_calibrated_test_brier"])
    if "best_test_rmse" in second_score:
        metrics["second_score_best_test_rmse"] = float(second_score["best_test_rmse"])

    return params, metrics


def _resolve_artifact_paths(full_result: dict, simple_result: dict) -> Iterable[Path]:
    keys_from_full = [
        "summary_metrics",
        "first_model_leaderboard",
        "second_model_leaderboard",
        "second_score_model_leaderboard",
        "simulation_csv",
    ]
    keys_from_simple = ["metrics"]

    paths = []
    for key in keys_from_full:
        val = full_result.get(key)
        if val:
            path = Path(val)
            if path.exists() and path.is_file():
                paths.append(path)

    for key in keys_from_simple:
        val = simple_result.get(key)
        if val:
            path = Path(val)
            if path.exists() and path.is_file():
                paths.append(path)

    seen = set()
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        yield path


def log_training_run(
    *,
    competition: str,
    source_dataset: Path,
    source_files: Iterable[Path],
    full_result: dict,
    simple_result: dict,
) -> bool:
    try:
        import mlflow
    except Exception:
        logger.info("MLflow is not available. Skipping experiment tracking.")
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Cricklytics")
    mlflow.set_experiment(experiment_name)

    run_name = os.getenv(
        "MLFLOW_RUN_NAME",
        f"{competition}-train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
    )

    source_files = list(source_files)
    summary_path = Path(full_result.get("summary_metrics", "")) if full_result.get("summary_metrics") else Path()
    summary_params, summary_metrics = _extract_dual_summary_values(summary_path)

    params = {
        "competition": competition,
        "train_profile": os.getenv("CRICKET_TRAIN_PROFILE", "balanced"),
        "max_tuning_rows": os.getenv("CRICKET_MAX_TUNING_ROWS", ""),
        "source_file_count": str(len(source_files)),
        "source_dataset": str(source_dataset),
        **summary_params,
    }

    params = {k: v for k, v in params.items() if str(v).strip() != ""}

    tags = {
        "project": "cricklytics",
        "pipeline": "dual_leakage_safe",
        "competition": competition,
    }

    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            if summary_metrics:
                mlflow.log_metrics(summary_metrics)

            mlflow.log_dict(full_result, "full_train_result.json")
            mlflow.log_dict(simple_result, "simple_artifact_result.json")

            for artifact_path in _resolve_artifact_paths(full_result, simple_result):
                mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")

        logger.info("MLflow logging completed for competition=%s", competition)
        return True
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)
        return False
