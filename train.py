import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.pipeline.full_train_pipeline import FullTrainPipeline
from src.utils.common import save_json
from src.utils.mlflow_tracker import log_training_run


logger = get_logger(__name__)


def discover_tournament_csvs(root: Path):
    files = list(root.glob("t20_wc_*_deliveries.csv"))

    def _sort_key(path: Path):
        m = re.search(r"t20_wc_(\d{4})_deliveries\.csv", path.name)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_sort_key)


def combine_tournament_csvs(csv_paths, output_path: Path):
    if not csv_paths:
        raise ValueError("No tournament CSV files found.")

    frames = []
    global_offset = 0

    for path in csv_paths:
        df = pd.read_csv(path)
        if "match_id" not in df.columns:
            raise ValueError(f"Required column 'match_id' missing in {path.name}")

        year_match = re.search(r"t20_wc_(\d{4})_deliveries\.csv", path.name)
        tournament_year = int(year_match.group(1)) if year_match else None

        local_match = pd.to_numeric(df["match_id"], errors="coerce")
        local_match = local_match.fillna(-1).astype(int)
        local_ids = sorted([x for x in local_match.unique().tolist() if x >= 0])

        local_map = {old_id: idx + 1 for idx, old_id in enumerate(local_ids)}
        remapped = local_match.map(local_map)

        if remapped.isna().any():
            missing_count = int(remapped.isna().sum())
            start_id = len(local_map) + 1
            remapped = remapped.astype("float64")
            remapped[remapped.isna()] = np.arange(start_id, start_id + missing_count)

        df["match_id"] = remapped.astype(int) + int(global_offset)
        global_offset = int(df["match_id"].max())

        if tournament_year is not None:
            df["tournament_year"] = tournament_year
        df["source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


def export_simple_artifacts(full_summary: dict, source_files):
    artifacts_dir = Path("artifacts")
    dual_dir = artifacts_dir / "dual"

    # Export standard single-flow files expected by simple workflows.
    shutil.copy2(Path(full_summary["source_dataset"]), artifacts_dir / "raw_dataset.csv")
    shutil.copy2(dual_dir / "full_context.csv", artifacts_dir / "processed_dataset.csv")
    shutil.copy2(dual_dir / "second_innings_train.csv", artifacts_dir / "train.csv")
    shutil.copy2(dual_dir / "second_innings_test.csv", artifacts_dir / "test.csv")
    shutil.copy2(dual_dir / "second_preprocessor.pkl", artifacts_dir / "preprocessor.pkl")
    shutil.copy2(dual_dir / "second_innings_model.pkl", artifacts_dir / "model.pkl")

    summary_path = dual_dir / "dual_model_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    metrics_payload = {
        "pipeline": "dual_leakage_safe",
        "training_source_files": [str(x) for x in source_files],
        "effective_training_dataset": str(full_summary.get("source_dataset", "")),
        "exported_for_simple_flow": {
            "train": str(artifacts_dir / "train.csv"),
            "test": str(artifacts_dir / "test.csv"),
            "preprocessor": str(artifacts_dir / "preprocessor.pkl"),
            "model": str(artifacts_dir / "model.pkl"),
        },
        "second_innings_best": summary.get("second_innings", {}),
    }
    save_json(artifacts_dir / "metrics.json", metrics_payload)

    return {
        "raw_dataset": str(artifacts_dir / "raw_dataset.csv"),
        "processed_dataset": str(artifacts_dir / "processed_dataset.csv"),
        "train": str(artifacts_dir / "train.csv"),
        "test": str(artifacts_dir / "test.csv"),
        "preprocessor": str(artifacts_dir / "preprocessor.pkl"),
        "model": str(artifacts_dir / "model.pkl"),
        "metrics": str(artifacts_dir / "metrics.json"),
    }


if __name__ == "__main__":
    root = Path(".").resolve()
    os.environ.setdefault("CRICKET_TRAIN_PROFILE", "balanced")
    csv_files = discover_tournament_csvs(root)
    if not csv_files:
        raise FileNotFoundError("No tournament files found matching pattern t20_wc_*_deliveries.csv")

    if len(csv_files) == 1:
        source_path = csv_files[0]
        logger.info("Using single tournament dataset: %s", source_path.name)
    else:
        source_path = combine_tournament_csvs(csv_files, Path("artifacts") / "combined_worldcup_deliveries.csv")
        logger.info("Combined %d tournament datasets into %s", len(csv_files), source_path)

    pipeline = FullTrainPipeline()
    full_result = pipeline.run(source_path)
    simple_result = export_simple_artifacts(full_result, csv_files)
    log_training_run(
        competition="international",
        source_dataset=Path(source_path),
        source_files=csv_files,
        full_result=full_result,
        simple_result=simple_result,
    )
    logger.info("Training summary (full): %s", full_result)
    logger.info("Exported simple artifacts: %s", simple_result)
    print("Training completed")
    print(simple_result)
