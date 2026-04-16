import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Keep IPL artifacts isolated from international artifacts.
os.environ.setdefault("CRICKET_ARTIFACTS_DIR", str((Path("artifacts") / "ipl").resolve()))
os.environ.setdefault("CRICKET_TRAIN_PROFILE", "balanced")

from src.logger import get_logger
from src.pipeline.full_train_pipeline import FullTrainPipeline
from src.utils.common import save_json


logger = get_logger(__name__)


def _season_to_numeric(value):
    text = str(value).strip()
    m = re.search(r"(20\d{2})", text)
    return int(m.group(1)) if m else np.nan


def normalize_ipl_schema(df: pd.DataFrame) -> pd.DataFrame:
    if "match_id" not in df.columns:
        raise ValueError("Required column 'match_id' missing")

    out = pd.DataFrame()

    out["match_id"] = pd.to_numeric(df.get("match_id"), errors="coerce")
    out["season"] = df.get("season", pd.Series([np.nan] * len(df))).apply(_season_to_numeric)

    if "match_no" in df.columns:
        out["match_no"] = pd.to_numeric(df["match_no"], errors="coerce")
    else:
        out["match_no"] = pd.factorize(out["match_id"], sort=True)[0] + 1

    out["innings"] = pd.to_numeric(df.get("innings"), errors="coerce")

    if "ball_no" in df.columns:
        out["over"] = pd.to_numeric(df["ball_no"], errors="coerce")
    else:
        out["over"] = pd.to_numeric(df.get("over"), errors="coerce")

    out["runs_of_bat"] = pd.to_numeric(df.get("runs_batter"), errors="coerce").fillna(0)
    out["extras"] = pd.to_numeric(df.get("runs_extras"), errors="coerce").fillna(0)

    extra_type = df.get("extra_type", pd.Series([""] * len(df))).astype(str).str.lower()
    out["wide"] = np.where(extra_type.str.contains("wide", na=False), out["extras"], 0)
    out["noballs"] = np.where(extra_type.str.contains("noball", na=False), out["extras"], 0)
    out["legbyes"] = np.where(extra_type.str.contains("legbye", na=False), out["extras"], 0)
    out["byes"] = np.where(
        extra_type.str.contains("bye", na=False) & ~extra_type.str.contains("legbye", na=False),
        out["extras"],
        0,
    )

    out["phase"] = ""
    out["date"] = df.get("date", pd.Series([""] * len(df)))
    out["venue"] = df.get("venue", pd.Series([""] * len(df)))
    out["batting_team"] = df.get("batting_team", pd.Series([""] * len(df)))
    out["bowling_team"] = df.get("bowling_team", pd.Series([""] * len(df)))
    out["striker"] = df.get("batter", pd.Series([""] * len(df)))
    out["bowler"] = df.get("bowler", pd.Series([""] * len(df)))
    out["wicket_type"] = df.get("wicket_kind", pd.Series([np.nan] * len(df)))
    out["player_dismissed"] = df.get("player_out", pd.Series([np.nan] * len(df)))
    out["fielder"] = df.get("fielders", pd.Series([np.nan] * len(df)))

    out = out.dropna(subset=["match_id", "innings", "over"]).copy()
    out["match_id"] = out["match_id"].astype(int)
    out["innings"] = out["innings"].astype(int)
    return out


def discover_ipl_csvs(root: Path):
    files = []
    for path in root.glob("*.csv"):
        name = path.name.lower()
        if "ipl" in name and "deliveries" in name:
            files.append(path)
        elif name == "ipl.csv":
            files.append(path)

    if not files:
        default_candidate = root / "IPL.csv"
        if default_candidate.exists():
            files.append(default_candidate)

    return sorted(set(files), key=lambda p: p.name.lower())


def combine_ipl_csvs(csv_paths, output_path: Path):
    if not csv_paths:
        raise ValueError("No IPL CSV files found.")

    frames = []
    global_offset = 0

    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        if "match_id" not in df.columns:
            raise ValueError(f"Required column 'match_id' missing in {path.name}")

        df = normalize_ipl_schema(df)

        year_match = re.search(r"(20\d{2})", path.name)
        season_year = int(year_match.group(1)) if year_match else None

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

        if season_year is not None:
            df["season_year"] = season_year
        df["source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


def export_simple_artifacts(full_summary: dict, source_files):
    artifacts_dir = Path(os.environ["CRICKET_ARTIFACTS_DIR"]).resolve()
    dual_dir = artifacts_dir / "dual"

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
        "competition": "ipl",
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
    csv_files = discover_ipl_csvs(root)
    if not csv_files:
        raise FileNotFoundError("No IPL files found. Expected IPL.csv or *ipl*deliveries*.csv")

    artifacts_root = Path(os.environ["CRICKET_ARTIFACTS_DIR"]).resolve()

    if len(csv_files) == 1:
        source_path_raw = csv_files[0]
        logger.info("Using single IPL dataset: %s", source_path_raw.name)
        single_df = pd.read_csv(source_path_raw, low_memory=False)
        single_df = normalize_ipl_schema(single_df)
        source_path = artifacts_root / "ipl_standardized.csv"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        single_df.to_csv(source_path, index=False)
        logger.info("Standardized IPL schema written to %s", source_path)
    else:
        source_path = combine_ipl_csvs(csv_files, artifacts_root / "combined_ipl_deliveries.csv")
        logger.info("Combined %d IPL datasets into %s", len(csv_files), source_path)

    pipeline = FullTrainPipeline()
    full_result = pipeline.run(source_path)
    simple_result = export_simple_artifacts(full_result, csv_files)
    logger.info("Training summary (IPL full): %s", full_result)
    logger.info("Exported IPL simple artifacts: %s", simple_result)
    print("IPL training completed")
    print(simple_result)
