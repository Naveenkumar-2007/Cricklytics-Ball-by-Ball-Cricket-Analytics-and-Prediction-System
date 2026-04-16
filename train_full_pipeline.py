import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.pipeline.full_train_pipeline import FullTrainPipeline


logger = get_logger(__name__)


def discover_tournament_csvs(root: Path):
    files = list(root.glob("t20_wc_*_deliveries.csv"))

    def _sort_key(path: Path):
        m = re.search(r"t20_wc_(\d{4})_deliveries\.csv", path.name)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_sort_key)


def combine_tournament_csvs(csv_paths, output_path: Path):
    frames = []
    global_offset = 0

    for path in csv_paths:
        df = pd.read_csv(path)
        local_match = pd.to_numeric(df["match_id"], errors="coerce").fillna(-1).astype(int)
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
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    root = Path(".").resolve()
    csv_files = discover_tournament_csvs(root)
    if not csv_files:
        raise FileNotFoundError("No tournament files found matching pattern t20_wc_*_deliveries.csv")

    if len(csv_files) == 1:
        source = csv_files[0]
    else:
        source = combine_tournament_csvs(csv_files, Path("artifacts") / "combined_worldcup_deliveries.csv")

    pipeline = FullTrainPipeline()
    output = pipeline.run(source)
    logger.info("Full pipeline outputs: %s", output)
    print("Full dual-model pipeline completed")
    print(output)
