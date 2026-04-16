from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import DualDataIngestionConfig
from src.exception import ProjectException
from src.logger import get_logger
from src.utils.common import save_json
from src.utils.dual_data_utils import build_connected_feature_frames, clean_cricket_dataset


logger = get_logger(__name__)


class DualDataIngestion:
    def __init__(self, config: DualDataIngestionConfig = DualDataIngestionConfig()):
        self.config = config

    def initiate(self, source_csv_path: Path):
        try:
            logger.info("Loading source data for dual pipeline: %s", source_csv_path)
            raw_df = pd.read_csv(source_csv_path)
            clean_df = clean_cricket_dataset(raw_df)
            full_context_df, first_df, second_df, second_score_df = build_connected_feature_frames(clean_df)

            self.config.full_context_path.parent.mkdir(parents=True, exist_ok=True)
            full_context_df.to_csv(self.config.full_context_path, index=False)

            # Group-aware split so deliveries from one match do not leak into train/test.
            first_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            first_train_idx, first_test_idx = next(
                first_splitter.split(first_df, first_df["projected_total_target"], groups=first_df["match_id"])
            )

            second_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            second_train_idx, second_test_idx = next(
                second_splitter.split(second_df, second_df["win"], groups=second_df["match_id"])
            )

            second_score_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            second_score_train_idx, second_score_test_idx = next(
                second_score_splitter.split(
                    second_score_df,
                    second_score_df["second_innings_total_target"],
                    groups=second_score_df["match_id"],
                )
            )

            first_train = first_df.iloc[first_train_idx].reset_index(drop=True)
            first_test = first_df.iloc[first_test_idx].reset_index(drop=True)
            second_train = second_df.iloc[second_train_idx].reset_index(drop=True)
            second_test = second_df.iloc[second_test_idx].reset_index(drop=True)
            second_score_train = second_score_df.iloc[second_score_train_idx].reset_index(drop=True)
            second_score_test = second_score_df.iloc[second_score_test_idx].reset_index(drop=True)

            first_overlap = set(first_train["match_id"]).intersection(set(first_test["match_id"]))
            second_overlap = set(second_train["match_id"]).intersection(set(second_test["match_id"]))
            second_score_overlap = set(second_score_train["match_id"]).intersection(set(second_score_test["match_id"]))

            if first_overlap or second_overlap or second_score_overlap:
                raise ValueError(
                    "Match-level data leakage detected in split: overlapping match_id values found across train/test."
                )

            first_train.to_csv(self.config.first_innings_train_path, index=False)
            first_test.to_csv(self.config.first_innings_test_path, index=False)
            second_train.to_csv(self.config.second_innings_train_path, index=False)
            second_test.to_csv(self.config.second_innings_test_path, index=False)
            second_score_train.to_csv(self.config.second_innings_score_train_path, index=False)
            second_score_test.to_csv(self.config.second_innings_score_test_path, index=False)

            split_audit = {
                "first_innings": {
                    "train_rows": int(len(first_train)),
                    "test_rows": int(len(first_test)),
                    "train_matches": int(first_train["match_id"].nunique()),
                    "test_matches": int(first_test["match_id"].nunique()),
                    "overlap_matches": 0,
                },
                "second_innings_win": {
                    "train_rows": int(len(second_train)),
                    "test_rows": int(len(second_test)),
                    "train_matches": int(second_train["match_id"].nunique()),
                    "test_matches": int(second_test["match_id"].nunique()),
                    "overlap_matches": 0,
                },
                "second_innings_score": {
                    "train_rows": int(len(second_score_train)),
                    "test_rows": int(len(second_score_test)),
                    "train_matches": int(second_score_train["match_id"].nunique()),
                    "test_matches": int(second_score_test["match_id"].nunique()),
                    "overlap_matches": 0,
                },
            }
            save_json(self.config.full_context_path.parent / "split_audit.json", split_audit)

            logger.info(
                "Dual data ingestion complete | first_train=%s first_test=%s second_train=%s second_test=%s second_score_train=%s second_score_test=%s",
                first_train.shape,
                first_test.shape,
                second_train.shape,
                second_test.shape,
                second_score_train.shape,
                second_score_test.shape,
            )

            return {
                "full_context_path": self.config.full_context_path,
                "first_train_path": self.config.first_innings_train_path,
                "first_test_path": self.config.first_innings_test_path,
                "second_train_path": self.config.second_innings_train_path,
                "second_test_path": self.config.second_innings_test_path,
                "second_score_train_path": self.config.second_innings_score_train_path,
                "second_score_test_path": self.config.second_innings_score_test_path,
            }

        except Exception as exc:
            raise ProjectException(exc, context="DualDataIngestion.initiate") from exc
