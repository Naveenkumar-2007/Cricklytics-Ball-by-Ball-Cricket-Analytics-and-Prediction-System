from dataclasses import dataclass
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("CRICKET_ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))).resolve()


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path = ARTIFACTS_DIR / "raw_dataset.csv"
    processed_data_path: Path = ARTIFACTS_DIR / "processed_dataset.csv"
    train_data_path: Path = ARTIFACTS_DIR / "train.csv"
    test_data_path: Path = ARTIFACTS_DIR / "test.csv"


@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_path: Path = ARTIFACTS_DIR / "preprocessor.pkl"


@dataclass(frozen=True)
class ModelTrainerConfig:
    model_path: Path = ARTIFACTS_DIR / "model.pkl"
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json"
    feature_columns_path: Path = ARTIFACTS_DIR / "feature_columns.json"


@dataclass(frozen=True)
class DualDataIngestionConfig:
    full_context_path: Path = ARTIFACTS_DIR / "dual" / "full_context.csv"
    first_innings_train_path: Path = ARTIFACTS_DIR / "dual" / "first_innings_train.csv"
    first_innings_test_path: Path = ARTIFACTS_DIR / "dual" / "first_innings_test.csv"
    second_innings_train_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_train.csv"
    second_innings_test_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_test.csv"
    second_innings_score_train_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_score_train.csv"
    second_innings_score_test_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_score_test.csv"


@dataclass(frozen=True)
class DualTransformationConfig:
    first_preprocessor_path: Path = ARTIFACTS_DIR / "dual" / "first_preprocessor.pkl"
    second_preprocessor_path: Path = ARTIFACTS_DIR / "dual" / "second_preprocessor.pkl"


@dataclass(frozen=True)
class DualModelTrainerConfig:
    first_best_model_path: Path = ARTIFACTS_DIR / "dual" / "first_innings_model.pkl"
    second_best_model_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_model.pkl"
    second_score_best_model_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_score_model.pkl"
    first_model_leaderboard_path: Path = ARTIFACTS_DIR / "dual" / "first_innings_model_leaderboard.csv"
    second_model_leaderboard_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_model_leaderboard.csv"
    second_score_model_leaderboard_path: Path = ARTIFACTS_DIR / "dual" / "second_innings_score_model_leaderboard.csv"
    summary_metrics_path: Path = ARTIFACTS_DIR / "dual" / "dual_model_summary.json"
    simulation_path: Path = ARTIFACTS_DIR / "dual" / "ball_by_ball_simulation.csv"
