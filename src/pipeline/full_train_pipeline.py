from pathlib import Path

from src.components.data_ingestion_dual import DualDataIngestion
from src.components.data_transformation_dual import DualDataTransformation
from src.components.model_trainer_dual import DualModelTrainer
from src.config import DualDataIngestionConfig, DualModelTrainerConfig
from src.exception import ProjectException
from src.logger import get_logger
from src.pipeline.realtime_engine import RealtimeEngine
from src.utils.common import save_json
from src.utils.visualization import generate_broadcast_charts


logger = get_logger(__name__)


class FullTrainPipeline:
    def run(self, source_csv_path: Path):
        try:
            logger.info("Starting full dual-model cricket pipeline")

            ingestion = DualDataIngestion()
            ingestion_paths = ingestion.initiate(source_csv_path)

            transformer = DualDataTransformation()
            transformed_bundle = transformer.initiate(
                ingestion_paths["first_train_path"],
                ingestion_paths["first_test_path"],
                ingestion_paths["second_train_path"],
                ingestion_paths["second_test_path"],
                ingestion_paths["second_score_train_path"],
                ingestion_paths["second_score_test_path"],
            )

            trainer = DualModelTrainer()
            training_summary = trainer.initiate(transformed_bundle)

            # Generate one simulation and charts for combined match view artifacts.
            engine = RealtimeEngine()
            full_context = engine.full_context_df
            candidate_matches = (
                full_context[full_context["innings"].isin([1, 2])]
                .groupby("match_id")["innings"]
                .nunique()
            )
            if not candidate_matches.empty:
                match_id = int(candidate_matches[candidate_matches >= 2].index[0])
            else:
                match_id = int(full_context["match_id"].iloc[0])

            sim_df = engine.simulate_match_ball_by_ball(match_id)
            sim_path = DualModelTrainerConfig().simulation_path
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            sim_df.to_csv(sim_path, index=False)

            charts_dir = sim_path.parent / "charts"
            generate_broadcast_charts(sim_df, charts_dir)

            final_summary = {
                "source_dataset": str(source_csv_path),
                "full_context_path": str(DualDataIngestionConfig().full_context_path),
                "first_innings_train": str(ingestion_paths["first_train_path"]),
                "first_innings_test": str(ingestion_paths["first_test_path"]),
                "second_innings_train": str(ingestion_paths["second_train_path"]),
                "second_innings_test": str(ingestion_paths["second_test_path"]),
                "second_innings_score_train": str(ingestion_paths["second_score_train_path"]),
                "second_innings_score_test": str(ingestion_paths["second_score_test_path"]),
                "first_best_model": training_summary["first_best_model"],
                "second_best_model": training_summary["second_best_model"],
                "second_score_best_model": training_summary["second_score_best_model"],
                "first_model_leaderboard": str(training_summary["first_leaderboard"]),
                "second_model_leaderboard": str(training_summary["second_leaderboard"]),
                "second_score_model_leaderboard": str(training_summary["second_score_leaderboard"]),
                "summary_metrics": str(training_summary["summary"]),
                "simulation_csv": str(sim_path),
                "charts_dir": str(charts_dir),
            }
            save_json(sim_path.parent / "pipeline_outputs.json", final_summary)

            logger.info("Full dual-model pipeline completed successfully")
            return final_summary

        except Exception as exc:
            raise ProjectException(exc, context="FullTrainPipeline.run") from exc
