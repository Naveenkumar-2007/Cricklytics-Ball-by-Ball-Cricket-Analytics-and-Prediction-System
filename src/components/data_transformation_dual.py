import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DualTransformationConfig
from src.exception import ProjectException
from src.logger import get_logger
from src.utils.common import save_object
from src.utils.dual_data_utils import FIRST_INNINGS_FEATURES, SECOND_INNINGS_FEATURES


logger = get_logger(__name__)


class DualDataTransformation:
    def __init__(self, config: DualTransformationConfig = DualTransformationConfig()):
        self.config = config

    @staticmethod
    def _build_preprocessor(feature_cols):
        cat_cols = [c for c in feature_cols if c in {"batting_team", "bowling_team", "venue", "phase"}]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        return ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", StandardScaler(), num_cols),
            ]
        )

    def initiate(
        self,
        first_train_path,
        first_test_path,
        second_train_path,
        second_test_path,
        second_score_train_path,
        second_score_test_path,
    ):
        try:
            first_train = pd.read_csv(first_train_path)
            first_test = pd.read_csv(first_test_path)
            second_train = pd.read_csv(second_train_path)
            second_test = pd.read_csv(second_test_path)
            second_score_train = pd.read_csv(second_score_train_path)
            second_score_test = pd.read_csv(second_score_test_path)

            first_preprocessor = self._build_preprocessor(FIRST_INNINGS_FEATURES)
            second_preprocessor = self._build_preprocessor(SECOND_INNINGS_FEATURES)

            # Fit only on train partitions.
            first_preprocessor.fit(first_train[FIRST_INNINGS_FEATURES])
            second_preprocessor.fit(second_train[SECOND_INNINGS_FEATURES])

            save_object(self.config.first_preprocessor_path, first_preprocessor)
            save_object(self.config.second_preprocessor_path, second_preprocessor)

            logger.info("Saved first innings preprocessor: %s", self.config.first_preprocessor_path)
            logger.info("Saved second innings preprocessor: %s", self.config.second_preprocessor_path)

            return {
                "first_train_df": first_train,
                "first_test_df": first_test,
                "second_train_df": second_train,
                "second_test_df": second_test,
                "second_score_train_df": second_score_train,
                "second_score_test_df": second_score_test,
                "first_preprocessor": first_preprocessor,
                "second_preprocessor": second_preprocessor,
            }

        except Exception as exc:
            raise ProjectException(exc, context="DualDataTransformation.initiate") from exc
