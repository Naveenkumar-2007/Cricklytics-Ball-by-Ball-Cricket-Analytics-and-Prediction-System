import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from src.config import ModelTrainerConfig
from src.exception import ProjectException
from src.logger import get_logger
from src.utils.common import save_json, save_object

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    @staticmethod
    def _evaluate(y_true, probas):
        preds = (probas >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probas)),
            "brier": float(brier_score_loss(y_true, probas)),
        }

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
                "random_forest": RandomForestClassifier(
                    n_estimators=500,
                    max_depth=14,
                    random_state=42,
                    n_jobs=-1,
                ),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
            }

            if HAS_XGB:
                models["xgboost"] = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.04,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                )
                
            all_metrics = {}
            best_model_name = None
            best_model = None
            best_auc = -np.inf

            for name, model in models.items():
                logger.info("Training model: %s", name)
                model.fit(X_train, y_train)
                calibrated = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
                calibrated.fit(X_train, y_train)
                probas = calibrated.predict_proba(X_test)[:, 1]
                metrics = self._evaluate(y_test, probas)
                all_metrics[name] = metrics

                if metrics["roc_auc"] > best_auc:
                    best_auc = metrics["roc_auc"]
                    best_model_name = name
                    best_model = calibrated

            save_object(self.config.model_path, best_model)
            metrics_payload = {
                "best_model": best_model_name,
                "best_roc_auc": best_auc,
                "all_models": all_metrics,
            }
            save_json(self.config.metrics_path, metrics_payload)

            logger.info("Best model: %s (ROC-AUC: %.4f)", best_model_name, best_auc)
            logger.info("Saved model to %s", self.config.model_path)
            logger.info("Saved metrics to %s", self.config.metrics_path)

            return best_model_name, best_auc

        except Exception as exc:
            raise ProjectException(exc, context="ModelTrainer.initiate_model_trainer") from exc
