from copy import deepcopy
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from src.config import DualModelTrainerConfig
from src.exception import ProjectException
from src.logger import get_logger
from src.utils.common import save_json, save_object
from src.utils.dual_data_utils import (
    FIRST_INNINGS_FEATURES,
    SECOND_INNINGS_FEATURES,
    SECOND_INNINGS_SCORE_FEATURES,
)

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False

logger = get_logger(__name__)


class DualModelTrainer:
    CV_SPLITS = 5
    CV_SPLITS_BALANCED = 4
    CV_SPLITS_FAST = 3
    MAX_TUNING_ROWS_DEFAULT = 40000
    MAX_TUNING_ROWS_BALANCED = 25000
    MAX_TUNING_ROWS_FAST = 15000
    TUNING_ITER_FULL = 24
    TUNING_ITER_BALANCED = 12
    TUNING_ITER_FAST = 6
    TUNING_CV_FULL = 4
    TUNING_CV_BALANCED = 3
    TUNING_CV_FAST = 2

    def __init__(self, config: DualModelTrainerConfig = DualModelTrainerConfig()):
        self.config = config
        profile = os.getenv("CRICKET_TRAIN_PROFILE", "balanced").strip().lower()
        if profile not in {"full", "balanced", "fast"}:
            profile = "balanced"
        self.train_profile = profile

        if self.train_profile == "full":
            self.cv_splits = self.CV_SPLITS
            default_max_tuning_rows = self.MAX_TUNING_ROWS_DEFAULT
            self.tuning_n_iter = self.TUNING_ITER_FULL
            self.tuning_cv_splits = self.TUNING_CV_FULL
        elif self.train_profile == "fast":
            self.cv_splits = self.CV_SPLITS_FAST
            default_max_tuning_rows = self.MAX_TUNING_ROWS_FAST
            self.tuning_n_iter = self.TUNING_ITER_FAST
            self.tuning_cv_splits = self.TUNING_CV_FAST
        else:
            self.cv_splits = self.CV_SPLITS_BALANCED
            default_max_tuning_rows = self.MAX_TUNING_ROWS_BALANCED
            self.tuning_n_iter = self.TUNING_ITER_BALANCED
            self.tuning_cv_splits = self.TUNING_CV_BALANCED

        self.max_tuning_rows = int(os.getenv("CRICKET_MAX_TUNING_ROWS", str(default_max_tuning_rows)))

    @staticmethod
    def _regression_metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def _classification_metrics(y_true, y_proba):
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "brier": float(brier_score_loss(y_true, y_proba)),
            "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        }

    @staticmethod
    def _build_pipeline(preprocessor, estimator):
        return Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("model", deepcopy(estimator)),
            ]
        )

    def _sample_tuning_frame(self, train_df: pd.DataFrame) -> pd.DataFrame:
        if len(train_df) <= self.max_tuning_rows:
            return train_df

        unique_matches = train_df["match_id"].dropna().unique()
        if len(unique_matches) == 0:
            return train_df.sample(n=self.max_tuning_rows, random_state=42)

        rng = np.random.default_rng(42)
        shuffled_matches = rng.permutation(unique_matches)
        grouped = train_df.groupby("match_id")

        selected_frames = []
        selected_rows = 0
        for match_id in shuffled_matches:
            if match_id not in grouped.groups:
                continue
            idx = grouped.groups[match_id]
            selected_frames.append(train_df.loc[idx])
            selected_rows += len(idx)
            if selected_rows >= self.max_tuning_rows:
                break

        sampled = pd.concat(selected_frames, ignore_index=True) if selected_frames else train_df
        if len(sampled) > self.max_tuning_rows:
            sampled = sampled.sample(n=self.max_tuning_rows, random_state=42).reset_index(drop=True)

        logger.info("Using sampled tuning frame: %s rows (from %s)", len(sampled), len(train_df))
        return sampled

    def _strict_regression_models(self, include_dl=False):
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=280,
                max_depth=14,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
            "extra_trees_regressor": ExtraTreesRegressor(
                n_estimators=350,
                max_depth=14,
                random_state=42,
                n_jobs=-1,
            ),
        }

        if self.train_profile == "fast":
            models.pop("random_forest_regressor", None)
            models.pop("extra_trees_regressor", None)
            models.pop("gradient_boosting_regressor", None)

        if HAS_XGB:
            models["xgboost_regressor"] = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
        else:
            logger.warning("XGBoost is not installed. Mandatory XGBoost regressor is unavailable.")

        if include_dl:
            models["mlp_regressor_dl"] = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                max_iter=180,
                early_stopping=True,
                random_state=42,
            )

        return models

    def _strict_classification_models(self, include_dl=False):
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=42,
            ),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=280,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting_classifier": GradientBoostingClassifier(random_state=42),
            "extra_trees_classifier": ExtraTreesClassifier(
                n_estimators=350,
                max_depth=14,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        }

        if self.train_profile == "fast":
            models.pop("random_forest_classifier", None)
            models.pop("extra_trees_classifier", None)
            models.pop("gradient_boosting_classifier", None)
            models.pop("hist_gradient_boosting_classifier", None)

        if HAS_XGB:
            models["xgboost_classifier"] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            logger.warning("XGBoost is not installed. Mandatory XGBoost classifier is unavailable.")

        if include_dl:
            models["mlp_classifier_dl"] = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                max_iter=180,
                early_stopping=True,
                random_state=42,
            )
        return models
    
    @staticmethod
    def _classification_tuning_space(model_name):
        spaces = {
            "logistic_regression": {
                "model__C": [0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 4.0, 8.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
            "random_forest_classifier": {
                "model__n_estimators": [220, 300, 380, 460],
                "model__max_depth": [8, 10, 12, 14, 18, None],
                "model__min_samples_leaf": [1, 2, 3, 5],
                "model__max_features": ["sqrt", "log2", None],
            },
            "gradient_boosting_classifier": {
                "model__n_estimators": [120, 180, 240, 320],
                "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            "extra_trees_classifier": {
                "model__n_estimators": [220, 320, 420, 520],
                "model__max_depth": [8, 10, 12, 14, 18, None],
                "model__min_samples_leaf": [1, 2, 3, 5],
                "model__max_features": ["sqrt", "log2", None],
            },
            "hist_gradient_boosting_classifier": {
                "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
                "model__max_depth": [None, 6, 10, 14],
                "model__max_leaf_nodes": [15, 31, 63, 127],
                "model__min_samples_leaf": [10, 20, 40, 60],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
            "xgboost_classifier": {
                "model__n_estimators": [180, 260, 340, 420],
                "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
                "model__max_depth": [3, 4, 5, 6],
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.75, 0.9, 1.0],
            },
        }
        return spaces.get(model_name)

    @staticmethod
    def _regression_tuning_space(model_name):
        spaces = {
            "random_forest_regressor": {
                "model__n_estimators": [220, 300, 380, 460],
                "model__max_depth": [8, 10, 12, 14, 18, None],
                "model__min_samples_leaf": [1, 2, 3, 5],
                "model__max_features": ["sqrt", "log2", None],
            },
            "gradient_boosting_regressor": {
                "model__n_estimators": [120, 180, 240, 320],
                "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            "extra_trees_regressor": {
                "model__n_estimators": [220, 320, 420, 520],
                "model__max_depth": [8, 10, 12, 14, 18, None],
                "model__min_samples_leaf": [1, 2, 3, 5],
                "model__max_features": ["sqrt", "log2", None],
            },
            "hist_gradient_boosting_regressor": {
                "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
                "model__max_depth": [None, 6, 10, 14],
                "model__max_leaf_nodes": [15, 31, 63, 127],
                "model__min_samples_leaf": [10, 20, 40, 60],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
            "xgboost_regressor": {
                "model__n_estimators": [180, 260, 340, 420],
                "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
                "model__max_depth": [3, 4, 5, 6],
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.75, 0.9, 1.0],
            },
        }
        return spaces.get(model_name)

    def _tuned_regression_models(self, preprocessor, train_df, feature_cols, target_col, include_dl=False):
        if self.train_profile == "fast":
            logger.info("Fast training profile active: skipping expensive hyperparameter tuning.")
            return {}

        tuned = {}
        tuning_df = self._sample_tuning_frame(train_df)
        X_train = tuning_df[feature_cols]
        y_train = tuning_df[target_col].values
        groups = tuning_df["match_id"]
        cv = GroupKFold(n_splits=self.tuning_cv_splits)

        for model_name, model in self._strict_regression_models(include_dl=include_dl).items():
            if self.train_profile == "balanced" and target_col == "second_innings_total_target":
                if model_name in {"extra_trees_regressor", "gradient_boosting_regressor"}:
                    continue
            space = self._regression_tuning_space(model_name)
            if not space:
                continue

            try:
                logger.info("Hyperparameter tuning regression model: %s", model_name)
                search = RandomizedSearchCV(
                    estimator=self._build_pipeline(preprocessor, model),
                    param_distributions=space,
                    n_iter=self.tuning_n_iter,
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    random_state=42,
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                search.fit(X_train, y_train, groups=groups)

                tuned_model = deepcopy(search.best_estimator_.named_steps["model"])
                tuned_name = f"{model_name}_tuned"
                tuned[tuned_name] = tuned_model
                logger.info(
                    "Best tuned %s score: %.5f with params: %s",
                    model_name,
                    float(search.best_score_),
                    search.best_params_,
                )
            except Exception as exc:
                logger.warning("Tuning failed for %s. Using baseline config. Reason: %s", model_name, exc)

        return tuned

    def _tuned_classification_models(self, preprocessor, train_df, include_dl=False):
        if self.train_profile == "fast":
            logger.info("Fast training profile active: skipping expensive hyperparameter tuning.")
            return {}

        tuned = {}

        tuning_df = self._sample_tuning_frame(train_df)
        X_train = tuning_df[SECOND_INNINGS_FEATURES]
        y_train = tuning_df["win"].values
        groups = tuning_df["match_id"]
        cv = GroupKFold(n_splits=self.tuning_cv_splits)

        for model_name, model in self._strict_classification_models(include_dl=include_dl).items():
            space = self._classification_tuning_space(model_name)
            if not space:
                continue

            try:
                logger.info("Hyperparameter tuning second-innings model: %s", model_name)
                search = RandomizedSearchCV(
                    estimator=self._build_pipeline(preprocessor, model),
                    param_distributions=space,
                    n_iter=self.tuning_n_iter,
                    scoring="neg_brier_score",
                    cv=cv,
                    random_state=42,
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                search.fit(X_train, y_train, groups=groups)

                tuned_model = deepcopy(search.best_estimator_.named_steps["model"])
                tuned_name = f"{model_name}_tuned"
                tuned[tuned_name] = tuned_model
                logger.info(
                    "Best tuned %s score: %.5f with params: %s",
                    model_name,
                    float(search.best_score_),
                    search.best_params_,
                )
            except Exception as exc:
                logger.warning("Tuning failed for %s. Using baseline config. Reason: %s", model_name, exc)

        return tuned

    def _evaluate_regression(self, preprocessor, model_name, model, train_df, test_df, feature_cols, target_col):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values

        pipeline = self._build_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        test_metrics = self._regression_metrics(y_test, test_pred)

        gkf = GroupKFold(n_splits=self.cv_splits)
        cv_scores = []
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups=train_df["match_id"]):
            fold_pipe = self._build_pipeline(preprocessor, model)
            fold_pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            val_pred = fold_pipe.predict(X_train.iloc[val_idx])
            cv_scores.append(self._regression_metrics(y_train[val_idx], val_pred))

        row = {
            "model": model_name,
            "cv_mae": float(np.mean([m["mae"] for m in cv_scores])),
            "cv_rmse": float(np.mean([m["rmse"] for m in cv_scores])),
            "cv_r2": float(np.mean([m["r2"] for m in cv_scores])),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }
        return pipeline, row

    def _evaluate_classification(self, preprocessor, model_name, model, train_df, test_df):
        X_train = train_df[SECOND_INNINGS_FEATURES]
        y_train = train_df["win"].values
        X_test = test_df[SECOND_INNINGS_FEATURES]
        y_test = test_df["win"].values

        base_pipeline = self._build_pipeline(preprocessor, model)
        base_pipeline.fit(X_train, y_train)
        base_proba = base_pipeline.predict_proba(X_test)[:, 1]
        base_metrics = self._classification_metrics(y_test, base_proba)

        calibrated = base_pipeline
        calibrated_metrics = base_metrics
        selected_calibration_method = "none"
        for calibration_method in ("sigmoid", "isotonic"):
            try:
                candidate = CalibratedClassifierCV(estimator=deepcopy(base_pipeline), method=calibration_method, cv=3)
                candidate.fit(X_train, y_train)
                calibrated_proba = candidate.predict_proba(X_test)[:, 1]
                candidate_metrics = self._classification_metrics(y_test, calibrated_proba)

                better_brier = candidate_metrics["brier"] < calibrated_metrics["brier"]
                tie_brier_better_auc = np.isclose(candidate_metrics["brier"], calibrated_metrics["brier"]) and (
                    candidate_metrics["roc_auc"] > calibrated_metrics["roc_auc"]
                )
                if better_brier or tie_brier_better_auc:
                    calibrated = candidate
                    calibrated_metrics = candidate_metrics
                    selected_calibration_method = calibration_method
            except Exception as exc:
                logger.warning("Skipping %s calibration for %s due to: %s", calibration_method, model_name, exc)

        gkf = GroupKFold(n_splits=self.cv_splits)
        cv_scores = []
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups=train_df["match_id"]):
            fold_pipe = self._build_pipeline(preprocessor, model)
            fold_pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            val_proba = fold_pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
            cv_scores.append(self._classification_metrics(y_train[val_idx], val_proba))

        row = {
            "model": model_name,
            "cv_accuracy": float(np.mean([m["accuracy"] for m in cv_scores])),
            "cv_precision": float(np.mean([m["precision"] for m in cv_scores])),
            "cv_recall": float(np.mean([m["recall"] for m in cv_scores])),
            "cv_f1": float(np.mean([m["f1"] for m in cv_scores])),
            "cv_roc_auc": float(np.mean([m["roc_auc"] for m in cv_scores])),
            "cv_brier": float(np.mean([m["brier"] for m in cv_scores])),
            "test_accuracy": base_metrics["accuracy"],
            "test_precision": base_metrics["precision"],
            "test_recall": base_metrics["recall"],
            "test_f1": base_metrics["f1"],
            "test_roc_auc": base_metrics["roc_auc"],
            "test_brier": base_metrics["brier"],
            "test_log_loss": base_metrics["log_loss"],
            "calibrated_test_accuracy": calibrated_metrics["accuracy"],
            "calibrated_test_precision": calibrated_metrics["precision"],
            "calibrated_test_recall": calibrated_metrics["recall"],
            "calibrated_test_f1": calibrated_metrics["f1"],
            "calibrated_test_roc_auc": calibrated_metrics["roc_auc"],
            "calibrated_test_brier": calibrated_metrics["brier"],
            "calibrated_test_log_loss": calibrated_metrics["log_loss"],
            "selected_calibration_method": selected_calibration_method,
        }
        return calibrated, row

    def _save_feature_importance(self, trained_model, X_test, y_test, features, task_key):
        scoring = "neg_root_mean_squared_error" if task_key != "second_innings" else "roc_auc"
        result = permutation_importance(
            trained_model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring=scoring,
            n_jobs=-1,
        )
        names = list(features)
        importance_df = pd.DataFrame(
            {
                "feature": names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        out_path = self.config.summary_metrics_path.parent / f"{task_key}_feature_importance.csv"
        importance_df.to_csv(out_path, index=False)
        return out_path

    def initiate(self, transformed_bundle):
        try:
            first_train = transformed_bundle["first_train_df"]
            first_test = transformed_bundle["first_test_df"]
            second_train = transformed_bundle["second_train_df"]
            second_test = transformed_bundle["second_test_df"]
            second_score_train = transformed_bundle["second_score_train_df"]
            second_score_test = transformed_bundle["second_score_test_df"]
            first_preprocessor = transformed_bundle["first_preprocessor"]
            second_preprocessor = transformed_bundle["second_preprocessor"]

            include_dl = False

            # First innings regression
            regression_models = self._strict_regression_models(include_dl=include_dl)
            tuned_regression_models = self._tuned_regression_models(
                first_preprocessor,
                first_train,
                FIRST_INNINGS_FEATURES,
                "projected_total_target",
                include_dl=include_dl,
            )
            regression_models.update(tuned_regression_models)
            first_rows = []
            first_best_model = None
            first_best_name = None
            first_best_rmse = np.inf

            for name, model in regression_models.items():
                logger.info("Training first-innings model: %s", name)
                try:
                    pipe, row = self._evaluate_regression(
                        first_preprocessor,
                        name,
                        model,
                        first_train,
                        first_test,
                        FIRST_INNINGS_FEATURES,
                        "projected_total_target",
                    )
                    first_rows.append(row)
                    if row["test_rmse"] < first_best_rmse:
                        first_best_rmse = row["test_rmse"]
                        first_best_name = name
                        first_best_model = pipe
                except Exception as exc:
                    logger.warning("Skipping first-innings model %s due to training failure: %s", name, exc)

            if first_best_model is None:
                raise ValueError("No valid first-innings regression model could be trained.")

            first_leaderboard = pd.DataFrame(first_rows).sort_values("test_rmse", ascending=True)
            first_leaderboard.to_csv(self.config.first_model_leaderboard_path, index=False)
            save_object(self.config.first_best_model_path, first_best_model)

            # Second innings classification
            classification_models = self._strict_classification_models(include_dl=include_dl)
            tuned_classification_models = self._tuned_classification_models(
                second_preprocessor,
                second_train,
                include_dl=include_dl,
            )
            classification_models.update(tuned_classification_models)

            second_rows = []
            second_best_model = None
            second_best_name = None
            second_best_auc = -np.inf
            second_best_brier = np.inf

            for name, model in classification_models.items():
                logger.info("Training second-innings model: %s", name)
                try:
                    pipe, row = self._evaluate_classification(
                        second_preprocessor,
                        name,
                        model,
                        second_train,
                        second_test,
                    )
                    second_rows.append(row)
                    if (
                        row["calibrated_test_roc_auc"] > second_best_auc
                        or (
                            np.isclose(row["calibrated_test_roc_auc"], second_best_auc)
                            and row["calibrated_test_brier"] < second_best_brier
                        )
                    ):
                        second_best_auc = row["calibrated_test_roc_auc"]
                        second_best_brier = row["calibrated_test_brier"]
                        second_best_name = name
                        second_best_model = pipe
                except Exception as exc:
                    logger.warning("Skipping second-innings model %s due to training failure: %s", name, exc)

            if second_best_model is None:
                raise ValueError("No valid second-innings classification model could be trained.")

            second_leaderboard = pd.DataFrame(second_rows).sort_values(
                ["calibrated_test_roc_auc", "calibrated_test_brier"],
                ascending=[False, True],
            )
            second_leaderboard.to_csv(self.config.second_model_leaderboard_path, index=False)
            save_object(self.config.second_best_model_path, second_best_model)

            # Second innings score regression
            second_score_regression_models = self._strict_regression_models(include_dl=include_dl)
            tuned_second_score_regression_models = self._tuned_regression_models(
                second_preprocessor,
                second_score_train,
                SECOND_INNINGS_SCORE_FEATURES,
                "second_innings_total_target",
                include_dl=include_dl,
            )
            second_score_regression_models.update(tuned_second_score_regression_models)
            second_score_rows = []
            second_score_best_model = None
            second_score_best_name = None
            second_score_best_rmse = np.inf

            for name, model in second_score_regression_models.items():
                logger.info("Training second-innings score model: %s", name)
                try:
                    pipe, row = self._evaluate_regression(
                        second_preprocessor,
                        name,
                        model,
                        second_score_train,
                        second_score_test,
                        SECOND_INNINGS_SCORE_FEATURES,
                        "second_innings_total_target",
                    )
                    second_score_rows.append(row)
                    if row["test_rmse"] < second_score_best_rmse:
                        second_score_best_rmse = row["test_rmse"]
                        second_score_best_name = name
                        second_score_best_model = pipe
                except Exception as exc:
                    logger.warning("Skipping second-innings score model %s due to training failure: %s", name, exc)

            if second_score_best_model is None:
                raise ValueError("No valid second-innings score regression model could be trained.")

            second_score_leaderboard = pd.DataFrame(second_score_rows).sort_values("test_rmse", ascending=True)
            second_score_leaderboard.to_csv(self.config.second_score_model_leaderboard_path, index=False)
            save_object(self.config.second_score_best_model_path, second_score_best_model)

            first_importance_path = self._save_feature_importance(
                first_best_model,
                first_test[FIRST_INNINGS_FEATURES],
                first_test["projected_total_target"].values,
                FIRST_INNINGS_FEATURES,
                "first_innings",
            )
            second_importance_path = self._save_feature_importance(
                second_best_model,
                second_test[SECOND_INNINGS_FEATURES],
                second_test["win"].values,
                SECOND_INNINGS_FEATURES,
                "second_innings",
            )
            second_score_importance_path = self._save_feature_importance(
                second_score_best_model,
                second_score_test[SECOND_INNINGS_SCORE_FEATURES],
                second_score_test["second_innings_total_target"].values,
                SECOND_INNINGS_SCORE_FEATURES,
                "second_innings_score",
            )

            leakage_guard = {
                "second_innings_auc_too_high": bool(second_best_auc > 0.95),
                "message": "Re-check leakage if calibrated_test_roc_auc is too close to 1.0.",
            }

            summary = {
                "first_innings": {
                    "best_model": first_best_name,
                    "selection_metric": "lowest_test_rmse",
                    "best_test_rmse": float(first_best_rmse),
                },
                "second_innings": {
                    "best_model": second_best_name,
                    "selection_metric": "highest_calibrated_test_roc_auc_then_lowest_calibrated_test_brier",
                    "best_calibrated_test_roc_auc": float(second_best_auc),
                    "best_calibrated_test_brier": float(second_best_brier),
                },
                "second_innings_score": {
                    "best_model": second_score_best_name,
                    "selection_metric": "lowest_test_rmse",
                    "best_test_rmse": float(second_score_best_rmse),
                },
                "leakage_guard": leakage_guard,
                "models_compared": {
                    "classification": list(classification_models.keys()),
                    "regression": list(regression_models.keys()),
                },
                "feature_importance_files": {
                    "first_innings": str(first_importance_path),
                    "second_innings": str(second_importance_path),
                    "second_innings_score": str(second_score_importance_path),
                },
                "xgboost_available": HAS_XGB,
            }
            save_json(self.config.summary_metrics_path, summary)

            logger.info(
                "Dual model training complete. First best: %s | Second win best: %s | Second score best: %s",
                first_best_name,
                second_best_name,
                second_score_best_name,
            )

            return {
                "first_best_model": first_best_name,
                "second_best_model": second_best_name,
                "second_score_best_model": second_score_best_name,
                "first_leaderboard": self.config.first_model_leaderboard_path,
                "second_leaderboard": self.config.second_model_leaderboard_path,
                "second_score_leaderboard": self.config.second_score_model_leaderboard_path,
                "summary": self.config.summary_metrics_path,
            }

        except Exception as exc:
            raise ProjectException(exc, context="DualModelTrainer.initiate") from exc
