"""
Titanic ML — Training Pipeline
================================
End-to-end training: feature engineering → preprocessing → model training.
Supports stacking ensemble, individual model training, and Optuna optimization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import Config
from src.data_loader import load_data
from src.evaluation import print_evaluation_report
from src.feature_engineering import TitanicFeatureEngineer
from src.models import StackingEnsemble, get_base_models, get_voting_ensemble
from src.preprocessing import get_preprocessor
from src.submission import generate_submission
from src.utils import get_logger, set_seed

logger = get_logger("training")


class TrainingResult:
    """Container for training artifacts."""

    def __init__(self) -> None:
        self.preprocessor: Any = None
        self.model: Any = None
        self.feature_engineer: TitanicFeatureEngineer | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.test_ids: pd.Series | None = None
        self.cv_scores: dict[str, Any] = {}
        self.predictions: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None
        self.training_time: float = 0.0


def train_pipeline(
    config: Config | None = None,
    optimize: bool = False,
    model_name: str | None = None,
) -> TrainingResult:
    """Execute the full training pipeline.

    Args:
        config: Project configuration.
        optimize: Whether to run Optuna hyperparameter optimization.
        model_name: Train only a specific model (e.g., 'xgboost'). None = ensemble.

    Returns:
        TrainingResult with all artifacts.
    """
    config = config or Config()
    set_seed(config.random_seed)

    result = TrainingResult()
    start_time = time.time()

    # --- 1. Load data ---
    logger.info("=" * 65)
    logger.info("  TITANIC ML — TRAINING PIPELINE")
    logger.info("=" * 65)

    train_df, test_df = load_data(config)

    # --- 2. Feature engineering ---
    engineer = TitanicFeatureEngineer(config)
    X_train_raw, y_train, X_test_raw, test_ids = engineer.fit_transform(train_df, test_df)
    result.feature_engineer = engineer
    result.test_ids = test_ids

    # --- 3. Preprocessing ---
    preprocessor = get_preprocessor(config)
    X_train = preprocessor.fit_transform(X_train_raw, y_train.values)
    X_test = preprocessor.transform(X_test_raw)

    result.preprocessor = preprocessor
    result.X_train = X_train
    result.y_train = y_train.values
    result.X_test = X_test

    logger.info("Preprocessed shapes: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    # --- 4. Get models ---
    base_models = get_base_models(config)

    if model_name:
        # Train a single model
        matched = [(n, m) for n, m in base_models if n.lower() == model_name.lower()]
        if not matched:
            available = [n for n, _ in base_models]
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        name, model = matched[0]
        logger.info("Training single model: %s", name)
        model.fit(X_train, y_train.values)
        result.model = model
        result.predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            result.probabilities = model.predict_proba(X_test)[:, 1]
    else:
        # Train ensemble
        if optimize:
            base_models = _optimize_hyperparameters(base_models, X_train, y_train.values, config)

        ensemble_method = config.ensemble_config.get("method", "stacking")

        if ensemble_method == "stacking":
            logger.info("Training Stacking Ensemble (%d base models, %d-fold CV)...",
                       len(base_models), config.cv_folds)

            meta_cfg = config.models_config.get("meta_learner", {})
            meta_model = LogisticRegression(
                C=meta_cfg.get("C", 1.0),
                max_iter=meta_cfg.get("max_iter", 1000),
                random_state=config.random_seed,
            )

            ensemble = StackingEnsemble(
                base_models=base_models,
                meta_model=meta_model,
                cv_folds=config.cv_folds,
                random_state=config.random_seed,
            )
            ensemble.fit(X_train, y_train.values)

            result.model = ensemble
            result.cv_scores = ensemble.oof_scores_
            result.predictions = ensemble.predict(X_test)
            result.probabilities = ensemble.predict_proba(X_test)[:, 1]

        else:  # voting
            logger.info("Training Voting Ensemble (%d models)...", len(base_models))
            voting = get_voting_ensemble(base_models, config)

            # Cross-validation for evaluation
            cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True,
                               random_state=config.random_seed)
            cv_scores = cross_val_score(voting, X_train, y_train.values,
                                       cv=cv, scoring="accuracy", n_jobs=-1)
            result.cv_scores = {
                "voting_mean": cv_scores.mean(),
                "voting_std": cv_scores.std(),
                "voting_folds": cv_scores.tolist(),
            }

            # Train on full data
            voting.fit(X_train, y_train.values)
            result.model = voting
            result.predictions = voting.predict(X_test)
            result.probabilities = voting.predict_proba(X_test)[:, 1]

    result.training_time = time.time() - start_time

    # --- 5. Evaluate ---
    print_evaluation_report(result, config)

    # --- 6. Save model ---
    _save_model(result, config)

    # --- 7. Generate submission ---
    generate_submission(result.predictions, test_ids, config)

    logger.info("Pipeline complete in %.1f seconds.", result.training_time)

    return result


def _optimize_hyperparameters(
    base_models: list[tuple[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
) -> list[tuple[str, Any]]:
    """Optimize hyperparameters using Optuna.

    Currently optimizes XGBoost and LightGBM parameters.

    Args:
        base_models: Original model list.
        X: Training features.
        y: Training labels.
        config: Project configuration.

    Returns:
        Updated model list with optimized parameters.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — skipping optimization.")
        return base_models

    n_trials = config.training.get("optuna_trials", 50)
    seed = config.random_seed

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    optimized_models = []

    for name, model in base_models:
        if name == "XGBoost":
            logger.info("Optimizing XGBoost hyperparameters (%d trials)...", n_trials)
            optimized = _optimize_xgboost(X, y, cv, n_trials, seed)
            optimized_models.append(("XGBoost", optimized))

        elif name == "LightGBM":
            logger.info("Optimizing LightGBM hyperparameters (%d trials)...", n_trials)
            optimized = _optimize_lightgbm(X, y, cv, n_trials, seed)
            optimized_models.append(("LightGBM", optimized))

        elif name == "GradientBoosting":
            logger.info("Optimizing GradientBoosting hyperparameters (%d trials)...", n_trials)
            optimized = _optimize_gradient_boosting(X, y, cv, n_trials, seed)
            optimized_models.append(("GradientBoosting", optimized))

        else:
            optimized_models.append((name, model))

    return optimized_models


def _optimize_xgboost(X, y, cv, n_trials, seed):
    """Optimize XGBoost hyperparameters."""
    import optuna
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": seed,
            "eval_metric": "logloss",
            "verbosity": 0,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("XGBoost best CV: %.4f", study.best_value)
    best = study.best_params
    best.update({"random_state": seed, "eval_metric": "logloss", "verbosity": 0, "n_jobs": -1})
    return XGBClassifier(**best)


def _optimize_lightgbm(X, y, cv, n_trials, seed):
    """Optimize LightGBM hyperparameters."""
    import optuna
    from lightgbm import LGBMClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": seed,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("LightGBM best CV: %.4f", study.best_value)
    best = study.best_params
    best.update({"random_state": seed, "verbose": -1, "n_jobs": -1})
    return LGBMClassifier(**best)


def _optimize_gradient_boosting(X, y, cv, n_trials, seed):
    """Optimize GradientBoosting hyperparameters."""
    import optuna
    from sklearn.ensemble import GradientBoostingClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": seed,
        }
        model = GradientBoostingClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("GradientBoosting best CV: %.4f", study.best_value)
    return GradientBoostingClassifier(**study.best_params, random_state=seed)


def _save_model(result: TrainingResult, config: Config) -> None:
    """Save trained model and preprocessor to disk.

    Args:
        result: Training result containing model and preprocessor.
        config: Project configuration.
    """
    model_dir = config.model_dir

    artifacts = {
        "model": result.model,
        "preprocessor": result.preprocessor,
        "cv_scores": result.cv_scores,
        "training_time": result.training_time,
    }

    model_path = model_dir / "titanic_model.joblib"
    joblib.dump(artifacts, model_path)
    logger.info("Model saved to: %s", model_path)

    # Also save preprocessor separately for dashboard use
    preprocessor_path = model_dir / "preprocessor.joblib"
    joblib.dump(result.preprocessor, preprocessor_path)
    logger.info("Preprocessor saved to: %s", preprocessor_path)


# Need this import at module level for the meta-learner
from sklearn.linear_model import LogisticRegression  # noqa: E402
