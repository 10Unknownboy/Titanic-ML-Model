"""
Titanic ML — Model Definitions
================================
Diverse model collection and Stacking Ensemble implementation.

Base models span 4 algorithm families for maximum diversity:
  - Linear:   LogisticRegression
  - Bagging:  RandomForest
  - Boosting: GradientBoosting, XGBoost, LightGBM, CatBoost
  - Margin:   SVC (RBF kernel)

The StackingEnsemble trains OOF (out-of-fold) predictions from each
base model, then feeds them to a meta-learner for final predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import Config
from src.utils import get_logger

logger = get_logger("models")


def get_base_models(config: Config | None = None) -> list[tuple[str, Any]]:
    """Create the collection of base models.

    Args:
        config: Project configuration with hyperparameters.

    Returns:
        List of (name, model) tuples.
    """
    config = config or Config()
    mc = config.models_config
    seed = config.random_seed

    models: list[tuple[str, Any]] = []

    # 1. Logistic Regression — linear decision boundary
    lr_cfg = mc.get("logistic_regression", {})
    models.append((
        "LogisticRegression",
        LogisticRegression(
            C=lr_cfg.get("C", 0.1),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver=lr_cfg.get("solver", "liblinear"),
            random_state=seed,
        ),
    ))

    # 2. Random Forest — bagging
    rf_cfg = mc.get("random_forest", {})
    models.append((
        "RandomForest",
        RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth", 5),
            min_samples_split=rf_cfg.get("min_samples_split", 4),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=seed,
            n_jobs=-1,
        ),
    ))

    # 3. Gradient Boosting (sklearn) — boosting
    gb_cfg = mc.get("gradient_boosting", {})
    models.append((
        "GradientBoosting",
        GradientBoostingClassifier(
            n_estimators=gb_cfg.get("n_estimators", 300),
            learning_rate=gb_cfg.get("learning_rate", 0.05),
            max_depth=gb_cfg.get("max_depth", 3),
            min_samples_split=gb_cfg.get("min_samples_split", 4),
            subsample=gb_cfg.get("subsample", 0.8),
            random_state=seed,
        ),
    ))

    # 4. XGBoost
    try:
        from xgboost import XGBClassifier
        xgb_cfg = mc.get("xgboost", {})
        models.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 300),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                max_depth=xgb_cfg.get("max_depth", 3),
                min_child_weight=xgb_cfg.get("min_child_weight", 3),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                reg_alpha=xgb_cfg.get("reg_alpha", 0.1),
                reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
            ),
        ))
    except ImportError:
        logger.warning("XGBoost not installed — skipping.")

    # 5. LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgbm_cfg = mc.get("lightgbm", {})
        models.append((
            "LightGBM",
            LGBMClassifier(
                n_estimators=lgbm_cfg.get("n_estimators", 300),
                learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                max_depth=lgbm_cfg.get("max_depth", 3),
                num_leaves=lgbm_cfg.get("num_leaves", 8),
                min_child_samples=lgbm_cfg.get("min_child_samples", 10),
                subsample=lgbm_cfg.get("subsample", 0.8),
                colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
                reg_alpha=lgbm_cfg.get("reg_alpha", 0.1),
                reg_lambda=lgbm_cfg.get("reg_lambda", 1.0),
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            ),
        ))
    except ImportError:
        logger.warning("LightGBM not installed — skipping.")

    # 6. CatBoost
    try:
        from catboost import CatBoostClassifier
        cb_cfg = mc.get("catboost", {})
        models.append((
            "CatBoost",
            CatBoostClassifier(
                iterations=cb_cfg.get("iterations", 300),
                learning_rate=cb_cfg.get("learning_rate", 0.05),
                depth=cb_cfg.get("depth", 4),
                l2_leaf_reg=cb_cfg.get("l2_leaf_reg", 3.0),
                random_seed=seed,
                verbose=0,
            ),
        ))
    except ImportError:
        logger.warning("CatBoost not installed — skipping.")

    # 7. SVC — margin-based (requires probability calibration for stacking)
    svc_cfg = mc.get("svc", {})
    base_svc = SVC(
        C=svc_cfg.get("C", 1.0),
        kernel=svc_cfg.get("kernel", "rbf"),
        gamma=svc_cfg.get("gamma", "scale"),
        random_state=seed,
    )
    models.append((
        "SVC",
        CalibratedClassifierCV(estimator=base_svc, ensemble=False),
    ))

    logger.info("Initialized %d base models: %s", len(models), [m[0] for m in models])
    return models


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Two-level stacking ensemble.

    Level-0: Base models generate out-of-fold (OOF) probability predictions.
    Level-1: Meta-learner trained on the OOF predictions.

    At prediction time, base model predictions are averaged across all
    fold-trained copies, then fed to the meta-learner.
    """

    def __init__(
        self,
        base_models: list[tuple[str, Any]] | None = None,
        meta_model: Any | None = None,
        cv_folds: int = 10,
        random_state: int = 42,
    ) -> None:
        self.base_models = base_models or []
        self.meta_model = meta_model or LogisticRegression(C=1.0, max_iter=1000)
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Populated during fit
        self.trained_base_models_: list[list[Any]] = []
        self.oof_scores_: dict[str, float] = {}
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """Train the stacking ensemble.

        1. For each base model, train on K-1 folds and predict fold K → OOF.
        2. Train meta-learner on the OOF predictions matrix.

        Args:
            X: Preprocessed feature matrix.
            y: Target labels.

        Returns:
            Fitted self.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        kf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        self.trained_base_models_ = [[] for _ in range(n_models)]

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                cloned_model = clone(model)
                cloned_model.fit(X_fold_train, y_fold_train)

                # OOF probability for class 1
                proba = cloned_model.predict_proba(X_fold_val)[:, 1]
                oof_predictions[val_idx, model_idx] = proba

                self.trained_base_models_[model_idx].append(cloned_model)

            logger.debug("Fold %d/%d complete", fold_idx + 1, self.cv_folds)

        # Compute per-model OOF accuracy
        for model_idx, (name, _) in enumerate(self.base_models):
            oof_binary = (oof_predictions[:, model_idx] >= 0.5).astype(int)
            acc = (oof_binary == y).mean()
            self.oof_scores_[name] = acc
            logger.info("  OOF Accuracy — %s: %.4f", name, acc)

        # Train meta-learner on OOF predictions
        logger.info("Training meta-learner on OOF predictions...")
        self.meta_model.fit(oof_predictions, y)
        
        # Calculate true ensemble OOF score
        meta_preds = self.meta_model.predict(oof_predictions)
        self.ensemble_oof_score_ = (meta_preds == y).mean()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Preprocessed feature matrix.

        Returns:
            Binary predictions (0 or 1).
        """
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Preprocessed feature matrix.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features by averaging fold model predictions.

        Args:
            X: Preprocessed feature matrix.

        Returns:
            Meta-feature matrix of shape (n_samples, n_models).
        """
        n_models = len(self.base_models)
        meta_features = np.zeros((X.shape[0], n_models))

        for model_idx in range(n_models):
            fold_preds = []
            for fold_model in self.trained_base_models_[model_idx]:
                fold_preds.append(fold_model.predict_proba(X)[:, 1])
            meta_features[:, model_idx] = np.mean(fold_preds, axis=0)

        return meta_features


def get_voting_ensemble(
    base_models: list[tuple[str, Any]],
    config: Config | None = None,
) -> VotingClassifier:
    """Create a soft-voting ensemble as a simpler alternative to stacking.

    Args:
        base_models: List of (name, model) tuples.
        config: Project configuration.

    Returns:
        VotingClassifier instance.
    """
    config = config or Config()
    weights = config.ensemble_config.get("voting_weights")

    # Trim weights to match number of models
    if weights and len(weights) >= len(base_models):
        weights = weights[: len(base_models)]
    else:
        weights = None

    return VotingClassifier(
        estimators=base_models,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )
