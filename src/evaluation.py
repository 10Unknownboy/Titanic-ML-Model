"""
Titanic ML — Evaluation
========================
Model evaluation metrics, cross-validation reporting, and comparison.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import Config
from src.utils import get_logger

if TYPE_CHECKING:
    from src.training import TrainingResult

logger = get_logger("evaluation")


def print_evaluation_report(result: "TrainingResult", config: Config | None = None) -> None:
    """Print a comprehensive evaluation report.

    Args:
        result: Training result with model and CV scores.
        config: Project configuration.
    """
    config = config or Config()

    print()
    print("=" * 65)
    print("  MODEL EVALUATION REPORT")
    print("=" * 65)

    # --- Per-model OOF scores ---
    if result.cv_scores:
        print("\n  [*] Model Scores (Out-of-Fold Accuracy):")
        print("  " + "-" * 45)

        for name, score in result.cv_scores.items():
            if isinstance(score, float):
                bar = "#" * int(score * 40)
                print(f"  {name:<25s} {score:.4f}  {bar}")
            elif isinstance(score, list):
                arr = np.array(score)
                print(f"  {name:<25s} {arr.mean():.4f} +/- {arr.std():.4f}")

    # --- Ensemble OOF accuracy (if stacking) ---
    from src.models import StackingEnsemble
    if isinstance(result.model, StackingEnsemble):
        if hasattr(result.model, "ensemble_oof_score_"):
            oof_acc = result.model.ensemble_oof_score_
            print(f"\n  [BEST] Stacking Ensemble OOF Accuracy: {oof_acc:.4f}")

    # --- Training time ---
    print(f"\n  Time: {result.training_time:.1f}s")

    # --- Stability check ---
    if result.cv_scores:
        scores = [v for v in result.cv_scores.values() if isinstance(v, float)]
        if scores:
            std = np.std(scores)
            if std > 0.05:
                print("  [!] Warning: High variance across models. Check for overfitting.")
            else:
                print("  [OK] Model scores are stable. Good generalization expected.")

    print("=" * 65)
    print()


def evaluate_individual_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: list[tuple[str, Any]],
    config: Config | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate each model individually with cross-validation.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        models: List of (name, model) tuples.
        config: Project configuration.

    Returns:
        Dict mapping model names to their evaluation metrics.
    """
    config = config or Config()
    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed
    )

    results = {}
    for name, model in models:
        logger.info("Evaluating %s...", name)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "fold_scores": scores.tolist(),
        }
        logger.info("  %s — CV Accuracy: %.4f ± %.4f", name, scores.mean(), scores.std())

    return results


def get_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance from a fitted model.

    Supports tree-based models, linear models, and ensembles.

    Args:
        model: Fitted model.
        feature_names: List of feature names matching the model input.

    Returns:
        Dict mapping feature names to importance scores (sorted descending).
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        logger.warning("Model type %s does not support feature importance.", type(model).__name__)
        return {}

    if len(importances) != len(feature_names):
        logger.warning(
            "Feature count mismatch: %d importances vs %d names",
            len(importances), len(feature_names),
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_dict = dict(zip(feature_names, importances))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def compare_submissions(file1: str, file2: str) -> dict[str, Any]:
    """Compare two submission files.

    Args:
        file1: Path to first submission CSV.
        file2: Path to second submission CSV.

    Returns:
        Comparison metrics.
    """
    import pandas as pd

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if len(df1) != len(df2):
        return {"error": f"Row count mismatch: {len(df1)} vs {len(df2)}"}

    agreement = (df1["Survived"].values == df2["Survived"].values).mean() * 100
    differences = (df1["Survived"].values != df2["Survived"].values).sum()

    # Find which passengers differ
    diff_mask = df1["Survived"] != df2["Survived"]
    diff_ids = df1.loc[diff_mask, "PassengerId"].tolist()

    return {
        "agreement_pct": agreement,
        "differences": differences,
        "diff_passenger_ids": diff_ids,
    }
