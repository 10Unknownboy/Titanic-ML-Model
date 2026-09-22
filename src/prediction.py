"""
Titanic ML — Prediction
========================
Load saved model and generate predictions on new data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.feature_engineering import TitanicFeatureEngineer, engineer_single_passenger
from src.preprocessing import get_preprocessor
from src.utils import get_logger

logger = get_logger("prediction")


def load_model(
    model_path: str | Path | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Load a saved model and its artifacts.

    Args:
        model_path: Path to the saved model file. Uses default if None.
        config: Project configuration.

    Returns:
        Dict with keys: 'model', 'preprocessor', 'cv_scores', 'training_time'.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    config = config or Config()

    if model_path is None:
        model_path = config.model_dir / "titanic_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run 'python train.py' first to train a model."
        )

    logger.info("Loading model from: %s", model_path)
    artifacts = joblib.load(model_path)
    logger.info("Model loaded successfully.")

    return artifacts


def predict_test_set(
    model_path: str | Path | None = None,
    config: Config | None = None,
) -> tuple[np.ndarray, pd.Series, np.ndarray | None]:
    """Load model and generate predictions on the test set.

    Args:
        model_path: Path to saved model. Uses default if None.
        config: Project configuration.

    Returns:
        Tuple of (predictions, test_ids, probabilities).
    """
    config = config or Config()

    # Load model artifacts
    artifacts = load_model(model_path, config)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    # Load and process test data
    from src.data_loader import load_data
    train_df, test_df = load_data(config)

    engineer = TitanicFeatureEngineer(config)
    _, _, X_test_raw, test_ids = engineer.fit_transform(train_df, test_df)

    # Preprocess
    X_test = preprocessor.transform(X_test_raw)

    # Predict
    predictions = model.predict(X_test)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    logger.info("Generated %d predictions.", len(predictions))

    return predictions, test_ids, probabilities


def predict_single_passenger(
    passenger: dict,
    model_path: str | Path | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Predict survival for a single passenger (for dashboard use).

    Args:
        passenger: Dict with raw passenger features.
            Required keys: Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
        model_path: Path to saved model.
        config: Project configuration.

    Returns:
        Dict with 'survived' (bool), 'probability' (float), 'label' (str).
    """
    config = config or Config()

    artifacts = load_model(model_path, config)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    # Engineer features for the single passenger
    X_single = engineer_single_passenger(passenger, config)

    # Preprocess
    X_processed = preprocessor.transform(X_single)

    # Predict
    prediction = model.predict(X_processed)[0]
    probability = 0.5
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_processed)[0, 1]

    return {
        "survived": bool(prediction),
        "probability": float(probability),
        "label": "Survived ✅" if prediction == 1 else "Did Not Survive ❌",
    }
