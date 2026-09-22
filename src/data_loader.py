"""
Titanic ML — Data Loading
==========================
Environment-aware data loading with validation.
"""

import pandas as pd

from src.config import Config
from src.utils import get_logger

logger = get_logger("data_loader")


def load_data(config: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test datasets.

    Args:
        config: Project configuration. Uses singleton if None.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        FileNotFoundError: If data files are missing.
        ValueError: If expected columns are missing.
    """
    if config is None:
        config = Config()

    train_path = config.train_path
    test_path = config.test_path

    logger.info("Environment: %s", config.environment)
    logger.info("Loading training data from: %s", train_path)
    logger.info("Loading test data from: %s", test_path)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Ensure data files are in '{config.data_dir}'."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            f"Ensure data files are in '{config.data_dir}'."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validate expected columns
    _validate_columns(train_df, is_train=True)
    _validate_columns(test_df, is_train=False)

    logger.info(
        "Loaded: train=%d rows × %d cols, test=%d rows × %d cols",
        train_df.shape[0], train_df.shape[1],
        test_df.shape[0], test_df.shape[1],
    )
    logger.info("Train missing values:\n%s", train_df.isnull().sum()[train_df.isnull().sum() > 0].to_string())

    return train_df, test_df


def _validate_columns(df: pd.DataFrame, is_train: bool) -> None:
    """Validate that required columns exist.

    Args:
        df: DataFrame to validate.
        is_train: Whether this is the training set (expects 'Survived' column).

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"}
    if is_train:
        required.add("Survived")

    missing = required - set(df.columns)
    if missing:
        dataset_name = "training" if is_train else "test"
        raise ValueError(f"Missing columns in {dataset_name} data: {missing}")
