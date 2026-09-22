"""
Titanic ML — Preprocessing
============================
Sklearn ColumnTransformer pipelines for numeric and categorical features.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import Config
from src.utils import get_logger

logger = get_logger("preprocessing")


def get_preprocessor(config: Config | None = None) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for numeric + categorical features.

    Args:
        config: Project configuration.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    config = config or Config()
    feat = config.features

    numeric_features = feat.get("numeric", [])
    categorical_features = feat.get("categorical", [])

    logger.info("Numeric features (%d): %s", len(numeric_features), numeric_features)
    logger.info("Categorical features (%d): %s", len(categorical_features), categorical_features)

    # Numeric: impute median → robust scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    # Categorical: impute mode → one-hot encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A fitted ColumnTransformer.

    Returns:
        List of output feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn
        names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out())
            else:
                names.extend(columns)
        return names
