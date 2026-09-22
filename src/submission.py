"""
Titanic ML — Submission Generation
====================================
Create properly formatted Kaggle submission files.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import get_logger

logger = get_logger("submission")


def generate_submission(
    predictions: np.ndarray,
    test_ids: pd.Series,
    config: Config | None = None,
    filename: str | None = None,
) -> Path:
    """Generate a Kaggle-formatted submission CSV.

    Format: PassengerId, Survived (0 or 1)
    Expected: 418 rows.

    Args:
        predictions: Binary predictions array.
        test_ids: PassengerId series.
        config: Project configuration.
        filename: Custom filename. Auto-versioned if None.

    Returns:
        Path to the generated submission file.

    Raises:
        ValueError: If predictions don't match expected format.
    """
    config = config or Config()
    output_dir = config.output_dir

    # Validate
    if len(predictions) != len(test_ids):
        raise ValueError(
            f"Prediction count ({len(predictions)}) doesn't match "
            f"test ID count ({len(test_ids)})."
        )

    expected_rows = 418
    if len(predictions) != expected_rows:
        logger.warning(
            "Expected %d predictions but got %d. "
            "Submission may be invalid for Kaggle.",
            expected_rows, len(predictions),
        )

    # Ensure binary predictions
    predictions = predictions.astype(int)
    unique_vals = set(predictions)
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Predictions must be 0 or 1, got: {unique_vals}")

    # Auto-version filename
    if filename is None:
        filename = _get_versioned_filename(output_dir)

    filepath = output_dir / filename

    # Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": test_ids.astype(int),
        "Survived": predictions,
    })

    submission.to_csv(filepath, index=False)

    # Summary
    survived_count = predictions.sum()
    survived_pct = survived_count / len(predictions) * 100

    logger.info("=" * 50)
    logger.info("  Submission saved: %s", filepath)
    logger.info("  Rows: %d | Survived: %d (%.1f%%)", len(predictions), survived_count, survived_pct)
    logger.info("=" * 50)

    return filepath


def _get_versioned_filename(output_dir: Path) -> str:
    """Generate auto-versioned submission filename.

    Scans existing submission files and increments version number.

    Args:
        output_dir: Directory containing submission files.

    Returns:
        Filename string (e.g., 'submission_v3.csv').
    """
    existing = list(output_dir.glob("submission_v*.csv"))

    if not existing:
        return "submission_v1.csv"

    versions = []
    for f in existing:
        try:
            v = int(f.stem.split("_v")[1])
            versions.append(v)
        except (ValueError, IndexError):
            pass

    next_v = max(versions) + 1 if versions else 1
    return f"submission_v{next_v}.csv"


def validate_submission(filepath: str | Path) -> dict:
    """Validate a submission file against Kaggle requirements.

    Args:
        filepath: Path to submission CSV.

    Returns:
        Dict with validation results.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return {"valid": False, "error": f"File not found: {filepath}"}

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {"valid": False, "error": f"Failed to read CSV: {e}"}

    errors = []

    # Check columns
    expected_cols = ["PassengerId", "Survived"]
    if list(df.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {list(df.columns)}")

    # Check row count
    if len(df) != 418:
        errors.append(f"Expected 418 rows, got {len(df)}")

    # Check Survived values
    if "Survived" in df.columns:
        unique = set(df["Survived"].unique())
        if not unique.issubset({0, 1}):
            errors.append(f"Survived must be 0 or 1, got {unique}")

    # Check PassengerId range
    if "PassengerId" in df.columns:
        if df["PassengerId"].min() != 892 or df["PassengerId"].max() != 1309:
            errors.append(
                f"PassengerId range should be 892-1309, "
                f"got {df['PassengerId'].min()}-{df['PassengerId'].max()}"
            )

    if errors:
        return {"valid": False, "errors": errors}

    return {
        "valid": True,
        "rows": len(df),
        "survived_count": int(df["Survived"].sum()),
        "survived_pct": float(df["Survived"].mean() * 100),
    }
