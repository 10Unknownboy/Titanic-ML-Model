#!/usr/bin/env python
"""
Titanic ML — Prediction CLI
==============================
Generate predictions from a saved model.

Usage::

    python predict.py                                      # Use default saved model
    python predict.py --model-path models/titanic_model.joblib  # Specify model file
    python predict.py --validate outputs/submission_v1.csv      # Validate a submission
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config
from src.prediction import predict_test_set
from src.submission import generate_submission, validate_submission
from src.utils import setup_logging


def main() -> None:
    """Parse arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description="Generate predictions from a saved Titanic model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model file (default: models/titanic_model.joblib).",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate an existing submission file instead of generating predictions.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml.",
    )

    args = parser.parse_args()

    config = Config(config_path=args.config)
    setup_logging(config.log_level)

    if args.validate:
        # Validate mode
        print(f"\nValidating: {args.validate}")
        result = validate_submission(args.validate)
        if result.get("valid"):
            print(f"  [OK] Valid submission!")
            print(f"  Rows: {result['rows']}")
            print(f"  Survived: {result['survived_count']} ({result['survived_pct']:.1f}%)")
        else:
            print(f"  [FAIL] Invalid submission!")
            for err in result.get("errors", [result.get("error", "Unknown error")]):
                print(f"     - {err}")
        return

    # Prediction mode
    print("\n[*] Generating predictions...")
    predictions, test_ids, probabilities = predict_test_set(
        model_path=args.model_path,
        config=config,
    )

    submission_path = generate_submission(predictions, test_ids, config)

    # Validate the generated submission
    validation = validate_submission(submission_path)
    if validation.get("valid"):
        print("[OK] Submission validated successfully!")
    else:
        print("[!] Submission validation warnings:")
        for err in validation.get("errors", []):
            print(f"   - {err}")

    print()


if __name__ == "__main__":
    main()
