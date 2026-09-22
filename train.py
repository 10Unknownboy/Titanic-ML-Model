#!/usr/bin/env python
"""
Titanic ML — Training CLI
===========================
Entry point for training the Titanic survival prediction model.

Usage::

    python train.py                    # Train with default settings
    python train.py --optimize         # Include Optuna hyperparameter search
    python train.py --model xgboost    # Train a specific model only
    python train.py --method voting    # Use voting instead of stacking
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config
from src.training import train_pipeline
from src.utils import set_seed, setup_logging


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train the Titanic survival prediction model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization (slower, potentially better).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Train a specific model only (e.g., xgboost, lightgbm, svc).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["stacking", "voting"],
        default=None,
        help="Ensemble method: 'stacking' (default) or 'voting'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml.",
    )

    args = parser.parse_args()

    # Initialize config
    config = Config(config_path=args.config)

    # Override ensemble method if specified
    if args.method:
        config._ensemble["method"] = args.method

    # Setup
    setup_logging(config.log_level)
    set_seed(config.random_seed)

    # Train
    result = train_pipeline(
        config=config,
        optimize=args.optimize,
        model_name=args.model,
    )

    print("\n[OK] Training complete!")
    if result.cv_scores:
        scores = [v for v in result.cv_scores.values() if isinstance(v, float)]
        if scores:
            print(f"   Best individual model OOF accuracy: {max(scores):.4f}")
            print(f"   Mean OOF accuracy across models:    {sum(scores)/len(scores):.4f}")
    print(f"   Training time: {result.training_time:.1f}s")
    print(f"   Submission saved to: {config.output_dir}")
    print()


if __name__ == "__main__":
    main()
