"""
Titanic ML — Utility Functions
===============================
Logging setup, random seed management, and general helpers.
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure project-wide logging.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("titanic_ml")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Attempt to set torch seed if available (for future DL extensions)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "titanic_ml") -> logging.Logger:
    """Get a child logger under the project namespace.

    Args:
        name: Logger name (will be prefixed with 'titanic_ml.').

    Returns:
        Logger instance.
    """
    if name == "titanic_ml":
        return logging.getLogger(name)
    return logging.getLogger(f"titanic_ml.{name}")


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
