"""
Titanic ML — Configuration Management
=======================================
Loads config.yaml with environment-aware path resolution.
Detects Kaggle, Google Colab, or local execution automatically.
"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Central configuration with environment-aware path resolution.

    Auto-detects:
      - **Kaggle**: ``/kaggle/input/titanic/`` data path
      - **Google Colab**: Cloned repo or mounted Drive paths
      - **Local**: Relative paths from project root
    """

    _instance: "Config | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Config":
        """Singleton — one Config instance across the whole project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str | None = None) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.environment = self._detect_environment()
        self.project_root = self._find_project_root()

        # Load YAML config
        if config_path is None:
            config_path = self.project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._raw = yaml.safe_load(f)
        else:
            # Fallback defaults when config not found (e.g., in Kaggle/Colab)
            self._raw = self._default_config()

        # Unpack top-level sections
        self._project = self._raw.get("project", {})
        self._paths = self._raw.get("paths", {})
        self._features = self._raw.get("features", {})
        self._training = self._raw.get("training", {})
        self._models = self._raw.get("models", {})
        self._ensemble = self._raw.get("ensemble", {})

        # Resolve paths for current environment
        self._resolve_paths()

    # --- Public properties ---

    @property
    def random_seed(self) -> int:
        return self._project.get("random_seed", 42)

    @property
    def log_level(self) -> str:
        return self._project.get("log_level", "INFO")

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def train_path(self) -> Path:
        return self._data_dir / self._paths.get("train_file", "train.csv")

    @property
    def test_path(self) -> Path:
        return self._data_dir / self._paths.get("test_file", "test.csv")

    @property
    def features(self) -> dict:
        return self._features

    @property
    def training(self) -> dict:
        return self._training

    @property
    def models_config(self) -> dict:
        return self._models

    @property
    def ensemble_config(self) -> dict:
        return self._ensemble

    @property
    def cv_folds(self) -> int:
        return self._training.get("cv_folds", 10)

    # --- Environment detection ---

    @staticmethod
    def _detect_environment() -> str:
        """Detect execution environment: 'kaggle', 'colab', or 'local'."""
        # Kaggle kernel
        if os.path.exists("/kaggle/input"):
            return "kaggle"

        # Google Colab
        try:
            import google.colab  # noqa: F401
            return "colab"
        except ImportError:
            pass

        return "local"

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        if self.environment == "kaggle":
            # In Kaggle, the working directory is /kaggle/working
            # Project code may be cloned there
            working = Path("/kaggle/working")
            if (working / "src").exists():
                return working
            return working

        if self.environment == "colab":
            # In Colab, check common clone locations
            for candidate in [
                Path("/content/Titanic-ML-Model"),
                Path("/content/drive/MyDrive/Titanic-ML-Model"),
                Path.cwd(),
            ]:
                if (candidate / "src").exists():
                    return candidate
            return Path.cwd()

        # Local: walk up from this file to find project root
        current = Path(__file__).resolve().parent.parent
        if (current / "config").exists():
            return current
        return Path.cwd()

    def _resolve_paths(self) -> None:
        """Resolve data/model/output paths for the current environment."""
        if self.environment == "kaggle":
            self._data_dir = Path("/kaggle/input/titanic")
            self._model_dir = Path("/kaggle/working/models")
            self._output_dir = Path("/kaggle/working/outputs")
        elif self.environment == "colab":
            self._data_dir = self.project_root / self._paths.get("data_dir", "data")
            self._model_dir = self.project_root / self._paths.get("model_dir", "models")
            self._output_dir = self.project_root / self._paths.get("output_dir", "outputs")
        else:
            self._data_dir = self.project_root / self._paths.get("data_dir", "data")
            self._model_dir = self.project_root / self._paths.get("model_dir", "models")
            self._output_dir = self.project_root / self._paths.get("output_dir", "outputs")

        # Ensure output directories exist
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_config() -> dict:
        """Fallback configuration when config.yaml is not found."""
        return {
            "project": {"name": "Titanic ML", "random_seed": 42, "log_level": "INFO"},
            "paths": {
                "data_dir": "data",
                "model_dir": "models",
                "output_dir": "outputs",
                "train_file": "train.csv",
                "test_file": "test.csv",
            },
            "features": {
                "title_map": {
                    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
                    "Lady": "Rare", "Dona": "Rare", "Countess": "Rare",
                    "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare",
                    "Capt": "Rare", "Col": "Rare", "Major": "Rare",
                    "Dr": "Rare", "Rev": "Rare",
                },
                "core_titles": ["Mr", "Miss", "Mrs", "Master"],
                "family_size_bins": [0, 1, 4, 100],
                "family_size_labels": ["Alone", "Small", "Large"],
                "age_bins": [0, 12, 20, 40, 60, 120],
                "age_labels": ["Child", "Teen", "Adult", "MiddleAge", "Senior"],
                "boy_age_threshold": 12,
                "default_survival_rate": 0.5,
                "numeric": [
                    "Age", "Fare_Log", "Age_Pclass", "FarePerPerson",
                    "HasCabin", "IsBoy", "IsMarriedWoman", "TicketFrequency",
                    "FamilySurvivalRate", "TicketSurvivalRate", "FamilySize",
                ],
                "categorical": [
                    "Pclass", "Sex", "Embarked", "Title",
                    "FamilySizeBin", "Deck", "AgeGroup",
                ],
            },
            "training": {"cv_folds": 10, "optimize": False, "optuna_trials": 50},
            "models": {},
            "ensemble": {"method": "stacking"},
        }

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def __repr__(self) -> str:
        return (
            f"Config(env={self.environment}, root={self.project_root}, "
            f"data={self._data_dir}, seed={self.random_seed})"
        )
