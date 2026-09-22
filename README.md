# 🚢 Titanic — Machine Learning from Disaster

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Titanic%20ML%20Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](LICENSE)
[![Ensemble](https://img.shields.io/badge/Ensemble-Voting%20(RF%20+%20GB)-FF6F00?style=for-the-badge)](#-model-architecture--voting-ensemble)
[![Cross-Validation](https://img.shields.io/badge/CV-10--Fold%20Stratified-success?style=for-the-badge)](#-model-architecture--voting-ensemble)
[![Dashboard](https://img.shields.io/badge/App-Streamlit%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#-streamlit-interactive-dashboard)

A production-grade, modular machine learning solution for the classic [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.

This repository implements an end-to-end classification pipeline centered around a core ML philosophy: **Simplicity and strong regularization prevent overfitting on small datasets.** It features a hyper-curated set of 7 powerful features, a highly stable **Voting Ensemble** (Random Forest + Gradient Boosting), **Bayesian hyperparameter optimization** via Optuna, **auto-environment detection** for Kaggle and Google Colab, and an interactive **Streamlit dashboard**.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture & Voting Ensemble](#-model-architecture--voting-ensemble)
- [CLI Usage](#-cli-usage)
- [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
- [Notebook Usage (Colab & Kaggle)](#-notebook-usage-colab--kaggle)
- [Configuration Management](#-configuration-management)
- [Model Evolution & Score Tracking](#-model-evolution--score-tracking)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🎯 Project Overview

Predicting survival on the RMS Titanic involves overcoming challenges common to tabular ML: subtle interactions, high-cardinality metadata, missing values across demographic subgroups, and — most importantly — **severe risk of test-set overfitting**.

After extensive experimentation with massive feature sets, complex target encoding, hardcoded WCG (Women-Child-Group) overrides, and 7-model stacking architectures, we discovered that the model was memorizing the tiny 891-row training set and failing to generalize. 

This final iteration embraces the ultimate Titanic ML lesson: **Less is More**.
- **Minimal, Robust Feature Set**: Stripped down from 17+ noisy columns to the 7 absolute strongest, least collinear predictors.
- **Stable Voting Ensemble**: Discarded the complex meta-learner and deep boosting models in favor of a heavily-regularized, weighted soft-voting ensemble of Random Forest and Gradient Boosting.
- **Clean Architecture**: Fully modularized Python package (`src/`) with strict typing, centralized YAML configuration, and unified logging.
- **Zero-Config Cross-Platform Support**: Automatically detects whether running locally, on **Google Colab**, or in a **Kaggle Kernel**.

---

## 📁 Project Structure

```text
Titanic-ML-Model/
├── config/
│   └── config.yaml              # Central configuration (hyperparameters, paths, features)
├── src/
│   ├── __init__.py
│   ├── config.py                # Singleton config loader with auto env detection
│   ├── data_loader.py           # Environment-aware data ingestion & column validation
│   ├── feature_engineering.py   # Core transforms (Title, Imputation, FamilySize)
│   ├── preprocessing.py         # Sklearn ColumnTransformer (imputation, scaling, OHE)
│   ├── models.py                # 7 base models, StackingEnsemble, VotingEnsemble
│   ├── training.py              # End-to-end training pipeline & Optuna optimization
│   ├── evaluation.py            # CV reporting, stability checks, classification metrics
│   ├── prediction.py            # Inference routines (test set batch & single passenger)
│   ├── submission.py            # Kaggle submission creation & automated validation
│   └── utils.py                 # Structured logging, random seeds, path helpers
├── data/
│   ├── train.csv                # 891 labelled passengers
│   ├── test.csv                 # 418 test passengers
│   └── gender_submission.csv    # Kaggle benchmark submission format
├── models/                      # Serialized model artifacts (titanic_model.joblib)
├── outputs/                     # Versioned submissions (submission_v1.csv, ...)
├── notebooks/
│   └── titanic_colab_kaggle.ipynb # Interactive notebook for Kaggle & Colab runs
├── app/
│   └── streamlit_app.py         # Interactive passenger survival simulator & EDA dashboard
├── train.py                     # CLI: model training entry point
├── predict.py                   # CLI: batch inference & submission validator
├── requirements.txt             # Python dependencies
├── .gitignore                   # Version control ignore rules
├── LICENSE                      # GNU General Public License v3.0
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/10Unknownboy/Titanic-ML-Model.git
cd Titanic-ML-Model
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate the virtual environment:
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Ensemble

```bash
python train.py
```
*Trains the ensemble using 10-fold Stratified CV, saves `models/titanic_model.joblib`, and outputs a validated submission to `outputs/`.*

### 5. Generate Predictions & Validate

```bash
# Generate submission using the saved model
python predict.py

# Validate any existing submission file
python predict.py --validate outputs/submission_v13.csv
```

---

## 🔬 Feature Engineering

Raw attributes on the Titanic passenger manifest often hide vital survival signals. However, engineering too many highly correlated features (e.g., `IsBoy`, `IsMarriedWoman`, `Title`, `Sex`) causes severe overfitting on small datasets.

Our final `TitanicFeatureEngineer` pipeline focuses strictly on **7 non-collinear, high-signal features**:

| Feature | Source / Transformation | Type | ML Rationale & Survival Signal |
|:---|:---|:---|:---|
| `Title` | Extracted from `Name` (`r" ([A-Za-z]+)\."`) | Categorical | Maps raw titles to `Mr`, `Miss`, `Mrs`, `Master`, or `Rare`. Captures social status and marital/gender standing. |
| `Age` | Imputed via Title-group median | Numeric | Age strongly correlates with title (e.g. Master median ~3.5 vs Mr ~30.0). Preserves group variance during imputation. |
| `Fare_Log` | `np.log1p(Fare)` | Numeric | Eliminates extreme skewness in ticket prices (£0.00 to £512.33) and stabilizes split boundaries. |
| `FamilySize` | `SibSp + Parch + 1` | Numeric | Total party size. Solo travelers and massive families suffered high mortality; moderate families survived. |
| `Pclass` | Passenger Ticket Class (1, 2, 3) | Categorical | Primary proxy for socioeconomic standing and physical cabin location on ship decks. |
| `Sex` | Gender (`male`, `female`) | Categorical | Single strongest individual split ("women and children first"). |
| `Embarked` | Port of departure (`C`, `Q`, `S`) | Categorical | Correlates with socio-economic status (Cherbourg had highest proportion of 1st class). |

### Preprocessing Pipeline

Engineered features pass through a scikit-learn `ColumnTransformer`:
- **Numeric Pipeline**: `SimpleImputer(strategy="median")` $\rightarrow$ `RobustScaler()` (robust against outliers in fares).
- **Categorical Pipeline**: `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

---

## 🏛️ Model Architecture & Voting Ensemble

After extensive testing, we concluded that complex Stacking ensembles (Meta-learners atop 7 base models) vastly overfit the Titanic dataset. 

This solution uses a **Highly-Regularized Soft Voting Ensemble** comprising the two most robust tree-based models.

### Base Models

1. **Random Forest** (`sklearn`): Bagging decorrelation. Heavily regularized with `max_depth=5`, `min_samples_leaf=2`, `min_samples_split=4`, and `n_estimators=300`.
2. **Gradient Boosting** (`sklearn`): Sequential trees with shrinkage. Constrained with `learning_rate=0.05`, `max_depth=3`, and `subsample=0.8`.

### Ensemble Configuration
- **Method**: Soft Voting (averages the predicted probabilities).
- **Weights**: Random Forest (Weight: 2), Gradient Boosting (Weight: 1).
- **Why?**: The Random Forest is more robust to the noise inherent in the Titanic training set, while Gradient Boosting provides fine-tuned margin corrections.

### Validation Performance

Our 10-fold Stratified CV evaluation on the training set yields highly stable, realistic out-of-fold accuracy scores:
- **Ensemble Mean OOF Accuracy**: ~`0.8395`

*(Note: While previous stacking iterations achieved `0.90+` local accuracy, they completely failed on the Kaggle public leaderboard due to overfitting. `0.8395` is a realistic and generalizable local score).*

---

## 💻 CLI Usage

The project includes two primary command-line interfaces: `train.py` and `predict.py`.

### 1. Training CLI (`train.py`)

```bash
# Default execution: 10-fold Voting Ensemble
python train.py

# Run with Optuna Bayesian hyperparameter optimization
python train.py --optimize

# Train a single specific model (e.g. xgboost, randomforest)
python train.py --model random_forest

# Run with a custom configuration file
python train.py --config config/custom_config.yaml
```

### 2. Prediction CLI (`predict.py`)

```bash
# Generate submission from default trained model (models/titanic_model.joblib)
python predict.py

# Validate an existing submission file format and statistics
python predict.py --validate outputs/submission_v13.csv
```

---

## 🖥️ Streamlit Interactive Dashboard

The repository includes a Streamlit web application providing interactive survival simulations, model diagnostic reports, and exploratory data analysis.

### Launching the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Dashboard Features

1. **🔮 Passenger Survival Simulator**:
   - Interactive UI sliders and dropdowns for passenger class, age, gender, ticket fare, family size, embarkation port, and cabin deck.
   - Computes real-time survival probability using the trained Ensemble.
2. **📊 Model Performance & Diagnostics**:
   - Training dataset inspection table and CV metrics.
3. **📈 Exploratory Data Analysis**:
   - Interactive visualizations of survival distribution by gender and passenger class.

---

## 📓 Notebook Usage (Colab & Kaggle)

The notebook at `notebooks/titanic_colab_kaggle.ipynb` is designed to run seamlessly across **Kaggle Kernels**, **Google Colab**, or your **Local machine** without changing code.

### Automated Environment Detection

The pipeline dynamically inspects runtime indicators:

```python
from src.config import Config
config = Config()
print(f"Detected runtime environment: {config.environment}")
```

- **Kaggle Kernel**:
  - Automatically identifies `/kaggle/input/titanic`.
  - Routes serialized models and outputs directly to `/kaggle/working/`.
- **Google Colab**:
  - Detects `google.colab` environment.
  - Automatically handles path resolution.

---

## ⚙️ Configuration Management

All hyperparameters, paths, and feature parameters are centralized in `config/config.yaml`. 

```yaml
# Example excerpt from config.yaml
features:
  numeric:
    - "Age"
    - "Fare_Log"
    - "FamilySize"

  categorical:
    - "Pclass"
    - "Sex"
    - "Embarked"
    - "Title"

ensemble:
  method: "voting"
  voting_weights: [0, 2, 1, 0, 0, 0, 0]  # Weights mapping to models
```

---

## 📈 Model Evolution & Score Tracking

The progression of model iterations, experimentation, and public leaderboard validation taught us the ultimate tabular ML lesson:

```text
START
  │
  ▼
1. Generalizable Soft-Voting Baseline
   - Core features: Age, Fare_Log, FamilySize, Pclass, Sex, Embarked
  │
  └── Kaggle Score: 0.77990  (Best submission, minimal overfit)
        │
        ▼
2. The Overfitting Trap (Adding Leakage)
   - Introduced leaky 'FamilySurvivalRate'
   - Stacking 7 models
  │
  └── Kaggle Score: 0.78947  (Artificial boost from CV leakage)
        │
        ▼
3. The Overfitting Trap (Too Many Features)
   - Fixed the leakage but kept 17 highly-correlated features (Deck, IsBoy, etc.)
   - Kept Stacking Meta-Learner
  │
  └── Kaggle Score: 0.76315  (Catastrophic generalization failure)
        │
        ▼
4. Current State: "Less is More" Pure ML Pipeline
   - Stripped down to 7 core, non-collinear features
   - Dropped Stacking for a simple, stable Random Forest + GB Voting Ensemble
   - Heavy tree regularization (max_depth=3 to 5)
  │
  └── Target: Stable generalization matching or exceeding 0.78+
```

### Key Takeaways from Iteration

1. **Beware the Cardinality and Collinearity Trap**: Adding features like `IsBoy` and `IsMarriedWoman` when `Sex` and `Title` already exist does not give tree models more information—it just gives them more ways to overfit a tiny 891-row dataset.
2. **Complex Ensembles Require Big Data**: A Stacking Ensemble with a Logistic Regression Meta-Learner analyzing 7 algorithms will perfectly memorize an 891-row training set (yielding local scores of 90%+), but will utterly fail on the Kaggle public leaderboard.
3. **Simplicity Wins**: On tiny datasets, heavily regularized Random Forests with a minimal, clean feature set usually beat XGBoost and Deep Stacking.

---

## 🛠️ Tech Stack

| Category | Technologies | Purpose |
|:---|:---|:---|
| **Language & Runtimes** | Python 3.10+, Google Colab, Kaggle Kernels | Core runtime environments |
| **Data Processing** | Pandas, NumPy | Data ingestion, aggregation, matrix manipulation |
| **Machine Learning** | Scikit-Learn | Pipelines, ColumnTransformer, RF, GB, Voting Ensemble |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization of tree parameters |
| **Web Dashboard** | Streamlit | Interactive prediction UI & performance visualization |
| **Data Visualization** | Matplotlib, Seaborn, Plotly | Exploratory charts, confusion matrices, CV metrics |
| **Serialization & Config** | Joblib, PyYAML | Model persistence and centralized configuration management |

---

## 📝 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for complete terms and conditions.
