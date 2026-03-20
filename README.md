# 🚢 Titanic — Machine Learning from Disaster

Binary classification model predicting passenger survival on the Titanic.  
Built for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

---

## 📁 Project Structure

```
Titanic-ML-Model/
├── data/
│   ├── train.csv              # 891 labelled passengers
│   ├── test.csv               # 418 unlabelled passengers
│   └── gender_submission.csv  # Sample submission format
├── titanic_pipeline.py        # End-to-end ML pipeline
├── compare_submissions.py     # Script to analyze differences
├── requirements.txt           # Python dependencies
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### 1. Clone & enter the repo

```bash
git clone https://github.com/10Unknownboy/Titanic-ML-Model.git
cd Titanic-ML-Model
```

### 2. Create a virtual environment

```bash
python -m venv .env
```

### 3. Activate the environment

**Windows (PowerShell):**
```powershell
.env\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.env\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .env/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the pipeline

```bash
python titanic_pipeline.py
```

This will:
- Preprocess train & test data
- Train and evaluate 4 models via **10-fold stratified CV**
- Generate `submission_vX.csv` with automated versioning

---

## 🔍 Data Understanding & Insights

From our deep analysis of the competition dataset:

### 📈 Survival Benchmarks
- **Overall Survival Rate:** 38.4%
- **The Gender Split:** Females (74.2%) vs Males (18.9%).
- **Pclass Impact:** Pclass 1 (63%) vs Pclass 3 (24%).

### 💡 Key Survival Signals
| Segment | Survival Rate | Analysis |
|---|---|---|
| **HasCabin=1** | **66.7%** | Massive wealth/status proxy. Passengers with a known Cabin lived 2x as often. |
| **IsBoy** | **57.5%** | Young males (Title: Master) survived at much higher rates than adult males. |
| **FamilySize 2-4** | **55-72%** | The "Sweet Spot". Mid-sized families prioritized survival over solo travelers. |
| **3rd Class Female**| **50.0%** | The "Coin Flip". The hardest segment for models due to high variance. |

---

## 🧠 Pipeline Overview

| Stage | Details |
|---|---|
| **Preprocessing** | Impute Age (median by Title), Fare (median), Embarked (mode) |
| **Feature Engineering**| Title, FamilySizeBin, IsBoy, HasCabin, Age_Pclass, FarePerPerson |
| **Encoding** | Sex → binary, Embarked → one-hot, Title & FamilySizeBin → one-hot |
| **Models** | Voting Ensemble: LogisticRegression, RandomForest, GradientBoosting, SVC |
| **Evaluation** | **10-fold Stratified Cross-Validation** (Stability Check) |
| **Output** | `submission_vX.csv` — `[PassengerId, Survived]` |

## 📊 Features Used

| Feature | Source | Type |
|---|---|---|
| Pclass | Original | Ordinal |
| Sex | Encoded (male=0, female=1) | Binary |
| Age | Imputed by Title median | Numeric |
| Fare_Log | log1p transform | Numeric |
| IsAlone | Binary: 1 if FS=1 | Binary |
| Title | Extracted from Name | Categorical |
| FamilySizeBin | Solo, Small, Large | Categorical |
| HasCabin | Binary: 1 if Cabin known | **Binary (Big Signal)** |
| IsBoy | Master Title or Age < 12 | **Binary (Big Signal)** |
| Age_Pclass | Interaction term | Numeric |
| FarePerPerson | Total Fare / FamilySize | Numeric |

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **scikit-learn** — ML models & evaluation

## 📈 Model Evolution & Score Tracking

```text
START
  │
  ▼
Baseline Pipeline (Basic features + models)
  │
  └── Score: ~0.76
        │
        ▼
Add Heavy Feature Engineering
(AgeBand, FareBand, interactions, many features)
        │
        └── Score: ↓ 0.75
              (Overfitting + noise introduced)
              │
              ▼
Add More Complex Features
(Ticket, Surname, Deck, multiple interactions)
              │
              └── Score: ↓ 0.753 → ↓ 0.748
                    (High cardinality noise)
                    │
                    ▼
Switch to Simpler Features + Clean Pipeline
(Removed noise, reduced features)
                    │
                    └── Score: ↑ 0.758
                          (Better generalization)
                          │
                          ▼
Introduce XGBoost Model
(Regularized boosting, no CV tuning yet)
                          │
                          └── Score: ↓ 0.748
                                (Underfitting / poor tuning)
                                │
                                ▼
Add Threshold Optimization (Single split)
                                │
                                └── Score: ↑ 0.753
                                      (Better decision boundary)
                                      │
                                      ▼
Switch to OOF Cross-Validation + Threshold
(Proper validation strategy)
                                      │
                                      └── Score: ~0.751
                                            (Stable but plateau)
                                            │
                                            ▼
Add Advanced Features
(FamilySurvival, Ticket survival mapping)
                                            │
                                            └── Score: ~0.751
                                                  (Noise > signal)
                                                  │
                                                  ▼
Revert to Generalizable Baseline Ensemble
(Drop XGBoost + OOF; Switch to Soft VotingClassifier)
(LogisticRegression weight=1 + GradientBoosting weight=2)
(Core features only: Title, Age, Fare_Log, IsAlone,)
(FamilySizeBin, Age_Pclass, Pclass, Sex, Embarked)
(Strict 80/20 stratified holdout validation)
                                                  │
                                                  └── Score: ↑ 0.77990
                                                        (Best score so far)
                                                        (Less overfit, better LB generalization)
                                                        │
                                                        ▼
Add Surgical Features & Model Diversity
(Add HasCabin, FarePerPerson, SVC to Ensemble)
(Switch to 10-Fold Stratified CV for stability)
                                                  │
                                                  └── Score: TBD (Target 0.80+)
                                                        (Improved class separation)
                                                        (Reduced local vs LB variance)
                                                        │
                                                        ▼
                                                  CURRENT STATE
                                                  Target: 0.80+
```

## 📝 License

This project is licensed under the terms in [LICENSE](LICENSE).
