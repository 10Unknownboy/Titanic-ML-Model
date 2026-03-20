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
├── requirements.txt           # Python dependencies
├── submission.csv             # Generated predictions (gitignored)
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
- Train and evaluate 3 models via 5-fold stratified CV
- Select the best model and generate `submission.csv`

---

## 🧠 Pipeline Overview

| Stage | Details |
|---|---|
| **Preprocessing** | Impute Age (median by Title), Fare (median), Embarked (mode) |
| **Feature Engineering** | Title extraction, FamilySize, IsAlone, AgeBand, FareBand, Deck |
| **Encoding** | Sex → binary, Embarked → one-hot, Title & Deck → label encoded |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting |
| **Evaluation** | 5-fold Stratified Cross-Validation (accuracy) |
| **Output** | `submission.csv` — `[PassengerId, Survived]` |

## 📊 Features Used

| Feature | Source | Type |
|---|---|---|
| Pclass | Original | Ordinal |
| Sex | Encoded (male=0, female=1) | Binary |
| Age | Imputed → binned into AgeBand | Ordinal |
| Fare | Imputed → binned into FareBand | Ordinal |
| SibSp, Parch | Original | Numeric |
| FamilySize | SibSp + Parch + 1 | Numeric |
| IsAlone | 1 if FamilySize == 1 | Binary |
| Title | Extracted from Name | Categorical |
| Deck | First letter of Cabin | Categorical |
| Embarked | One-hot encoded (S, C, Q) | Categorical |

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **scikit-learn** — ML models & evaluation

## 📝 License

This project is licensed under the terms in [LICENSE](LICENSE).
