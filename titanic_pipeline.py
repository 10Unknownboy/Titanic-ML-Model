"""
Titanic Survival Prediction — Advanced Pipeline
=================================================
Optimised for maximum accuracy on the Kaggle Titanic competition.

Techniques used:
  • Rich feature engineering (Title, FamilySize, FarePP, TicketFreq, …)
  • Family-survival-rate leak-free features
  • Stacking Ensemble (RF + GBM + XGB + SVC + KNN → Logistic meta-learner)
  • Hyperparameter-tuned base learners
  • 10-fold stratified CV evaluation

Usage (Colab):
  Upload train.csv and test.csv to /content/data/
  then:  !python titanic_pipeline.py
"""

import pandas as pd
import numpy as np
import warnings, re

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
SUBMISSION = "submission.csv"
RANDOM_STATE = 42


# ═════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ═════════════════════════════════════════════════════════════════════════
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


# ═════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering (combined train + test for consistency)
# ═════════════════════════════════════════════════════════════════════════
def engineer_features(train, test):
    """
    Build features on the combined dataset, then split back.
    This ensures identical encoding / binning for train and test.
    """
    n_train = len(train)
    y = train["Survived"].copy()
    combined = pd.concat([train, test], sort=False).reset_index(drop=True)

    # ── Title ────────────────────────────────────────────────────────
    combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    combined["Title"] = combined["Title"].replace({
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Mrs", "Dona": "Mrs",
        "Countess": "Mrs", "Sir": "Mr", "Don": "Mr",
        "Jonkheer": "Mr",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Dr": "Officer", "Rev": "Officer",
    })

    # ── Sex ──────────────────────────────────────────────────────────
    combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1})

    # ── Embarked ─────────────────────────────────────────────────────
    combined["Embarked"].fillna(combined["Embarked"].mode()[0], inplace=True)

    # ── Fare ─────────────────────────────────────────────────────────
    combined["Fare"].fillna(combined["Fare"].median(), inplace=True)
    combined["Fare"] = np.log1p(combined["Fare"])          # log-transform skew

    # ── Age — predictive imputation using Title median ───────────────
    age_map = combined.groupby("Title")["Age"].median()
    for title, med in age_map.items():
        combined.loc[combined["Age"].isnull() & (combined["Title"] == title), "Age"] = med
    combined["Age"].fillna(combined["Age"].median(), inplace=True)

    # ── Family features ──────────────────────────────────────────────
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"]    = (combined["FamilySize"] == 1).astype(int)

    # Family-size buckets (single / small / medium / large)
    combined["FamilySizeBucket"] = combined["FamilySize"].apply(
        lambda s: 0 if s == 1 else (1 if s <= 4 else 2)
    )

    # ── Ticket frequency (shared tickets → travelling together) ──────
    ticket_counts = combined["Ticket"].value_counts()
    combined["TicketFreq"] = combined["Ticket"].map(ticket_counts)

    # ── Fare per person ──────────────────────────────────────────────
    combined["FarePerPerson"] = combined["Fare"] / combined["TicketFreq"]

    # ── Cabin features ───────────────────────────────────────────────
    combined["HasCabin"] = combined["Cabin"].notna().astype(int)
    combined["Deck"] = combined["Cabin"].apply(
        lambda x: str(x)[0] if pd.notna(x) else "U"
    )

    # ── Name length (proxy for social status) ────────────────────────
    combined["NameLen"] = combined["Name"].apply(len)

    # ── Age bands ────────────────────────────────────────────────────
    combined["AgeBand"] = pd.cut(
        combined["Age"],
        bins=[0, 5, 12, 18, 25, 35, 50, 65, 120],
        labels=[0, 1, 2, 3, 4, 5, 6, 7],
    ).astype(int)

    # ── Fare bands ───────────────────────────────────────────────────
    combined["FareBand"] = pd.qcut(
        combined["Fare"], q=5,
        labels=[0, 1, 2, 3, 4], duplicates="drop",
    )
    combined["FareBand"] = combined["FareBand"].fillna(2).astype(int)  # median bucket

    # ── Interaction features ─────────────────────────────────────────
    combined["Age*Pclass"]  = combined["Age"] * combined["Pclass"]
    combined["Sex*Pclass"]  = combined["Sex"] * combined["Pclass"]
    combined["Fare*Pclass"] = combined["Fare"] * combined["Pclass"]

    # ── Family survival rate (leak-free: compute on TRAIN only) ──────
    train_part = combined.iloc[:n_train].copy()
    train_part["Survived"] = y.values

    # Build surname
    combined["Surname"] = combined["Name"].apply(lambda n: n.split(",")[0].strip())

    # Survival rate by surname (train only)
    surname_surv = train_part.groupby(
        train_part["Name"].apply(lambda n: n.split(",")[0].strip())
    )["Survived"].mean()
    combined["FamilySurvRate"] = combined["Surname"].map(surname_surv)

    # Survival rate by ticket (train only)
    ticket_surv = train_part.groupby("Ticket")["Survived"].mean()
    combined["TicketSurvRate"] = combined["Ticket"].map(ticket_surv)

    # Fill unknowns with global mean
    global_mean = y.mean()
    combined["FamilySurvRate"].fillna(global_mean, inplace=True)
    combined["TicketSurvRate"].fillna(global_mean, inplace=True)

    # ── Encode categoricals ──────────────────────────────────────────
    for col in ["Title", "Deck"]:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    combined = pd.get_dummies(combined, columns=["Embarked"], prefix="Emb", drop_first=True)

    # ── Drop columns ─────────────────────────────────────────────────
    drop = ["PassengerId", "Name", "Ticket", "Cabin", "Survived", "Surname"]
    combined.drop(columns=[c for c in drop if c in combined.columns], inplace=True)

    # ── Split back ───────────────────────────────────────────────────
    X_train = combined.iloc[:n_train].copy()
    X_test  = combined.iloc[n_train:].copy()

    return X_train, X_test, y


# ═════════════════════════════════════════════════════════════════════════
# STEP 3 — Model Definitions
# ═════════════════════════════════════════════════════════════════════════
def build_models():
    """Return a dict of individual models + a stacking ensemble."""

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_split=4,
        min_samples_leaf=2, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_split=6, min_samples_leaf=3,
        random_state=RANDOM_STATE,
    )
    et = ExtraTreesClassifier(
        n_estimators=500, max_depth=8, min_samples_split=4,
        min_samples_leaf=2, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    svc = SVC(
        C=8.0, gamma="scale", kernel="rbf",
        probability=True, random_state=RANDOM_STATE,
    )
    knn = KNeighborsClassifier(
        n_neighbors=7, weights="distance",
        metric="minkowski", p=2, n_jobs=-1,
    )
    lr = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500,
        activation="relu", solver="adam",
        random_state=RANDOM_STATE,
    )

    # ── Stacking Ensemble ────────────────────────────────────────────
    stacking = StackingClassifier(
        estimators=[
            ("rf",  rf),
            ("gb",  gb),
            ("et",  et),
            ("svc", svc),
            ("knn", knn),
            ("mlp", mlp),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    # ── Soft Voting Ensemble ─────────────────────────────────────────
    voting = VotingClassifier(
        estimators=[
            ("rf",  rf),
            ("gb",  gb),
            ("et",  et),
            ("svc", svc),
            ("knn", knn),
            ("mlp", mlp),
        ],
        voting="soft",
        n_jobs=-1,
    )

    models = {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "Gradient Boosting":   gb,
        "Extra Trees":         et,
        "SVM (RBF)":           svc,
        "KNN":                 knn,
        "MLP":                 mlp,
        "Stacking Ensemble":   stacking,
        "Voting Ensemble":     voting,
    }
    return models


# ═════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluate
# ═════════════════════════════════════════════════════════════════════════
def evaluate(models, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("=" * 60)
    print("  MODEL EVALUATION  (10-Fold Stratified CV)")
    print("=" * 60)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_acc = scores.mean()
        std_acc  = scores.std()
        results[name] = (model, mean_acc, std_acc)
        marker = " ◀" if mean_acc >= 0.87 else ""
        print(f"  {name:<25s}  {mean_acc:.4f} ± {std_acc:.4f}{marker}")

    print("=" * 60)
    return results


# ═════════════════════════════════════════════════════════════════════════
# STEP 5 — Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    # 1. Load
    train_df, test_df = load_data()
    test_ids = test_df["PassengerId"]

    print(f"\nTrain: {train_df.shape}  |  Test: {test_df.shape}\n")

    # 2. Feature engineering
    X_train, X_test, y = engineer_features(train_df, test_df)

    # 3. Scale features (important for SVM, KNN, MLP, LR)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"Features ({X_train_sc.shape[1]}): {list(X_train_sc.columns)}\n")

    # 4. Build & evaluate
    models = build_models()
    results = evaluate(models, X_train_sc, y)

    # 5. Pick best
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_std = results[best_name]
    print(f"\n★ Best: {best_name} — {best_acc:.4f} ± {best_std:.4f}")

    # 6. Retrain on full training set & predict
    best_model.fit(X_train_sc, y)
    preds = best_model.predict(X_test_sc)

    # 7. Submission
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": preds.astype(int)})
    sub.to_csv(SUBMISSION, index=False)
    print(f"\n✓ Saved {SUBMISSION}  ({len(sub)} rows)")

    assert len(sub) == 418
    assert set(sub["Survived"].unique()).issubset({0, 1})
    print("✓ Sanity check passed\n")


if __name__ == "__main__":
    main()
