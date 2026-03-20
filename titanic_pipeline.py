"""
Titanic Survival Prediction Pipeline
=====================================
End-to-end ML pipeline for the Kaggle Titanic competition.
Trains Logistic Regression, Random Forest, and Gradient Boosting,
evaluates via 5-fold stratified CV, and generates submission.csv.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "submission.csv"


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Title Extraction
# ═══════════════════════════════════════════════════════════════════════
def extract_title(df):
    """Extract title from Name and map rare titles to common groups."""
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Group rare titles
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
        "Col": "Rare", "Don": "Rare", "Dr": "Rare",
        "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
        "Jonkheer": "Rare", "Dona": "Rare",
    }
    df["Title"] = df["Title"].replace(title_map)
    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Preprocessing & Feature Engineering
# ═══════════════════════════════════════════════════════════════════════
def preprocess(df, fit_params=None):
    """
    Preprocess a DataFrame (train or test).

    Parameters
    ----------
    df : pd.DataFrame
    fit_params : dict or None
        If None (training), imputation values and encoders are computed
        and returned.  If provided (test), reuse the fitted values.

    Returns
    -------
    df_processed, fit_params
    """
    is_train = fit_params is None
    if is_train:
        fit_params = {}

    df = df.copy()

    # --- Title ---
    df = extract_title(df)

    # --- Age: impute by median per Title ---
    if is_train:
        fit_params["age_medians"] = df.groupby("Title")["Age"].median()
    for title, med in fit_params["age_medians"].items():
        df.loc[(df["Age"].isnull()) & (df["Title"] == title), "Age"] = med
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # --- Fare ---
    if is_train:
        fit_params["fare_median"] = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(fit_params["fare_median"])

    # --- Embarked ---
    if is_train:
        fit_params["embarked_mode"] = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(fit_params["embarked_mode"])

    # --- Sex: binary ---
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)

    # --- Deck from Cabin ---
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "U")

    # --- Family features ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- Age bands (fixed edges) ---
    df["AgeBand"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 60, 120],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # --- Fare bands (use training quantile edges for both) ---
    if is_train:
        _, bin_edges = pd.qcut(df["Fare"], q=4, retbins=True, duplicates="drop")
        bin_edges[0] = -np.inf          # extend left
        bin_edges[-1] = np.inf          # extend right
        fit_params["fare_bins"] = bin_edges
    df["FareBand"] = pd.cut(
        df["Fare"], bins=fit_params["fare_bins"],
        labels=range(len(fit_params["fare_bins"]) - 1),
        include_lowest=True,
    ).astype(int)

    # --- Encode categoricals (fit on train, reuse on test) ---
    if is_train:
        le_title = LabelEncoder()
        df["Title"] = le_title.fit_transform(df["Title"])
        fit_params["le_title"] = le_title

        le_deck = LabelEncoder()
        df["Deck"] = le_deck.fit_transform(df["Deck"])
        fit_params["le_deck"] = le_deck
    else:
        le_title = fit_params["le_title"]
        # Handle unseen labels gracefully
        known_titles = set(le_title.classes_)
        df["Title"] = df["Title"].apply(
            lambda t: t if t in known_titles else "Rare"
        )
        df["Title"] = le_title.transform(df["Title"])

        le_deck = fit_params["le_deck"]
        known_decks = set(le_deck.classes_)
        df["Deck"] = df["Deck"].apply(
            lambda d: d if d in known_decks else "U"
        )
        df["Deck"] = le_deck.transform(df["Deck"])

    # One-hot encode Embarked
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Emb", drop_first=True)

    # --- Drop unused columns ---
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    if "Survived" in df.columns:
        drop_cols.append("Survived")
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df, fit_params


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Model Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════
def evaluate_models(X, y):
    """Train & evaluate models with 5-fold stratified CV."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_split=4,
            min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_split=4, min_samples_leaf=2, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("=" * 55)
    print("  MODEL EVALUATION  (5-Fold Stratified CV)")
    print("=" * 55)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[name] = (model, mean_acc, std_acc)
        print(f"  {name:<25s}  {mean_acc:.4f} ± {std_acc:.4f}")

    print("=" * 55)
    return results


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — Main Pipeline
# ═══════════════════════════════════════════════════════════════════════
def main():
    # 1. Load
    train_df, test_df = load_data()
    y = train_df["Survived"]
    test_ids = test_df["PassengerId"]

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test  shape: {test_df.shape}\n")

    # 2. Preprocess (fit on train, transform test with same params)
    X_train, fit_params = preprocess(train_df)
    X_test, _ = preprocess(test_df, fit_params=fit_params)

    # Align columns (one-hot may differ)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}\n")

    # 3. Evaluate
    results = evaluate_models(X_train, y)

    # 4. Select best model
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_std = results[best_name]
    print(f"\n★ Best model: {best_name} — {best_acc:.4f} ± {best_std:.4f}")

    # 5. Retrain on full training set
    best_model.fit(X_train, y)

    # 6. Predict on test
    predictions = best_model.predict(X_test)

    # 7. Save submission
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": predictions.astype(int),
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n✓ Submission saved → {SUBMISSION_PATH}  ({len(submission)} rows)")

    # Quick sanity check
    assert len(submission) == 418, f"Expected 418 rows, got {len(submission)}"
    assert set(submission["Survived"].unique()).issubset({0, 1})
    print("✓ Sanity check passed (418 rows, values ∈ {0, 1})")


if __name__ == "__main__":
    main()
