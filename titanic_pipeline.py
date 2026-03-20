"""
Titanic Survival Prediction — Kaggle Generalization Pipeline
============================================================
Optimized to extract strong signal and maximize the true
Kaggle Leaderboard test score (~0.80 - 0.83).

Usage:
  python titanic_pipeline.py
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
SUBMISSION = "submission.csv"
RANDOM_STATE = 42


def load_data():
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def extract_title(df):
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Dona": "Rare", "Countess": "Rare", 
        "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare",
        "Capt": "Rare", "Col": "Rare", "Major": "Rare", "Dr": "Rare", "Rev": "Rare"
    }
    df["Title"] = df["Title"].replace(title_map)
    df["Title"] = df["Title"].fillna("Mr")
    
    # Restrict strictly to core stable titles
    core_titles = ["Mr", "Miss", "Mrs", "Master"]
    df.loc[~df["Title"].isin(core_titles), "Title"] = "Rare"
    return df


class TitanicFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A unified Custom Transformer implementing highly-stable Kaggle features.
    It fits on the training set (e.g., learning medians and bin boundaries)
    and strictly applies them to validation/test sets to guarantee zero leakage.
    """
    def __init__(self):
        self.age_medians_ = {}
        self.global_age_median_ = None
        self.fare_median_ = None
        self.fare_bins_ = None
        self.ticket_counts_ = {} # 🔥 IMPROVED: Ticket count frequency mapping
        
    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy = extract_title(X_copy)
        
        # 1. Learn Age Medians per Title
        self.age_medians_ = X_copy.groupby("Title")["Age"].median().to_dict()
        self.global_age_median_ = X_copy["Age"].median()
        
        # 2. Learn Fare Median
        self.fare_median_ = X_copy["Fare"].median()
        
        # 3. Learn Fare Quantile Bins (4 bins)
        _, bins = pd.qcut(X_copy["Fare"].fillna(self.fare_median_), q=4, retbins=True, duplicates="drop")
        bins[0] = -np.inf
        bins[-1] = np.inf
        self.fare_bins_ = bins
        
        # 🔥 IMPROVED: Learn Ticket Frequency from training
        self.ticket_counts_ = X_copy["Ticket"].value_counts().to_dict()
        
        return self

    def transform(self, X):
        df = X.copy()
        df = extract_title(df)
        
        # ── Handle Missing Values safely using fitted medians ──
        for title, med in self.age_medians_.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        df["Age"] = df["Age"].fillna(self.global_age_median_)
        
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Fare_Log"] = np.log1p(df["Fare"])
        
        # ── Feature: Family Size and Classification ──
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        
        # 🔥 IMPROVED: FamilyType refinement
        df["FamilyType"] = pd.cut(df["FamilySize"], bins=[0, 1, 4, 20], labels=["Solo", "Small", "Large"], right=True)
        
        # ── Feature: Powerful Interactions ──
        df["Age_Pclass"] = df["Age"] * df["Pclass"]
        df["Fare_Pclass"] = df["Fare_Log"] * df["Pclass"]
        
        df["IsChild"] = (df["Age"] < 12).astype(int)
        df["IsMother"] = ((df["Sex"] == "female") & (df["Age"] > 18) & (df["Parch"] > 0)).astype(int)
        
        # ── Feature: Static Bins ──
        # 🔥 IMPROVED: Removed excessive binning and Age/Sex interactions
        df["FareBand"] = pd.cut(df["Fare"], bins=self.fare_bins_, labels=False)
        
        # 🔥 IMPROVED: TicketGroupSize (frequency of the ticket)
        df["TicketGroupSize"] = df["Ticket"].map(self.ticket_counts_).fillna(1)
        
        # 🔥 IMPROVED: Cabin Deck
        df["Deck"] = df["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "U")
        
        # 🔥 IMPROVED: Surname extraction
        df["Surname"] = df["Name"].apply(lambda x: x.split(",")[0].strip())
        
        # ── Cleanup ──
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        # Convert categoricals to strings to play nicely with OneHotEncoder
        df["FareBand"] = df["FareBand"].astype(str)
        df["Pclass"] = df["Pclass"].astype(str)
        
        return df


def get_preprocessor():
    """ 
    Creates the final scikit-learn preprocessing pipeline. 
    It cascades custom feature engineering into strict standard preprocessing.
    """
    # 🔥 IMPROVED: Included new robust features
    numeric_features = ["Age", "Fare_Log", "SibSp", "Parch", "FamilySize", "Age_Pclass", "Fare_Pclass", "IsChild", "IsMother", "TicketGroupSize"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "FamilyType", "FareBand", "Deck", "Surname"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    
    # Combine custom engineered features with column transformer
    preprocessor = Pipeline([
        ("feature_engineering", TitanicFeatureTransformer()),
        ("col_transformer", col_transformer)
    ])
    
    return preprocessor


def build_models(preprocessor):
    """
    Controlled, carefully tuned models targeting Kaggle Test out-of-sample generalization.
    """
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE))
    ])
    
    svc_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', SVC(C=0.5, gamma='scale', probability=True, random_state=RANDOM_STATE))
    ])

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        # 🔥 IMPROVED: Deeper trees for RF to fix underfitting
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=3, random_state=RANDOM_STATE))
    ])
    
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        # 🔥 IMPROVED: More estimators and slightly higher learning rate for GB
        ('classifier', GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, min_samples_leaf=4, subsample=0.85, random_state=RANDOM_STATE))
    ])
    
    # 🔥 IMPROVED: Weighted Voting favoring Gradient Boosting
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_pipe),
            ('rf', rf_pipe),
            ('gb', gb_pipe),
            ('svc', svc_pipe)
        ],
        voting='soft',
        weights=[1, 2, 3, 1] # GradientBoosting has highest weight
    )
    
    return {
        "Logistic Regression": lr_pipe,
        "Random Forest": rf_pipe,
        "Gradient Boosting": gb_pipe,
        "SVC": svc_pipe,
        "Weighted Voting": voting_clf
    }


def main():
    print("Loading data...")
    full_train_df, test_df = load_data()
    
    # ─── HOLD-OUT VALIDATION SPLIT ───────────────────────────────────────
    # Simulates Kaggle Test environment strictly
    df_train, df_val = train_test_split(
        full_train_df, test_size=0.2, stratify=full_train_df["Survived"], random_state=RANDOM_STATE
    )
    
    X_train_cv = df_train.drop(columns=["Survived"])
    y_train_cv = df_train["Survived"]
    
    X_val = df_val.drop(columns=["Survived"])
    y_val = df_val["Survived"]
    
    X_test_final = test_df
    test_ids = pd.DataFrame({"PassengerId": pd.read_csv(TEST_PATH)["PassengerId"]})
    
    # Setup robust Pipeline
    preprocessor = get_preprocessor()
    models = build_models(preprocessor)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    
    print("=" * 80)
    print("  MODEL EVALUATION (3x5 Repeated Stratified CV & 20% Holdout)")
    print("=" * 80)
    
    best_name = None
    best_val_score = 0
    best_model = None
    
    for name, model in models.items():
        # CV strictly on training fold
        scores = cross_val_score(model, X_train_cv, y_train_cv, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_cv = scores.mean()
        std_cv = scores.std()
        
        # Test against unseen 20% local holdout
        model.fit(X_train_cv, y_train_cv)
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        
        overfit_flag = "⚠️ Guard" if (mean_cv - val_acc) > 0.03 else "✅ Flow"
        print(f"  {name:<25s} | CV: {mean_cv:.4f} (±{std_cv:.3f}) | Val: {val_acc:.4f} {overfit_flag}")
        
        # Final selection dictates Kaggle strength! Priority is Holdout.
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_name = name
            best_model = model
            
    print("=" * 80)
    print(f"\n★ Best Generalizing Model: {best_name} (Holdout Acc: {best_val_score:.4f})")
    
    # ─── FINAL TRAINING (100% Data) ──────────────────────────────────────
    print(f"\nTraining {best_name} on the full 100% training set...")
    X_full_train = full_train_df.drop(columns=["Survived"])
    y_full_train = full_train_df["Survived"]
    
    best_model.fit(X_full_train, y_full_train)

    # 🔥 IMPROVED: All plotting functions fully removed.

    print("\nGenerating predictions...")
    preds = best_model.predict(X_test_final)
    
    sub = pd.DataFrame({"PassengerId": test_ids["PassengerId"], "Survived": preds.astype(int)})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")


if __name__ == "__main__":
    main()
