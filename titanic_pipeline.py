"""
Titanic Survival Prediction — Extreme Gradient Generalization Pipeline
====================================================================
Utilizes advanced OOF (Out-Of-Fold) Threshold Optimization
coupled with tightly regularized XGBoost for 0.80+ Leaderboard accuracy.

Usage:
  python titanic_pipeline.py
"""

import pandas as pd
import numpy as np
import warnings
import glob

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
RANDOM_STATE = 42

def get_submission_filename():
    files = glob.glob("submission_v*.csv")
    if not files: return "submission_v1.csv"
    versions = []
    for f in files:
        try:
            versions.append(int(f.split("_v")[1].split(".csv")[0]))
        except ValueError:
            pass
    next_v = max(versions) + 1 if versions else 1
    return f"submission_v{next_v}.csv"

SUBMISSION = get_submission_filename()

def load_data():
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)

def optimize_threshold(y_true, y_probs):
    """Finds the optimal probability threshold to maximize accuracy heavily based on OOF distribution."""
    best_thresh = 0.5
    best_acc = 0
    # Search from 0.3 to 0.7
    for thresh in np.arange(0.3, 0.71, 0.01):
        preds = (y_probs >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


class TitanicFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced Kaggle Transformer encapsulating leak-free structural mapping
    and macro-pattern feature generation.
    """
    def __init__(self):
        self.age_medians_ = {}
        self.global_age_median_ = None
        self.ticket_counts_ = {}
        self.ticket_surv_ = {}
        self.global_surv_ = 0.38
        
    def extract_titles(self, df):
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_map = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Rare", "Dona": "Rare", "Countess": "Rare", 
            "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare",
            "Capt": "Rare", "Col": "Rare", "Major": "Rare", "Dr": "Rare", "Rev": "Rare"
        }
        df["Title"] = df["Title"].replace(title_map).fillna("Mr")
        core_titles = ["Mr", "Miss", "Mrs", "Master"]
        df.loc[~df["Title"].isin(core_titles), "Title"] = "Rare"
        return df

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy = self.extract_titles(X_copy)
        
        self.age_medians_ = X_copy.groupby("Title")["Age"].median().to_dict()
        self.global_age_median_ = X_copy["Age"].median()
        
        # Safe isolation: ONLY learn counts and survival from the training fold
        self.ticket_counts_ = X_copy["Ticket"].value_counts().to_dict()
        self.global_surv_ = y.mean() if y is not None else 0.38
        
        if y is not None:
            X_copy["Survived"] = y
            ticket_stats = X_copy.groupby("Ticket")["Survived"].agg(["mean", "count"])
            # Only record survival bias for groups size > 1
            self.ticket_surv_ = ticket_stats[ticket_stats["count"] > 1]["mean"].to_dict()
            
        return self

    def transform(self, X):
        df = X.copy()
        df = self.extract_titles(df)
        
        # 1. Age Imputation
        for title, med in self.age_medians_.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        df["Age"] = df["Age"].fillna(self.global_age_median_)
        
        # 2. Fare Engineering
        df["Fare"] = df["Fare"].fillna(32.2) # fallback
        df["Fare_Log"] = np.log1p(df["Fare"])
        
        # 3. Family Architecture
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        
        # 4. Deep Structural Interactions
        df["Age_Pclass"] = df["Age"] * df["Pclass"]
        
        # High value features requested
        df["IsFemaleChild"] = ((df["Sex"] == "female") | (df["Age"] < 12)).astype(int)
        df["Pclass_Sex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
        
        # 5. Survival Rate Mappings (No Leakage when routed through KFold Pipeline)
        df["TicketGroupSize"] = df["Ticket"].map(self.ticket_counts_).fillna(1)
        df["FamilySurvival"] = df["Ticket"].map(self.ticket_surv_).fillna(self.global_surv_)
        
        # Purge raw cardinalities explicitly
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId", "Fare"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        df["Pclass"] = df["Pclass"].astype(str)
        return df

def get_preprocessor():
    numeric_features = [
        "Age", "Fare_Log", "SibSp", "Parch", "FamilySize", 
        "FarePerPerson", "Age_Pclass", "IsFemaleChild", 
        "TicketGroupSize", "FamilySurvival"
    ]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "Pclass_Sex"]

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_transformer = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    
    return Pipeline([
        ("feature_engineering", TitanicFeatureTransformer()),
        ("col_transformer", col_transformer)
    ])

def main():
    print("Loading datasets...")
    train_df, test_df = load_data()
    
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]
    test_ids = pd.DataFrame({"PassengerId": test_df["PassengerId"]})
    
    # ─── XGBoost MODEL WITH HEAVY REGULARIZATION ───
    xgb_pipe = Pipeline([
        ('preprocessor', get_preprocessor()),
        ('classifier', XGBClassifier(
            n_estimators=450,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.2,
            random_state=RANDOM_STATE,
            eval_metric='auc'
        ))
    ])
    
    print("=" * 65)
    print("  OOF EVALUATION & PROBABILITY TUNING (5-Fold Stratified)")
    print("=" * 65)
    
    # Generate Out-Of-Fold probabilities (Completely Leak-Free natively via Sklearn Pipeline mechanics)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(xgb_pipe, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Analyze OOF metrics
    oof_auc = roc_auc_score(y, oof_probs)
    print(f"OOF Validation AUC: {oof_auc:.4f}")
    
    # Sweep optimal threshold strictly against the OOF probabilities
    best_thresh, best_oof_acc = optimize_threshold(y, oof_probs)
    default_acc = accuracy_score(y, (oof_probs >= 0.5).astype(int))
    
    print(f"Default 0.5 Threshold Accuracy: {default_acc:.4f}")
    print(f"Optimal Threshold ({best_thresh:.2f}) Accuracy: {best_oof_acc:.4f}")
    print("=" * 65)
    
    # ─── FINAL TRAINING ───
    print("\nTraining primary XGBoost model on full 100% dataset...")
    xgb_pipe.fit(X, y)

    print(f"\nGenerating predictions projecting via optimal threshold: {best_thresh:.2f}...")
    test_probs = xgb_pipe.predict_proba(test_df)[:, 1]
    final_preds = (test_probs >= best_thresh).astype(int)
    
    sub = pd.DataFrame({"PassengerId": test_ids["PassengerId"], "Survived": final_preds})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")

if __name__ == "__main__":
    main()
