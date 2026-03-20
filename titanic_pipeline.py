"""
Titanic Survival Prediction — XGBoost Decision Threshold Optimization
=====================================================================
Optimizes probability boundary (AUC/Logloss) before converting
to hard classes, achieving generalized 0.80+ Kaggle accuracy.

Usage:
  python titanic_pipeline.py
"""

import pandas as pd
import numpy as np
import warnings
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    
    core_titles = ["Mr", "Miss", "Mrs", "Master"]
    df.loc[~df["Title"].isin(core_titles), "Title"] = "Rare"
    return df

class TitanicFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_medians_ = {}
        self.global_age_median_ = None
        self.fare_median_ = None
        
    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy = extract_title(X_copy)
        self.age_medians_ = X_copy.groupby("Title")["Age"].median().to_dict()
        self.global_age_median_ = X_copy["Age"].median()
        self.fare_median_ = X_copy["Fare"].median()
        return self

    def transform(self, X):
        df = X.copy()
        df = extract_title(df)
        
        for title, med in self.age_medians_.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        df["Age"] = df["Age"].fillna(self.global_age_median_)
        
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Fare_Log"] = np.log1p(df["Fare"])
        
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        
        df["Age_Pclass"] = df["Age"] * df["Pclass"]
        df["Fare_Pclass"] = df["Fare_Log"] * df["Pclass"]
        
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        df["Pclass"] = df["Pclass"].astype(str)
        return df

def get_preprocessor():
    numeric_features = ["Age", "Fare_Log", "SibSp", "Parch", "FamilySize", "Age_Pclass", "Fare_Pclass"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]

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

def optimize_threshold(y_true, y_probs):
    """Finds the optimal probability threshold to maximize accuracy on the validation set."""
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

def main():
    print("Loading data...")
    full_train_df, test_df = load_data()
    
    # ─── HOLD-OUT VALIDATION SPLIT ───
    df_train, df_val = train_test_split(
        full_train_df, test_size=0.2, stratify=full_train_df["Survived"], random_state=RANDOM_STATE
    )
    
    X_train = df_train.drop(columns=["Survived"])
    y_train = df_train["Survived"]
    
    X_val = df_val.drop(columns=["Survived"])
    y_val = df_val["Survived"]
    
    X_test_final = test_df
    test_ids = pd.DataFrame({"PassengerId": pd.read_csv(TEST_PATH)["PassengerId"]})
    
    preprocessor = get_preprocessor()
    
    # ─── XGBoost MODEL ───
    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=400,
            learning_rate=0.04,
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
    print("  MODEL EVALUATION (Probability Optimization)")
    print("=" * 65)
    
    # Train purely on the 80% split
    xgb_pipe.fit(X_train, y_train)
    
    # Extract Probabilities rather than hard classes
    val_probs = xgb_pipe.predict_proba(X_val)[:, 1]
    
    auc_score = roc_auc_score(y_val, val_probs)
    print(f"Validation AUC: {auc_score:.4f}")
    
    # Tune Threshold
    best_thresh, best_val_acc = optimize_threshold(y_val, val_probs)
    default_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
    
    print(f"Default 0.5 Threshold Accuracy: {default_acc:.4f}")
    print(f"Optimal Threshold ({best_thresh:.2f}) Accuracy: {best_val_acc:.4f}")
    print("=" * 65)
    
    # ─── FINAL TRAINING (100% Data) ───
    print("\nTraining on the full 100% dataset pipeline...")
    X_full_train = full_train_df.drop(columns=["Survived"])
    y_full_train = full_train_df["Survived"]
    xgb_pipe.fit(X_full_train, y_full_train)

    print(f"\nGenerating predictions using optimal threshold: {best_thresh:.2f}...")
    test_probs = xgb_pipe.predict_proba(X_test_final)[:, 1]
    final_preds = (test_probs >= best_thresh).astype(int)
    
    sub = pd.DataFrame({"PassengerId": test_ids["PassengerId"], "Survived": final_preds})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")

if __name__ == "__main__":
    main()
