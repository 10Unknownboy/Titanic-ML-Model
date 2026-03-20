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

def optimize_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_acc = 0
    # Search from 0.35 to 0.65 step 0.005
    for thresh in np.arange(0.35, 0.655, 0.005):
        preds = (y_probs >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc

class TitanicFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_medians_ = {}
        self.global_age_median_ = None
        self.fare_median_ = None
        self.ticket_counts_ = {}
        
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
        self.fare_median_ = X_copy["Fare"].median()
        
        # Learn counts exactly from the training fold
        self.ticket_counts_ = X_copy["Ticket"].value_counts().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        df = self.extract_titles(df)
        
        # 1. Age Imputation
        for title, med in self.age_medians_.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        df["Age"] = df["Age"].fillna(self.global_age_median_)
        
        # 2. Fare Engineering
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Fare_Log"] = np.log1p(df["Fare"])
        
        # 3. Family Architecture
        family_size = df["SibSp"] + df["Parch"] + 1
        df["FarePerPerson"] = df["Fare"] / family_size
        
        # FamilySize BIN
        df["FamilySizeBin"] = pd.cut(family_size, bins=[0, 1, 4, 100], labels=["Alone", "Small", "Large"], right=True)
        
        # 4. Deep Structural Interactions
        df["Age_Pclass"] = df["Age"] * df["Pclass"]
        df["IsFemaleChild"] = ((df["Sex"] == "female") | (df["Age"] < 12)).astype(int)
        df["Pclass_Sex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
        
        # 5. Cabin Deck
        df["Deck"] = df["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "U")
        
        # 6. TicketGroupSize (frequency of the ticket map from training)
        df["TicketGroupSize"] = df["Ticket"].map(self.ticket_counts_).fillna(1)
        
        # Purge explicitly dropped cardinalities
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId", "Fare", "SibSp", "Parch"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        df["Pclass"] = df["Pclass"].astype(str)
        return df

def get_preprocessor():
    numeric_features = [
        "Age", "Fare_Log", "FarePerPerson", "Age_Pclass", "IsFemaleChild", "TicketGroupSize"
    ]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "FamilySizeBin", "Pclass_Sex", "Deck"]

    # StandardScaler added to numeric features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
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
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            eval_metric='auc'
        ))
    ])
    
    print("=" * 65)
    print("  OOF EVALUATION & PROBABILITY TUNING (5-Fold Stratified)")
    print("=" * 65)
    
    # Generate Out-Of-Fold probabilities (Leak-Free)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(xgb_pipe, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Analyze OOF metrics
    oof_auc = roc_auc_score(y, oof_probs)
    print(f"OOF Validation AUC: {oof_auc:.4f}")
    
    # Sweep optimal threshold strictly against OOF distribution
    best_thresh, best_oof_acc = optimize_threshold(y, oof_probs)
    default_acc = accuracy_score(y, (oof_probs >= 0.5).astype(int))
    
    print(f"Default 0.5 Threshold Accuracy: {default_acc:.4f}")
    print(f"Optimal Threshold ({best_thresh:.3f}) Accuracy: {best_oof_acc:.4f}")
    print("=" * 65)
    
    # ─── FINAL TRAINING ───
    print("\nTraining primary XGBoost model on full 100% dataset...")
    xgb_pipe.fit(X, y)

    print(f"\nGenerating predictions projecting via optimal threshold: {best_thresh:.3f}...")
    test_probs = xgb_pipe.predict_proba(test_df)[:, 1]
    final_preds = (test_probs >= best_thresh).astype(int)
    
    sub = pd.DataFrame({"PassengerId": test_ids["PassengerId"], "Survived": final_preds})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")

if __name__ == "__main__":
    main()
