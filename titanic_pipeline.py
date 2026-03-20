"""
Titanic Survival Pipeline — Kaggle 0.80+ Generalization Architecture
====================================================================
Eliminates local holdout variance via 10-Fold CV. 
Leverages pure, highly-diversified Soft Voting (Linear + Non-Linear)
over strictly controlled fundamental features.
"""

import pandas as pd
import numpy as np
import warnings
import glob

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

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

class GeneralizationFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_medians_ = {}
        self.global_age_median_ = None
        self.fare_median_ = None

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        # Isolate Title
        X_copy["Title"] = X_copy["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Dona": "Rare", "Countess": "Rare", "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare", "Capt": "Rare", "Col": "Rare", "Major": "Rare", "Dr": "Rare", "Rev": "Rare"}
        X_copy["Title"] = X_copy["Title"].replace(title_map).fillna("Mr")
        core_titles = ["Mr", "Miss", "Mrs", "Master"]
        X_copy.loc[~X_copy["Title"].isin(core_titles), "Title"] = "Rare"
        
        self.age_medians_ = X_copy.groupby("Title")["Age"].median().to_dict()
        self.global_age_median_ = X_copy["Age"].median()
        self.fare_median_ = X_copy["Fare"].median()
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Exact Title Processing
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Dona": "Rare", "Countess": "Rare", "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare", "Capt": "Rare", "Col": "Rare", "Major": "Rare", "Dr": "Rare", "Rev": "Rare"}
        df["Title"] = df["Title"].replace(title_map).fillna("Mr")
        core_titles = ["Mr", "Miss", "Mrs", "Master"]
        df.loc[~df["Title"].isin(core_titles), "Title"] = "Rare"
        
        # 2. Strict Age Imputation protecting the 'Master' (Boy) signal
        for title, med in self.age_medians_.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        df["Age"] = df["Age"].fillna(self.global_age_median_)
        
        # 3. Robust Fare Logging
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Fare_Log"] = np.log1p(df["Fare"])
        
        # 4. Critical Kaggle Feature: Family Size Bins (Alone vs Small vs Large)
        family_size = df["SibSp"] + df["Parch"] + 1
        df["FamilySizeBin"] = pd.cut(family_size, bins=[0, 1, 4, 100], labels=["Alone", "Small", "Large"], right=True)
        
        # 5. Core Macro-Interactions (IsBoy captures the only subset of males who reliably survived)
        df["IsBoy"] = ((df["Title"] == "Master") | ((df["Sex"] == "male") & (df["Age"] <= 12))).astype(int)
        df["Age_Pclass"] = df["Age"] * df["Pclass"]
        
        # Purge absolute noise and highly correlated excess
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId", "Fare", "SibSp", "Parch"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        df["Pclass"] = df["Pclass"].astype(str)
        return df

def get_preprocessor():
    numeric_features = ["Age", "Fare_Log", "Age_Pclass", "IsBoy"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "FamilySizeBin"]

    # RobustScaler inherently manages severe Titanic outliers better than standard scaler
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()) 
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

def main():
    print("Loading datasets...")
    train_df, test_df = load_data()
    
    X_train = train_df.drop(columns=["Survived"])
    y_train = train_df["Survived"]
    test_ids = pd.DataFrame({"PassengerId": test_df["PassengerId"]})
    
    base_pipeline = Pipeline([
        ("feature_engineering", GeneralizationFeatureTransformer()),
        ("preprocessor", get_preprocessor())
    ])
    
    # ─── MODELS & ENSEMBLE DIVERSITY ───
    # 1. High Regularization Linear Model (Extracts smooth absolute macro trends)
    lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)
    
    # 2. Shallow Gradient Boost (Extracts non-linear boundaries safely)
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_split=4, subsample=0.8, random_state=RANDOM_STATE
    )
    
    # 3. Random Forest (Adds extreme ensemble diversity guarding against variance)
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=RANDOM_STATE
    )
    
    # We combine Linear + Boosted + Bagged approaches. 
    # This triad physically prevents the ensemble from submitting high-confidence errors on Test data.
    voting_clf = VotingClassifier(
        estimators=[('lr', lr_model), ('gb', gb_model), ('rf', rf_model)],
        voting='soft', weights=[1, 2, 1] 
    )
    
    full_model = Pipeline([
        ('base', base_pipeline),
        ('classifier', voting_clf)
    ])
    
    print("=" * 65)
    print("  MODEL EVALUATION (10-Fold Stratified Cross-Validation)")
    print("=" * 65)
    
    # Ditch 80/20 Holdout entirely. We use 10-Fold CV covering 100% of train data implicitly.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(full_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"10-Fold CV Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"10-Fold CV Std Deviation: ±{cv_scores.std():.4f}")
    if cv_scores.std() > 0.05:
        print("⚠️ Warning: High fold variance detected. Generalization risk is elevated.")
    else:
        print("✅ Stable fold distribution. High confidence in Leaderboard generalization.")
    print("=" * 65)
    
    # ─── FINAL TRAINING ───
    print("\nTraining Final Pipeline on FULL 100% dataset to maximize pattern extraction...")
    full_model.fit(X_train, y_train)

    print("Generating pure ensemble predictions...")
    final_preds = full_model.predict(test_df)
    
    sub = pd.DataFrame({"PassengerId": test_ids["PassengerId"], "Survived": final_preds})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")

if __name__ == "__main__":
    main()
