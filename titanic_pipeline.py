"""
Titanic Survival Prediction — Kaggle Robust Generalization Pipeline
===================================================================
A clean, minimal, leak-free pipeline targeting max out-of-sample Kaggle score (0.80+).
"""

import pandas as pd
import numpy as np
import warnings
import os
import glob

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

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
        
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        df["Pclass"] = df["Pclass"].astype(str)
        return df

def get_preprocessor():
    numeric_features = ["Age", "Fare_Log", "SibSp", "Parch", "FamilySize", "Age_Pclass"]
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

def build_models(preprocessor):
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE))
    ])
    
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE))
    ])
    
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, min_samples_leaf=5, subsample=0.8, random_state=RANDOM_STATE))
    ])
    
    voting_clf = VotingClassifier(
        estimators=[('lr', lr_pipe), ('rf', rf_pipe), ('gb', gb_pipe)],
        voting='soft', weights=[1, 1, 3] # Favors GradientBoosting strictly
    )
    
    return {
        "Logistic Regression": lr_pipe,
        "Random Forest": rf_pipe,
        "Gradient Boosting": gb_pipe,
        "Weighted Voting": voting_clf
    }

def main():
    print("Loading data...")
    full_train_df, test_df = load_data()
    
    df_train, df_val = train_test_split(
        full_train_df, test_size=0.2, stratify=full_train_df["Survived"], random_state=RANDOM_STATE
    )
    
    X_train_cv = df_train.drop(columns=["Survived"])
    y_train_cv = df_train["Survived"]
    
    X_val = df_val.drop(columns=["Survived"])
    y_val = df_val["Survived"]
    
    X_test_final = test_df
    test_ids = pd.DataFrame({"PassengerId": pd.read_csv(TEST_PATH)["PassengerId"]})
    
    preprocessor = get_preprocessor()
    models = build_models(preprocessor)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    
    print("=" * 65)
    print("  MODEL EVALUATION (3x5 Repeated CV & 20% Holdout)")
    print("=" * 65)
    
    best_name, best_val_score, best_model = None, 0, None
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train_cv, y_train_cv, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_cv, std_cv = scores.mean(), scores.std()
        
        model.fit(X_train_cv, y_train_cv)
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        
        flag = "⚠️ Guard" if (mean_cv - val_acc) > 0.03 else "✅ Flow"
        print(f"  {name:<22s} | CV: {mean_cv:.4f} (±{std_cv:.3f}) | Val: {val_acc:.4f} {flag}")
        
        # Strictly prioritize holdout score!
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_name = name
            best_model = model
            
    print("=" * 65)
    print(f"\n★ Best Generalizing Model: {best_name} (Holdout Acc: {best_val_score:.4f})")
    
    print(f"Training {best_name} on the full 100% training set...")
    X_full_train = full_train_df.drop(columns=["Survived"])
    y_full_train = full_train_df["Survived"]
    best_model.fit(X_full_train, y_full_train)

    print("Generating predictions...")
    preds = best_model.predict(X_test_final)
    
    sub = pd.DataFrame({"PassengerId": test_ids["PassengerId"], "Survived": preds.astype(int)})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")

if __name__ == "__main__":
    main()
