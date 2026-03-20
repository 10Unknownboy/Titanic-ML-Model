"""
Titanic Survival Prediction — Generalization Optimized Pipeline
==============================================================
Refactored to eliminate cross-validation bias, reduce overfitting,
and align local CV scores with the actual Kaggle leaderboard.

Key Fixes:
  • Replaced StratifiedKFold with RepeatedStratifiedKFold.
  • Added a 20% hold-out validation set to simulate the Kaggle test set.
  • Heavily regularized tree models (GBM, RF) to prevent overfitting.
  • Stripped noisy features (HasCabin) that suffer from distribution shift.

Usage (Colab):
  !pip install pandas numpy scikit-learn matplotlib seaborn
  !python titanic_pipeline.py
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
    # Simplify titles aggressively to avoid overfitting to rare test-set occurrences
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Miss", "Dona": "Miss", "Countess": "Mrs",  # Female rare -> regular
        "Sir": "Mr", "Don": "Mr", "Jonkheer": "Mr",         # Male rare -> regular
        "Capt": "Mr", "Col": "Mr", "Major": "Mr", "Dr": "Mr", "Rev": "Mr" # Male officer -> Mr
    }
    # Notice we map almost all rare titles back to standard ones to drastically reduce overfitting
    df["Title"] = df["Title"].replace(title_map)
    df["Title"] = df["Title"].fillna("Mr")
    return df


def feature_engineering(df):
    """
    Apply row-wise feature engineering.
    Kept minimal and robust to prevent distribution shift generalization errors.
    """
    df = df.copy()
    df = extract_title(df)
    
    # Family Features - keep simple
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    
    # Fare - Log Transform to handle skewness
    df["Fare"] = np.log1p(df["Fare"])
    
    # Drop columns that are noisy, leak-prone, or processed
    # Dropped 'Cabin' entirely as its missingness distribution shifts heavily in test
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    return df


def impute_age(train, test):
    """ Impute Age using median per Title, trained strictly on the train set """
    title_age_median = train.groupby("Title")["Age"].median()
    
    for df in [train, test]:
        for title, med in title_age_median.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        df["Age"] = df["Age"].fillna(title_age_median.median())
        
    return train, test


def get_preprocessor():
    """ 
    Creates a scikit-learn ColumnTransformer. 
    """
    # Removed highly volatile features
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    return preprocessor


def build_models(preprocessor):
    """
    Build extensively regularized models to prevent 84% CV -> 76% Kaggle drops.
    """
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE))
    ])
    
    svc_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        # Stronger regularization (C=0.5 instead of 1.0)
        ('classifier', SVC(C=0.5, gamma='scale', probability=True, random_state=RANDOM_STATE))
    ])

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        # Regularized RF: shallower depth, more samples per leaf
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=4, random_state=RANDOM_STATE))
    ])
    
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        # Regularized GB: tiny learning rate, subsample to prevent overfitting noise
        ('classifier', GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, min_samples_leaf=5, subsample=0.7, random_state=RANDOM_STATE))
    ])
    
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_pipe),
            ('rf', rf_pipe),
            ('gb', gb_pipe),
            ('svc', svc_pipe)
        ],
        voting='soft'
    )
    
    return {
        "Logistic Regression": lr_pipe,
        "Random Forest": rf_pipe,
        "Gradient Boosting": gb_pipe,
        "SVC": svc_pipe,
        "Voting Ensemble": voting_clf
    }


def generate_plots(model, X_train, y_train):
    """ Generates graphical representations for the data and model """
    print("\n[+] Generating graphical representations...")
    try:
        if hasattr(model, 'named_estimators_'):
            preprocessor = model.named_estimators_['gb'].named_steps['preprocessor']
            clf = model.named_estimators_['gb'].named_steps['classifier']
        else:
            preprocessor = model.named_steps['preprocessor']
            clf = model.named_steps['classifier']
            
        cat_enc = preprocessor.named_transformers_['cat'].named_steps['ohe']
        cat_features = preprocessor.transformers_[1][2]
        num_features = preprocessor.transformers_[0][2]
        
        ohe_features = list(cat_enc.get_feature_names_out(cat_features))
        feature_names = num_features + ohe_features
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances ({type(clf).__name__})")
            plt.bar(range(len(importances)), importances[indices], align="center", color='#2ca02c')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("feature_importances.png")
            print("  -> Saved 'feature_importances.png'")
        
        X_transformed = preprocessor.transform(X_train)
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        df_transformed['Survived'] = y_train.values
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_transformed.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        print("  -> Saved 'correlation_heatmap.png'")
        
    except Exception as e:
        print(f"  -> Could not generate plots: {e}")


def main():
    print("Loading data...")
    full_train_df, test_df = load_data()
    
    print("Engineering features...")
    full_train_df = feature_engineering(full_train_df)
    test_df = feature_engineering(test_df)
    
    print("Imputing age without leakage...")
    full_train_df, test_df = impute_age(full_train_df, test_df)
    
    # ─── HOLD-OUT VALIDATION SPLIT (Simulate Kaggle test set) ────────────
    df_train, df_val = train_test_split(
        full_train_df, test_size=0.2, stratify=full_train_df["Survived"], random_state=RANDOM_STATE
    )
    
    X_train_cv = df_train.drop(columns=["Survived"])
    y_train_cv = df_train["Survived"]
    
    X_val = df_val.drop(columns=["Survived"])
    y_val = df_val["Survived"]
    
    X_test_final = test_df
    test_ids = pd.read_csv(TEST_PATH)["PassengerId"]
    
    preprocessor = get_preprocessor()
    models = build_models(preprocessor)
    
    # Use RepeatedStratifiedKFold for much more stable CV accuracy estimates
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    
    print("=" * 75)
    print("  MODEL EVALUATION (3x5 Repeated Stratified CV & 20% Holdout)")
    print("=" * 75)
    
    best_name = None
    best_val_score = 0
    best_model = None
    
    for name, model in models.items():
        # 1. Cross Validation on the 80% training set
        scores = cross_val_score(model, X_train_cv, y_train_cv, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_cv = scores.mean()
        std_cv = scores.std()
        
        # 2. Train on 80% and evaluate on 20% holdout
        model.fit(X_train_cv, y_train_cv)
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        
        # Flag models where mean_cv overestimates val_acc noticeably
        overfit_flag = "⚠️ Guard" if (mean_cv - val_acc) > 0.03 else "✅ Flow"
        
        print(f"  {name:<20s} | CV: {mean_cv:.4f} (±{std_cv:.3f}) | Val: {val_acc:.4f} {overfit_flag}")
        
        # Select BEST based on HOLDOUT VALIDATION, simulating actual unseen Kaggle test
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_name = name
            best_model = model
            
    print("=" * 75)
    print(f"\n★ Best Generalizing Model: {best_name} (Holdout Acc: {best_val_score:.4f})")
    
    # ─── FINAL TRAINING ──────────────────────────────────────────────────
    print(f"\nTraining {best_name} on the full 100% training set...")
    X_full_train = full_train_df.drop(columns=["Survived"])
    y_full_train = full_train_df["Survived"]
    best_model.fit(X_full_train, y_full_train)
    
    # Generate graphical plots
    generate_plots(best_model, X_full_train, y_full_train)

    print("\nGenerating predictions...")
    preds = best_model.predict(X_test_final)
    
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": preds.astype(int)})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")


if __name__ == "__main__":
    main()
