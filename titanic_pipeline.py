"""
Titanic Survival Prediction — Optimized Robust Pipeline
======================================================
Refactored to eliminate data leakage, simplify the ensemble, 
and improve out-of-sample generalization (aiming for ~82-84% on Kaggle).

Key Fixes:
  • Dropped leaky family/ticket survival rate features.
  • Stricter Train/Test separation (imputations fit on train only via Pipelines).
  • Simplified stacking to reduce overfitting.
  • Added graphical representations (Feature Importances & Correlation Matrix).

Usage (Colab):
  !pip install pandas numpy scikit-learn matplotlib seaborn
  !python titanic_pipeline.py
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

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
        "Lady": "Mrs", "Dona": "Mrs", "Countess": "Mrs", 
        "Sir": "Mr", "Don": "Mr", "Jonkheer": "Mr",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Dr": "Officer", "Rev": "Officer"
    }
    df["Title"] = df["Title"].replace(title_map)
    # Re-map unmapped rare titles to "Rare"
    common_titles = ["Mr", "Miss", "Mrs", "Master", "Officer"]
    df.loc[~df["Title"].isin(common_titles), "Title"] = "Rare"
    return df


def feature_engineering(df):
    """
    Apply row-wise feature engineering that doesn't depend on population statistics.
    This prevents data leakage between train and test.
    """
    df = df.copy()
    df = extract_title(df)
    
    # Family Features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    
    # Fare - Log Transform to handle skewness
    df["Fare"] = np.log1p(df["Fare"])
    
    # Deck
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    
    # Drop columns that are noisy, leak-prone, or processed
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    return df


def impute_age(train, test):
    """ 
    Impute Age using median per Title, trained STRICTLY on the train set.
    """
    title_age_median = train.groupby("Title")["Age"].median()
    
    for df in [train, test]:
        for title, med in title_age_median.items():
            df.loc[df["Age"].isnull() & (df["Title"] == title), "Age"] = med
        # Fallback for any remaining NaNs
        df["Age"] = df["Age"].fillna(title_age_median.median())
        
    return train, test


def get_preprocessor():
    """ 
    Creates a scikit-learn ColumnTransformer. 
    Keeps numerical and categorical processing robust and completely prevents data leakage.
    """
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin"]

    # Numerical pipeline: Impute missing (if any) then pass through 
    # (Scaling is handled specifically for models that need it inside their own pipelines)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical pipeline: Impute missing with mode, then OneHotEncode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' safely handles test categories not seen in train
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
    Build models wrapped in pipelines.
    Models that require scaling (LR, SVC) get a StandardScaler in their pipeline.
    Tree models (RF, GB) do not.
    """
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE))
    ])
    
    svc_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', SVC(C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE))
    ])

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE))
    ])
    
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE))
    ])
    
    # Simplified Voting Ensemble (Soft Voting)
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
        # Extract the preprocessor and a tree classifier to get feature importances
        if hasattr(model, 'named_estimators_'):
            # It's the Voting Ensemble, grab Gradient Boosting
            preprocessor = model.named_estimators_['gb'].named_steps['preprocessor']
            clf = model.named_estimators_['gb'].named_steps['classifier']
        else:
            preprocessor = model.named_steps['preprocessor']
            clf = model.named_steps['classifier']
            
        # Get feature names after OneHotEncoding
        cat_enc = preprocessor.named_transformers_['cat'].named_steps['ohe']
        cat_features = preprocessor.transformers_[1][2]
        num_features = preprocessor.transformers_[0][2]
        
        ohe_features = list(cat_enc.get_feature_names_out(cat_features))
        feature_names = num_features + ohe_features
        
        # 1. Feature Importance (Only valid if tree based)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title(f"Feature Importances ({type(clf).__name__})")
            plt.bar(range(len(importances)), importances[indices], align="center", color='#2ca02c')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("feature_importances.png")
            print("  -> Saved 'feature_importances.png'")
        
        # 2. Correlation Heatmap
        X_transformed = preprocessor.transform(X_train)
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        df_transformed['Survived'] = y_train.values
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(df_transformed.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        print("  -> Saved 'correlation_heatmap.png'")
        
    except Exception as e:
        print(f"  -> Could not generate plots: {e}")


def main():
    print("Loading data...")
    train_df, test_df = load_data()
    y = train_df["Survived"]
    
    print("Engineering features...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    print("Imputing age without leakage...")
    train_df, test_df = impute_age(train_df, test_df)
    
    # Drop Survived from training features
    X_train = train_df.drop(columns=["Survived"])
    X_test = test_df
    
    # Fetch PassengerIds for submission
    test_ids = pd.read_csv(TEST_PATH)["PassengerId"]
    
    # Setup robust Pipeline
    preprocessor = get_preprocessor()
    models = build_models(preprocessor)
    
    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    print("=" * 60)
    print("  MODEL EVALUATION (5-Fold Stratified CV - Leak-Free)")
    print("=" * 60)
    
    best_name = None
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        # Because we use sklearn Pipelines, data scaling and encoding are correctly
        # fitted purely on the training folds and applied to validation folds.
        scores = cross_val_score(model, X_train, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean()
        std_acc = scores.std()
        print(f"  {name:<25s} {mean_acc:.4f} ± {std_acc:.4f}")
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name
            best_model = model
            
    print("=" * 60)
    print(f"\n★ Best Model Evaluated: {best_name} ({best_score:.4f})")
    
    # Train best model on full training set
    print(f"\nTraining {best_name} on full training set...")
    best_model.fit(X_train, y)
    
    # Generate graphical plots
    generate_plots(best_model, X_train, y)

    # Generate test predictions
    print("\nGenerating predictions...")
    preds = best_model.predict(X_test)
    
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": preds.astype(int)})
    sub.to_csv(SUBMISSION, index=False)
    print(f"✓ Saved {SUBMISSION} ({len(sub)} rows)\n")

if __name__ == "__main__":
    main()
