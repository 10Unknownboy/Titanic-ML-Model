"""
Titanic ML — Feature Engineering
=================================
Core feature engineering pipeline.

Key score-boosting features:
  - FamilySurvivalRate: Survival rate of other family members (by surname)
  - TicketSurvivalRate: Survival rate of other passengers sharing the same ticket
  - Deck: Extracted from Cabin letter
  - TicketFrequency: Number of passengers sharing the same ticket
  - Title, FamilySizeBin, IsBoy, IsMarriedWoman, AgeGroup, etc.

The family/ticket survival features require combining train+test data
so that group membership can propagate across the train/test boundary.
Only known survival labels (from train rows) are used — no test leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import get_logger

logger = get_logger("feature_engineering")


class TitanicFeatureEngineer:
    """Transforms raw Titanic data into ML-ready features.

    Usage::

        engineer = TitanicFeatureEngineer(config)
        X_train, y_train, X_test, test_ids = engineer.fit_transform(train_df, test_df)
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._feat = self.config.features

        # Learned statistics (fitted from training data)
        self.age_medians_by_title: dict[str, float] = {}
        self.global_age_median: float = 0.0
        self.fare_median: float = 0.0

    def _add_group_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kaggle Women-Child-Group (WCG) survival trick.
        
        Extracts Surname and groups by Surname and Ticket.
        Tracks the survival of Women and Boys in the group.
        If a WCG member survived, GroupSurvival = 1.
        If all WCG members died, GroupSurvival = 0.
        Otherwise 0.5.
        """
        df["Surname"] = df["Name"].apply(lambda x: x.split(",")[0].strip())
        df["GroupSurvival"] = 0.5
        
        is_wcg = (df["Sex"] == "female") | (df["Title"] == "Master")
        
        for grp_col in ["Surname", "Ticket"]:
            for _, group in df.groupby(grp_col):
                if len(group) <= 1:
                    continue
                for idx in group.index:
                    others = group.drop(idx)
                    # Only look at WCG members (Women and Boys) in the 'others' group
                    wcg_others = others[is_wcg.loc[others.index]]
                    
                    known_survived = wcg_others[wcg_others["Survived"] == 1]
                    known_died = wcg_others[wcg_others["Survived"] == 0]
                    
                    if len(known_survived) > 0:
                        df.at[idx, "GroupSurvival"] = 1.0
                    elif len(known_died) > 0:
                        df.at[idx, "GroupSurvival"] = 0.0
                        
        return df

    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Engineer features for both train and test sets."""
        logger.info("Starting feature engineering...")

        train_ids = train_df["PassengerId"].copy()
        test_ids = test_df["PassengerId"].copy()
        y_train = train_df["Survived"].copy()

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["_is_train"] = True
        test_df["_is_train"] = False

        combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)

        # --- Sequential feature engineering ---
        combined = self._extract_title(combined)
        combined = self._impute_age(combined)
        combined = self._process_fare(combined)
        combined = self._family_features(combined)
        combined = self._ticket_features(combined)
        combined = self._cabin_features(combined)
        combined = self._interaction_features(combined)
        combined = self._add_group_survival(combined)
        combined = self._age_group(combined)
        combined = self._married_woman(combined)

        # Drop raw columns not needed for modeling
        drop_cols = [
            "Name", "Ticket", "Cabin", "PassengerId",
            "Fare", "SibSp", "Parch", "Survived", "_is_train", "Surname",
        ]
        combined = combined.drop(
            columns=[c for c in drop_cols if c in combined.columns],
            errors="ignore",
        )

        # Cast Pclass to string for one-hot encoding
        combined["Pclass"] = combined["Pclass"].astype(str)

        # Split back
        X_train = combined[combined.index < len(train_df)].reset_index(drop=True)
        X_test = combined[combined.index >= len(train_df)].reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        test_ids = test_ids.reset_index(drop=True)

        feature_count = X_train.shape[1]
        logger.info("Feature engineering complete: %d features", feature_count)
        logger.info("Features: %s", list(X_train.columns))

        return X_train, y_train, X_test, test_ids

    # ------------------------------------------------------------------
    # Individual feature transforms
    # ------------------------------------------------------------------

    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract title from Name and map to canonical groups."""
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

        title_map = self._feat.get("title_map", {})
        core_titles = self._feat.get("core_titles", ["Mr", "Miss", "Mrs", "Master"])

        df["Title"] = df["Title"].replace(title_map).fillna("Mr")
        df.loc[~df["Title"].isin(core_titles), "Title"] = "Rare"

        logger.debug("Title distribution:\n%s", df["Title"].value_counts().to_string())
        return df

    def _impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing Age using Title-group medians (fitted on train rows)."""
        train_mask = df["_is_train"] == True  # noqa: E712

        # Fit medians on training data only
        self.age_medians_by_title = (
            df.loc[train_mask].groupby("Title")["Age"].median().to_dict()
        )
        self.global_age_median = df.loc[train_mask, "Age"].median()

        # Apply to all rows
        for title, median_age in self.age_medians_by_title.items():
            mask = df["Age"].isnull() & (df["Title"] == title)
            df.loc[mask, "Age"] = median_age

        df["Age"] = df["Age"].fillna(self.global_age_median)

        logger.debug("Age imputation complete. Remaining nulls: %d", df["Age"].isnull().sum())
        return df

    def _process_fare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing Fare and apply log transform."""
        train_mask = df["_is_train"] == True  # noqa: E712
        self.fare_median = df.loc[train_mask, "Fare"].median()

        df["Fare"] = df["Fare"].fillna(self.fare_median)
        df["Fare_Log"] = np.log1p(df["Fare"])

        return df

    def _family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create family size, bins, fare-per-person, and surname."""
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        bins = self._feat.get("family_size_bins", [0, 1, 4, 100])
        labels = self._feat.get("family_size_labels", ["Alone", "Small", "Large"])
        df["FamilySizeBin"] = pd.cut(
            df["FamilySize"], bins=bins, labels=labels, right=True,
        )

        df["FarePerPerson"] = df["Fare_Log"] / df["FamilySize"]

        # Extract surname for family grouping
        df["Surname"] = df["Name"].apply(lambda x: x.split(",")[0].strip())

        return df

    def _ticket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ticket frequency (passengers sharing the same ticket)."""
        ticket_counts = df["Ticket"].value_counts()
        df["TicketFrequency"] = df["Ticket"].map(ticket_counts)

        return df

    def _cabin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract deck from Cabin letter and create HasCabin flag."""
        df["HasCabin"] = df["Cabin"].notna().astype(int)

        df["Deck"] = df["Cabin"].apply(
            lambda x: x[0] if pd.notna(x) and len(str(x)) > 0 else "Unknown"
        )
        # Group rare decks
        df["Deck"] = df["Deck"].replace({"T": "Unknown", "G": "Unknown"})

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and derived features."""
        boy_threshold = self._feat.get("boy_age_threshold", 12)

        df["IsBoy"] = (
            (df["Title"] == "Master")
            | ((df["Sex"] == "male") & (df["Age"] <= boy_threshold))
        ).astype(int)

        df["Age_Pclass"] = df["Age"] * df["Pclass"].astype(float)

        return df

    def _compute_family_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute survival rate of OTHER family members.

        For each passenger, looks at other passengers with the same
        surname and similar family size. Uses only known survival labels
        (from training rows). Passengers with no known family members
        get the default rate (0.5).
        """
        default_rate = self._feat.get("default_survival_rate", 0.5)

        # Create family group key (surname + family size bucket)
        df["_family_key"] = df["Surname"] + "_" + df["FamilySize"].astype(str)

        family_survival = pd.Series(default_rate, index=df.index, dtype=float)

        for _, group in df.groupby("_family_key"):
            if len(group) <= 1:
                continue  # Solo passengers get the default

            for idx in group.index:
                # Other members of this family
                others = group.drop(idx)
                # Only use known survival labels (train rows)
                known = others[others["Survived"].notna()]

                if len(known) > 0:
                    family_survival.at[idx] = known["Survived"].mean()
                # else: keep default_rate

        df["FamilySurvivalRate"] = family_survival
        df = df.drop(columns=["_family_key"], errors="ignore")

        logger.info(
            "FamilySurvivalRate — non-default entries: %d / %d",
            (df["FamilySurvivalRate"] != default_rate).sum(),
            len(df),
        )
        return df

    def _compute_ticket_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute survival rate of OTHER passengers with the same ticket.

        Same logic as family survival but grouped by ticket number.
        """
        default_rate = self._feat.get("default_survival_rate", 0.5)

        ticket_survival = pd.Series(default_rate, index=df.index, dtype=float)

        for _, group in df.groupby("Ticket"):
            if len(group) <= 1:
                continue

            for idx in group.index:
                others = group.drop(idx)
                known = others[others["Survived"].notna()]

                if len(known) > 0:
                    ticket_survival.at[idx] = known["Survived"].mean()

        df["TicketSurvivalRate"] = ticket_survival

        logger.info(
            "TicketSurvivalRate — non-default entries: %d / %d",
            (df["TicketSurvivalRate"] != default_rate).sum(),
            len(df),
        )
        return df

    def _age_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bin age into categorical groups."""
        bins = self._feat.get("age_bins", [0, 12, 20, 40, 60, 120])
        labels = self._feat.get("age_labels", ["Child", "Teen", "Adult", "MiddleAge", "Senior"])

        df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

        return df

    def _married_woman(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag married women (Mrs title — very high survival)."""
        df["IsMarriedWoman"] = (df["Title"] == "Mrs").astype(int)
        return df


def engineer_single_passenger(
    passenger: dict,
    config: Config | None = None,
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Engineer features for a single passenger (for dashboard prediction).

    Args:
        passenger: Dict with raw passenger features.
        config: Project configuration.
        train_df: Training data for reference stats. Loads from disk if None.

    Returns:
        Single-row DataFrame with engineered features.
    """
    config = config or Config()

    if train_df is None:
        from src.data_loader import load_data
        train_df, _ = load_data(config)

    # Create a minimal test row
    test_row = pd.DataFrame([passenger])

    # Ensure all required columns exist with defaults
    for col in ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp",
                 "Parch", "Ticket", "Fare", "Cabin", "Embarked"]:
        if col not in test_row.columns:
            test_row[col] = np.nan if col in ["Age", "Fare", "Cabin"] else ""

    if "PassengerId" not in test_row.columns or pd.isna(test_row["PassengerId"].iloc[0]):
        test_row["PassengerId"] = 0

    engineer = TitanicFeatureEngineer(config)
    _, _, X_single, _ = engineer.fit_transform(train_df, test_row)

    return X_single
