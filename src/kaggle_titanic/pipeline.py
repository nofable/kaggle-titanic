"""Assemble the 20 model-input columns from raw train/test data.

Every step follows the pipeline rule in JOURNAL.md: feature-only steps
(imputers, scaler, ticket counts) are fitted on train + test features;
nothing here touches `Survived` except returning it as y.

Because nothing here uses the label, the feature matrix can be built once and
reused across all CV folds without leaking labels.
"""

from pathlib import Path

import pandas as pd

from kaggle_titanic.features import (
    AgeImputer,
    EmbarkedImputer,
    FarePerPersonImputer,
    cabin_known,
    deck,
    family_size,
    fare_per_person,
    title_group,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "titanic-data"

# Fixed category lists so train and test always get identical one-hot columns.
PCLASSES = [1, 2, 3]
PORTS = ["C", "Q", "S"]
DECKS = ["A", "B", "C", "D", "E", "F", "G", "U"]
STANDARDISED = ["Age", "FamilySize", "FarePerPerson"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def _one_hot(values: pd.Series, categories: list, prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(values, categories=categories)
    return pd.get_dummies(cat, prefix=prefix, dtype=float).set_axis(values.index)


def build_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X_train, y_train, X_test) with the 20 agreed columns."""
    both = pd.concat([train.drop(columns="Survived"), test], ignore_index=True)

    # Helpers needed by imputers (not model inputs themselves).
    both["TitleGroup"] = title_group(both["Name"])

    # Age: flag first, then impute (class x title median).
    both["AgeMissing"] = both["Age"].isna().astype(float)
    both["Age"] = AgeImputer().fit(both).transform(both)

    # Family and fare.
    both["FamilySize"] = family_size(both).astype(float)
    both["FarePerPerson"] = fare_per_person(both, both)
    both["FarePerPerson"] = FarePerPersonImputer().fit(both).transform(both)

    # Embarked needs FarePerPerson filled first.
    both["Embarked"] = EmbarkedImputer().fit(both).transform(both)

    # Binary features.
    both["is_female"] = (both["Sex"] == "female").astype(float)
    both["CabinKnown"] = cabin_known(both).astype(float)

    # Standardise continuous features (z-score, fitted on train + test).
    for col in STANDARDISED:
        both[col] = (both[col] - both[col].mean()) / both[col].std()

    X = pd.concat(
        [
            both[["is_female"]],
            _one_hot(both["Pclass"], PCLASSES, "Pclass"),
            both[["Age", "AgeMissing", "FamilySize", "FarePerPerson"]],
            _one_hot(both["Embarked"], PORTS, "Embarked"),
            both[["CabinKnown"]],
            _one_hot(deck(both), DECKS, "Deck"),
        ],
        axis=1,
    )

    n_train = len(train)
    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    return X_train, train["Survived"].reset_index(drop=True), X_test
