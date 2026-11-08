"""Profile the gender rule's two kinds of mistake (train.csv).

  Women: who died vs who survived  (the gender rule wrongly predicts 'survived' for the women who died)
  Men:   who survived vs who died  (the gender rule wrongly predicts 'died' for the men who survived)

For each sex, compares the two outcome groups on class, age, title, family
size, per-person fare, port, and cabin. Shares are within each outcome group.

Usage (from the repo root):
    uv run python scripts/subset_profiles.py
"""

import pandas as pd

from kaggle_titanic.features import cabin_known, family_size, fare_per_person, title_group
from kaggle_titanic.pipeline import load_raw

AGE_BANDS = [0, 15, 25, 50, 100]
AGE_LABELS = ["0-14", "15-24", "25-49", "50+"]
FAM_BANDS = [0, 1, 4, 20]
FAM_LABELS = ["alone", "2-4", "5+"]


def shares(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Share of each value of `col` within died / survived, plus counts."""
    t = pd.crosstab(df[col], df["Outcome"], normalize="columns").round(2)
    t["n_died"] = pd.crosstab(df[col], df["Outcome"]).get("died", 0)
    t["n_survived"] = pd.crosstab(df[col], df["Outcome"]).get("survived", 0)
    return t


def profile(df: pd.DataFrame, title: str) -> None:
    counts = df["Outcome"].value_counts()
    print(f"\n{'=' * 70}\n{title}: died n={counts.get('died', 0)}, survived n={counts.get('survived', 0)}\n{'=' * 70}")
    for col in ["Pclass", "AgeBand", "Title", "Family", "Embarked", "CabinKnown"]:
        print(f"\n--- {col} (share within each outcome) ---")
        print(shares(df, col).to_string())
    print("\n--- medians ---")
    print(df.groupby("Outcome")[["Age", "FarePerPerson", "FamilySize"]].median().round(1).to_string())


def main() -> None:
    pd.set_option("display.width", 200)
    train, test = load_raw()
    everyone = pd.concat([train, test], ignore_index=True)

    df = train.copy()
    df["Outcome"] = df["Survived"].map({0: "died", 1: "survived"})
    df["Title"] = title_group(df["Name"])
    df["FamilySize"] = family_size(df)
    df["Family"] = pd.cut(df["FamilySize"], FAM_BANDS, labels=FAM_LABELS)
    df["FarePerPerson"] = fare_per_person(df, everyone)
    df["CabinKnown"] = cabin_known(df)
    df["AgeBand"] = pd.cut(df["Age"], AGE_BANDS, right=False, labels=AGE_LABELS).cat.add_categories("missing")
    df["AgeBand"] = df["AgeBand"].fillna("missing")

    profile(df[df["Sex"] == "female"], "WOMEN")
    profile(df[df["Sex"] == "male"], "MEN")


if __name__ == "__main__":
    main()
