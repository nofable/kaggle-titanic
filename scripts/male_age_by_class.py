"""H10 check: is the 25-50 male survival bump explained by class mix?

For males in train.csv with a known Age, by age band:
  1. Survival rate, count, and class mix (share in 1st/2nd/3rd).
  2. Survival rate by age band *within each class* (stratified), with counts n.

Boys (0-14) are kept as their own band so they don't blur the "under 25" group.

Usage (from the repo root):
    uv run python scripts/male_age_by_class.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
EDGES = [0, 15, 25, 50, 100]
LABELS = ["0-14", "15-24", "25-49", "50+"]


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv")
    men = df[(df["Sex"] == "male") & df["Age"].notna()].copy()
    men["AgeBand"] = pd.cut(men["Age"], bins=EDGES, right=False, labels=LABELS)

    print("=== 1. Males by age band: survival and class mix ===")
    summary = men.groupby("AgeBand", observed=True)["Survived"].agg(rate="mean", n="size")
    class_mix = pd.crosstab(men["AgeBand"], men["Pclass"], normalize="index")
    class_mix.columns = [f"share_class_{c}" for c in class_mix.columns]
    print(summary.join(class_mix).round(2).to_string())

    print("\n=== 2. Survival rate by age band, within each class ===")
    rate = pd.crosstab(men["AgeBand"], men["Pclass"], values=men["Survived"], aggfunc="mean")
    print(rate.round(2).to_string())
    print("\nPassenger count n:")
    print(pd.crosstab(men["AgeBand"], men["Pclass"]).to_string())


if __name__ == "__main__":
    main()
