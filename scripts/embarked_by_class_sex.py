"""Follow-up to H8/H9: is the Cherbourg effect explained by sex mix?

Prints, for train.csv:
  1. Share female by Pclass x Port.
  2. Survival rate by Port within each Pclass x Sex cell, with counts n.

Usage (from the repo root):
    uv run python scripts/embarked_by_class_sex.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
PORTS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["Embarked"])
    df["Port"] = df["Embarked"].map(PORTS)

    print("=== 1. Share female, by class and port ===")
    female = pd.crosstab(df["Pclass"], df["Port"], values=df["Sex"] == "female", aggfunc="mean")
    print(female.round(2).to_string())

    rows = [df["Sex"], df["Pclass"]]
    print("\n=== 2. Survival rate by port, within each sex x class ===")
    rate = pd.crosstab(rows, df["Port"], values=df["Survived"], aggfunc="mean")
    print(rate.round(2).to_string())

    print("\nPassenger count n:")
    print(pd.crosstab(rows, df["Port"]).to_string())


if __name__ == "__main__":
    main()
