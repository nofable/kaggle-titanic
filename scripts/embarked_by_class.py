"""H8 test: does Embarked still matter once we control for Pclass?

Prints, for train.csv:
  1. Class mix per port (row %), i.e. who boarded where.
  2. Survival rate by port, overall (unstratified).
  3. Survival rate by port *within each class* (stratified), with counts n.

Usage (from the repo root):
    uv run python scripts/embarked_by_class.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
PORTS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["Embarked"])
    df["Port"] = df["Embarked"].map(PORTS)

    print("=== 1. Class mix per port (share of each port's passengers) ===")
    print(pd.crosstab(df["Port"], df["Pclass"], normalize="index").round(2).to_string())

    print("\n=== 2. Survival rate by port (unstratified) ===")
    overall = df.groupby("Port")["Survived"].agg(rate="mean", n="size")
    print(overall.round(2).to_string())

    print("\n=== 3. Survival rate by port, within each class (stratified) ===")
    rate = pd.crosstab(df["Pclass"], df["Port"], values=df["Survived"], aggfunc="mean")
    n = pd.crosstab(df["Pclass"], df["Port"])
    print("Survival rate:")
    print(rate.round(2).to_string())
    print("\nPassenger count n:")
    print(n.to_string())


if __name__ == "__main__":
    main()
