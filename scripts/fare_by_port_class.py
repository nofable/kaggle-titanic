"""Follow-up to H8: did Cherbourg passengers pay more *within* each class?

Prints mean and median Fare by Pclass x Port, plus counts, for train.csv.
Median is included because Fare is right-skewed (a few huge fares pull the mean up).

Usage (from the repo root):
    uv run python scripts/fare_by_port_class.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
PORTS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["Embarked"])
    df["Port"] = df["Embarked"].map(PORTS)

    for stat in ("mean", "median"):
        table = pd.crosstab(df["Pclass"], df["Port"], values=df["Fare"], aggfunc=stat)
        print(f"{stat.capitalize()} Fare:")
        print(table.round(2).to_string())
        print()

    print("Passenger count n:")
    print(pd.crosstab(df["Pclass"], df["Port"]).to_string())


if __name__ == "__main__":
    main()
