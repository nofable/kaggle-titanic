"""H7 check: how many passengers in each class have a recorded Cabin?

Usage (from the repo root):
    uv run python scripts/check_cabins_by_class.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
CLASS_NAMES = {1: "first", 2: "second", 3: "third"}


def cabins_by_class(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("Pclass").agg(
        count=("PassengerId", "size"),
        cabins=("Cabin", "count"),  # .count() skips NaN, so this counts non-missing cabins
    )
    summary.index = summary.index.map(CLASS_NAMES)
    summary.index.name = "class"
    summary.loc["total"] = summary.sum()
    return summary


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    datasets = {
        "train": train,
        "test": test,
        "combined": pd.concat([train, test], ignore_index=True),
    }

    for name, df in datasets.items():
        print(f"\n=== {name} ({len(df)} rows) ===")
        print(cabins_by_class(df).to_string())


if __name__ == "__main__":
    main()
