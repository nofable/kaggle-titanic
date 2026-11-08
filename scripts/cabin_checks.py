"""H6/H7 Cabin checks (train.csv).

  1. Survival rate with vs without a recorded Cabin, within each class.
  2. Same, within each class x sex (to rule out sex mix as the explanation).
  3. Deck letter (first character of Cabin) by class, with survival rate.

Usage (from the repo root):
    uv run python scripts/cabin_checks.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"


def rate_table(df: pd.DataFrame, rows: list[str]) -> pd.DataFrame:
    g = df.groupby(rows + ["CabinKnown"])["Survived"].agg(rate="mean", n="size").unstack("CabinKnown")
    g.columns = [f"{stat}_{'cabin' if known else 'no_cabin'}" for stat, known in g.columns]
    return g[["rate_no_cabin", "rate_cabin", "n_no_cabin", "n_cabin"]]


def main() -> None:
    pd.set_option("display.width", 200)
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["CabinKnown"] = df["Cabin"].notna()
    df["Deck"] = df["Cabin"].str[0]

    print("=== 1. Survival with vs without a recorded Cabin, by class ===")
    print(rate_table(df, ["Pclass"]).round(2).to_string(na_rep=""))

    print("\n=== 2. Same, by class x sex ===")
    print(rate_table(df, ["Pclass", "Sex"]).round(2).to_string(na_rep=""))

    print("\n=== 3. Deck letter: passenger counts by class, and survival rate ===")
    decks = df.dropna(subset=["Deck"])
    counts = pd.crosstab(decks["Deck"], decks["Pclass"])
    counts.columns = [f"n_class_{c}" for c in counts.columns]
    counts["survival_rate"] = decks.groupby("Deck")["Survived"].mean().round(2)
    print(counts.to_string())


if __name__ == "__main__":
    main()
