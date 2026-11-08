"""H4 confounder checks (train.csv), FamilySize = SibSp + Parch + 1.

  1. Males split by title: survival rate by family size for Mr vs Master.
  2. Class mix by family size (all passengers), with survival rate.

Usage (from the repo root):
    uv run python scripts/family_confounders.py
"""

from pathlib import Path

import pandas as pd

from kaggle_titanic.features import title_group

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"


def main() -> None:
    pd.set_option("display.width", 200)
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["TitleGroup"] = title_group(df["Name"])

    print("=== 1. Males: survival by family size, Mr vs Master ===")
    males = df[df["TitleGroup"].isin(["Mr", "Master"])]
    rate = pd.crosstab(males["FamilySize"], males["TitleGroup"], values=males["Survived"], aggfunc="mean")
    n = pd.crosstab(males["FamilySize"], males["TitleGroup"])
    table = rate.round(2).add_prefix("rate_").join(n.add_prefix("n_"))
    print(table.to_string(na_rep=""))

    print("\n=== 2. Class mix by family size (all passengers) ===")
    summary = df.groupby("FamilySize")["Survived"].agg(survival_rate="mean", n="size")
    mix = pd.crosstab(df["FamilySize"], df["Pclass"], normalize="index")
    mix.columns = [f"share_class_{c}" for c in mix.columns]
    print(summary.join(mix).round(2).to_string())


if __name__ == "__main__":
    main()
