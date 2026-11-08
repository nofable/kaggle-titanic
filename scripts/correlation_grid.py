"""Quick-and-dirty correlation heatmap over train.csv.

- Pearson correlation, pairwise: NaNs are ignored pair by pair (pandas default).
- Only columns that can be made numeric cheaply are included:
    * numeric columns as-is
    * Sex -> is_female (0/1)
    * Embarked -> one-hot (Embarked_C / _Q / _S)
- Name, Ticket, Cabin are skipped (free text).

Usage (from the repo root):
    uv run python scripts/correlation_grid.py
Writes figures/correlation_grid.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "titanic-data"
FIG_DIR = ROOT / "figures"


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.select_dtypes("number").copy()
    out["is_female"] = (df["Sex"] == "female").astype(int)
    embarked = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=float)
    embarked[df["Embarked"].isna()] = float("nan")  # keep missing as missing
    return pd.concat([out, embarked], axis=1)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    corr = to_numeric_frame(train).corr()

    # Survived first so its row/column is easy to read.
    order = ["Survived"] + [c for c in corr.columns if c != "Survived"]
    corr = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Pearson correlation, train.csv (pairwise, NaNs ignored)")
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out_path = FIG_DIR / "correlation_grid.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    print("\nCorrelation with Survived, sorted:")
    print(corr["Survived"].drop("Survived").sort_values().to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
