"""H4 check: survival rate by a family-count column, split by sex.

FEATURE is one of:
  FamilySize (default) = SibSp + Parch + 1, so 1 = travelling without family
  SibSp                = siblings + spouses aboard
  Parch                = parents + children aboard
  TicketGroupSize      = passengers sharing the same Ticket (counted over train + test)
Each bar is labelled with its count n.

Usage (from the repo root):
    uv run python scripts/family_survival_by_sex.py [FEATURE]
Writes figures/<feature>_survival_by_sex.png and prints the table.
"""

import sys

from pathlib import Path

from kaggle_titanic.features import ticket_group_size

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "titanic-data"
FIG_DIR = ROOT / "figures"

COLORS = {"female": "#2a78d6", "male": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
XLABELS = {
    "FamilySize": "Family size (SibSp + Parch + 1; 1 = no family aboard)",
    "SibSp": "SibSp (siblings + spouses aboard)",
    "Parch": "Parch (parents + children aboard)",
    "TicketGroupSize": "Ticket group size (passengers on the same ticket, train + test)",
}


def main() -> None:
    feature = sys.argv[1] if len(sys.argv) > 1 else "FamilySize"
    if feature not in XLABELS:
        sys.exit(f"FEATURE must be one of {list(XLABELS)}")

    df = pd.read_csv(DATA_DIR / "train.csv")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    everyone = pd.concat([df, pd.read_csv(DATA_DIR / "test.csv")], ignore_index=True)
    df["TicketGroupSize"] = ticket_group_size(df, everyone)

    stats = (
        df.groupby([feature, "Sex"])["Survived"]
        .agg(rate="mean", n="size")
        .unstack("Sex")
        .reindex(range(df[feature].min(), df[feature].max() + 1))
    )
    stats[[("n", "female"), ("n", "male")]] = stats[[("n", "female"), ("n", "male")]].fillna(0).astype(int)
    print(stats.round(2).to_string(na_rep=""))

    sizes = stats.index.to_numpy()
    x = np.arange(len(sizes))
    bar_w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, sex in enumerate(["female", "male"]):
        rates = stats[("rate", sex)].to_numpy()
        ns = stats[("n", sex)].to_numpy()
        pos = x + (i - 0.5) * (bar_w + 0.02)
        bars = ax.bar(pos, np.nan_to_num(rates), bar_w, color=COLORS[sex], label=sex)
        for bar, n in zip(bars, ns):
            ax.annotate(
                f"n={n}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=MUTED,
            )

    ax.set_xticks(x, [str(s) for s in sizes])
    ax.set_xlabel(XLABELS[feature], color=INK)
    ax.set_ylabel("Survival rate", color=INK)
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False)
    ax.set_title(f"Survival rate by {feature} and sex — train.csv", color=INK, loc="left")
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{feature.lower()}_survival_by_sex.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
