"""H2 check: survival rate by age bin, split by sex.

Rows with missing Age are excluded (count printed). Each bar is labelled with
its passenger count n, because small bins give noisy rates.

Usage (from the repo root):
    uv run python scripts/age_survival_by_sex.py [BIN_WIDTH]   # default 10
Writes figures/age_survival_by_sex_<BIN_WIDTH>y.png and prints the table.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "titanic-data"
FIG_DIR = ROOT / "figures"

# Categorical slots 1 and 2 of the validated default palette.
COLORS = {"female": "#2a78d6", "male": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main() -> None:
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    df = pd.read_csv(DATA_DIR / "train.csv")
    n_missing = df["Age"].isna().sum()
    df = df.dropna(subset=["Age"])

    # Integer edges; extend past the max so the oldest passenger is included (bins are [lo, hi)).
    edges = np.arange(0, int(df["Age"].max()) + width + 1, width)
    edges = edges[: np.searchsorted(edges, df["Age"].max(), side="right") + 1]
    labels = [f"{lo}–{lo + width - 1}" for lo in edges[:-1]]
    df["AgeBin"] = pd.cut(df["Age"], bins=edges, right=False, labels=labels)

    stats = (
        df.groupby(["AgeBin", "Sex"], observed=False)["Survived"]
        .agg(rate="mean", n="size")
        .unstack("Sex")
    )
    print(f"Excluded {n_missing} rows with missing Age.\n")
    print(stats.round(2).to_string())

    x = np.arange(len(labels))
    bar_w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, sex in enumerate(["female", "male"]):
        rates = stats[("rate", sex)].to_numpy()
        ns = stats[("n", sex)].to_numpy()
        pos = x + (i - 0.5) * (bar_w + 0.02)  # small gap between paired bars
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

    ax.set_xticks(x, labels)
    ax.set_xlabel("Age (years)", color=INK)
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
    ax.set_title(
        f"Survival rate by age ({width}-year bins) and sex — train.csv, "
        f"{n_missing} rows with missing Age excluded",
        color=INK,
        loc="left",
    )
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"age_survival_by_sex_{width}y.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
