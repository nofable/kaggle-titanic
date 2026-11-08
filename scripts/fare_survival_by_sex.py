"""Survival rate by Fare band, split by sex (train.csv).

Two binning modes:
  BAND_WIDTH (e.g. 20) - equal-width bands [lo, lo + width); empty bands dropped
  qN (e.g. q8)         - N quantile bins (pd.qcut): roughly equal passenger
                         counts per bin. Bins may merge where many passengers
                         share the same fare (ties), so you can get fewer than N.
  logN (e.g. log8)     - N equal-width bins on log(1 + value), labelled in
                         original units; empty bins dropped
Each bar shows its count n.

COLUMN is Fare (default), FarePerFamily (= Fare / FamilySize) or
FarePerPerson (= Fare / passengers on the same ticket, counted over train + test).

Usage (from the repo root):
    uv run python scripts/fare_survival_by_sex.py [BAND_WIDTH | qN | logN] [COLUMN]
Writes figures/<column>_survival_by_sex_<arg>.png and prints the table.
"""

import sys
from pathlib import Path

from kaggle_titanic.features import fare_per_family, fare_per_person

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "titanic-data"
FIG_DIR = ROOT / "figures"

COLORS = {"female": "#2a78d6", "male": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "20"
    col = sys.argv[2] if len(sys.argv) > 2 else "Fare"
    if col not in ("Fare", "FarePerFamily", "FarePerPerson"):
        sys.exit("COLUMN must be Fare, FarePerFamily or FarePerPerson")

    df = pd.read_csv(DATA_DIR / "train.csv")
    df["FarePerFamily"] = fare_per_family(df)
    everyone = pd.concat([df, pd.read_csv(DATA_DIR / "test.csv")], ignore_index=True)
    df["FarePerPerson"] = fare_per_person(df, everyone)
    if arg.startswith("log"):
        n_bins = int(arg[3:])
        log_edges = np.linspace(0, np.log1p(df[col].max()), n_bins + 1)
        log_edges[-1] += 1e-9  # include the maximum in the last bin
        edges = np.expm1(log_edges)
        labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
        df["FareBand"] = pd.cut(df[col], bins=edges, right=False, labels=labels)
        desc = f"{n_bins} equal-width bins on log(1 + {col}); empty bins omitted"
    elif arg.startswith("q"):
        q = int(arg[1:])
        _, edges = pd.qcut(df[col], q=q, retbins=True, duplicates="drop")
        labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
        df["FareBand"] = pd.qcut(df[col], q=q, labels=labels, duplicates="drop")
        desc = f"{len(labels)} quantile bins"
    else:
        width = int(arg)
        edges = np.arange(0, int(df[col].max()) + width + 1, width)
        labels = [f"{lo}–{lo + width}" for lo in edges[:-1]]
        df["FareBand"] = pd.cut(df[col], bins=edges, right=False, labels=labels)
        desc = f"width {width}; empty bands omitted"

    stats = (
        df.groupby(["FareBand", "Sex"], observed=False)["Survived"]
        .agg(rate="mean", n="size")
        .unstack("Sex")
    )
    stats = stats[(stats[("n", "female")] + stats[("n", "male")]) > 0]  # drop empty bands
    print(stats.round(2).to_string(na_rep=""))

    x = np.arange(len(stats))
    bar_w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
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
                fontsize=7,
                color=MUTED,
            )

    ax.set_xticks(x, stats.index.astype(str), rotation=45, ha="right")
    ax.set_xlabel(f"{col} band ({desc})", color=INK)
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
    ax.set_title(f"Survival rate by {col} band ({desc}) and sex — train.csv", color=INK, loc="left")
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{col.lower()}_survival_by_sex_{arg}.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
