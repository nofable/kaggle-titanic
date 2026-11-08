"""Collapse raw titles into Mr / Mrs / Miss / Master / Fancy and summarise.

Uses kaggle_titanic.features.title_group (mapping lives there).

Usage (from the repo root):
    uv run python scripts/title_groups.py
"""

from pathlib import Path

import pandas as pd

from kaggle_titanic.features import TITLE_GROUPS, extract_title, title_group

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
ORDER = ["Mr", "Mrs", "Miss", "Master", "Fancy"]


def main() -> None:
    pd.set_option("display.width", 200)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    both = pd.concat([train.assign(set="train"), test.assign(set="test")], ignore_index=True)

    raw = extract_title(both["Name"])
    unmapped = sorted(set(raw.dropna()) - set(TITLE_GROUPS))
    print(f"Raw titles not in the mapping (sent to Fancy): {unmapped or 'none'}")

    both["TitleGroup"] = title_group(both["Name"])

    print("\n=== Which raw titles went where ===")
    print(pd.crosstab(raw, both["TitleGroup"])[ORDER].to_string())

    print(f"\n=== Title groups (train + test, {len(both)} rows) ===")
    summary = both.groupby("TitleGroup").agg(
        n=("Name", "size"),
        n_train=("set", lambda s: (s == "train").sum()),
        n_test=("set", lambda s: (s == "test").sum()),
        share_female=("Sex", lambda s: (s == "female").mean()),
        age_missing=("Age", lambda s: s.isna().sum()),
        age_min=("Age", "min"),
        age_median=("Age", "median"),
        age_max=("Age", "max"),
        survival_rate_train=("Survived", "mean"),
    )
    print(summary.loc[ORDER].round(2).to_string())


if __name__ == "__main__":
    main()
