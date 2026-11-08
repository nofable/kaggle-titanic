"""Two checks on Age:

1. Missing-Age vs known-Age passengers (train.csv): class mix and survival rate.
   Is missingness random, or does it depend on who you were?
2. Titles parsed from Name (train + test combined): how common each one is,
   how many are missing Age, the Age distribution where known, and survival (train only).

Name format is "Surname, Title. Given names", e.g. "Braund, Mr. Owen Harris".

Usage (from the repo root):
    uv run python scripts/missing_age_and_titles.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
TITLE_RE = r",\s*([^.]+)\."  # text between the first comma and the next full stop


def extract_title(names: pd.Series) -> pd.Series:
    return names.str.extract(TITLE_RE, expand=False).str.strip()


def main() -> None:
    pd.set_option("display.width", 200)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # --- 1. Missing vs known Age -------------------------------------------
    train["AgeStatus"] = train["Age"].isna().map({True: "missing", False: "known"})
    print("=== 1. Missing-Age vs known-Age passengers (train) ===")
    summary = train.groupby("AgeStatus")["Survived"].agg(survival_rate="mean", n="size")
    class_mix = pd.crosstab(train["AgeStatus"], train["Pclass"], normalize="index")
    class_mix.columns = [f"share_class_{c}" for c in class_mix.columns]
    print(summary.join(class_mix).round(2).to_string())

    # --- 2. Titles ---------------------------------------------------------
    both = pd.concat([train.assign(set="train"), test.assign(set="test")], ignore_index=True)
    both["Title"] = extract_title(both["Name"])

    unparsed = both["Title"].isna().sum()
    print(f"\n=== 2. Titles parsed from Name (train + test, {len(both)} rows) ===")
    print(f"Names where no title was found: {unparsed}")

    titles = both.groupby("Title").agg(
        n=("Name", "size"),
        n_train=("set", lambda s: (s == "train").sum()),
        n_test=("set", lambda s: (s == "test").sum()),
        age_missing=("Age", lambda s: s.isna().sum()),
        age_min=("Age", "min"),
        age_median=("Age", "median"),
        age_max=("Age", "max"),
        survival_rate_train=("Survived", "mean"),  # test rows have NaN, so this is train-only
    )
    titles = titles.sort_values("n", ascending=False)
    print(titles.round(2).to_string(na_rep=""))


if __name__ == "__main__":
    main()
