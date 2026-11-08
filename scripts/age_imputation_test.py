"""Masking test for Age imputation methods.

Among train.csv passengers with a known Age, hide one fold of ages at a time
(K-fold, repeated with different shuffles), compute medians from the
*remaining* rows only, fill in the hidden ages, and score with MAE (years).

Methods:
  overall        - one median for everyone
  class          - median by Pclass
  title          - median by TitleGroup (Mr/Mrs/Miss/Master/Fancy)
  class x title  - median by Pclass x TitleGroup; empty cells fall back to
                   the title median, then the overall median

Usage (from the repo root):
    uv run python scripts/age_imputation_test.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_titanic.features import title_group

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"
K = 5
REPEATS = 20
METHODS = {
    "overall": [],
    "class": ["Pclass"],
    "title": ["TitleGroup"],
    "class x title": ["Pclass", "TitleGroup"],
}


def impute(fit: pd.DataFrame, apply: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Predict Age for `apply` rows using medians computed on `fit` rows only."""
    overall = fit["Age"].median()
    if not keys:
        return pd.Series(overall, index=apply.index)
    medians = fit.groupby(keys)["Age"].median()
    pred = apply.join(medians.rename("pred"), on=keys)["pred"]
    if "TitleGroup" in keys and len(keys) > 1:  # back off to title-only, then overall
        title_med = fit.groupby("TitleGroup")["Age"].median()
        pred = pred.fillna(apply["TitleGroup"].map(title_med))
    return pred.fillna(overall)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["TitleGroup"] = title_group(df["Name"])
    known = df[df["Age"].notna()].reset_index(drop=True)

    scores = {name: [] for name in METHODS}
    for r in range(REPEATS):
        folds = np.random.default_rng(r).permutation(len(known)) % K
        for k in range(K):
            fit, hidden = known[folds != k], known[folds == k]
            for name, keys in METHODS.items():
                pred = impute(fit, hidden, keys)
                scores[name].append((pred - hidden["Age"]).abs().mean())

    print(f"{len(known)} passengers with known Age; {K}-fold x {REPEATS} repeats "
          f"= {K * REPEATS} scores per method.\n")
    result = pd.DataFrame(
        {name: {"MAE_mean": np.mean(s), "MAE_std": np.std(s)} for name, s in scores.items()}
    ).T.sort_values("MAE_mean")
    print(result.round(2).to_string())


if __name__ == "__main__":
    main()
