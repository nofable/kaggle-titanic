"""Step 5 tuning: grid search for GradientBoostingClassifier.

Every combination is scored with the agreed CV (stratified 10-fold x 3 repeats)
on the 20-column feature matrix. Prints the top 15 by CV mean, with standard
error and train accuracy, and the full results go to results/gb_grid.csv.

Caveat: picking the best of many settings makes its CV score optimistic
(selection bias). Prefer a simpler setting within ~1 SE of the best.

Usage (from the repo root):
    uv run python scripts/tune_gb.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from kaggle_titanic.pipeline import build_features, load_raw

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_FOLDS = 30  # 10 folds x 3 repeats
GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0],
}


def main() -> None:
    X, y, _ = build_features(*load_raw())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
    search = GridSearchCV(
        GradientBoostingClassifier(random_state=SEED),
        GRID,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
    search.fit(X, y)

    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame(
        {
            "learning_rate": res["param_learning_rate"],
            "n_estimators": res["param_n_estimators"],
            "max_depth": res["param_max_depth"],
            "subsample": res["param_subsample"],
            "cv_mean": res["mean_test_score"],
            "cv_se": res["std_test_score"] / np.sqrt(N_FOLDS),
            "train_acc": res["mean_train_score"],
        }
    ).sort_values("cv_mean", ascending=False)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "gb_grid.csv", index=False)

    print(f"{len(table)} combinations x {N_FOLDS} folds. Default GB was 0.8354.\n")
    print("Top 15:")
    print(table.head(15).round(4).to_string(index=False))

    best = table.iloc[0]
    within = table[table["cv_mean"] >= best["cv_mean"] - best["cv_se"]]
    print(f"\n{len(within)} combinations are within 1 SE of the best "
          f"({best['cv_mean']:.4f} - {best['cv_se']:.4f} = {best['cv_mean'] - best['cv_se']:.4f}).")
    print("Saved results/gb_grid.csv")


if __name__ == "__main__":
    main()
