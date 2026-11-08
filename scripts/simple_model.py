"""Diagnostic: a deliberately simple model using only Sex, Age and Pclass.

Columns used from the pipeline: is_female, Pclass_1/2/3, Age (imputed by
class x title median, standardised). Same CV as everything else
(stratified 10-fold x 3). Writes one submission file per model to
submissions/ so any of them can be uploaded for comparison.

Extra pipeline columns can be added as arguments, e.g. CabinKnown.

Usage (from the repo root):
    uv run python scripts/simple_model.py [EXTRA_COLUMN ...]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from kaggle_titanic.pipeline import build_features, load_raw

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
COLUMNS = ["is_female", "Pclass_1", "Pclass_2", "Pclass_3", "Age"]
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "tree_depth5": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "gb_default": GradientBoostingClassifier(random_state=SEED),
    "gb_tuned": GradientBoostingClassifier(
        learning_rate=0.01, n_estimators=500, max_depth=4, subsample=0.8, random_state=SEED
    ),
}


def main() -> None:
    train, test = load_raw()
    X_all, y, X_test_all = build_features(train, test)
    extra = sys.argv[1:]
    cols = COLUMNS + extra
    tag = "simple" + "".join(f"_{c}" for c in extra)
    X, X_test = X_all[cols], X_test_all[cols]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
    gender_rule = (test["Sex"] == "female").astype(int).to_numpy()

    out_dir = ROOT / "submissions"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for name, model in MODELS.items():
        res = cross_validate(model, X, y, cv=cv, scoring="accuracy", return_train_score=True)
        pred = model.fit(X, y).predict(X_test).astype(int)
        pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred}).to_csv(
            out_dir / f"{tag}_{name}.csv", index=False
        )
        rows.append(
            {
                "model": name,
                "cv_mean": res["test_score"].mean(),
                "cv_se": res["test_score"].std() / np.sqrt(len(res["test_score"])),
                "train_acc": res["train_score"].mean(),
                "test_overrides_vs_gender_rule": int((pred != gender_rule).sum()),
            }
        )

    print(f"Features: {cols}")
    print("Reference: gender rule CV 0.7868 / LB 0.76555; full 20-feature tuned GB CV 0.8421 / LB 0.75837\n")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())
    print(f"\nSubmission files written to {out_dir}/{tag}_<model>.csv")


if __name__ == "__main__":
    main()
