"""Step 5: compare cheap scikit-learn models with the agreed CV.

Validation: stratified 10-fold, repeated 3x (30 fits per model), on the
20-column feature matrix from kaggle_titanic.pipeline.
All models use default / lightly chosen settings (no tuning yet).

Reports per model:
  cv_mean   - mean validation accuracy over 30 folds
  cv_std    - spread of the 30 fold scores
  cv_se     - standard error of the mean, cv_std / sqrt(30)
              (folds share data, so this somewhat understates true uncertainty)
  train_acc - mean accuracy on the training part of each fold
              (a big gap to cv_mean = overfitting)

Usage (from the repo root):
    uv run python scripts/cheap_models.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from kaggle_titanic.pipeline import build_features, load_raw

# numpy 2.x + macOS Accelerate emits spurious "overflow/divide by zero in matmul"
# warnings inside sklearn's LogisticRegression. Results were cross-checked with a
# different solver (liblinear) and are identical, so silence them.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

SEED = 42
MODELS = {
    "logistic regression": LogisticRegression(max_iter=1000),
    "decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "decision tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "decision tree (unlimited)": DecisionTreeClassifier(random_state=SEED),
    "random forest (300 trees)": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "gradient boosting (default)": GradientBoostingClassifier(random_state=SEED),
}


def main() -> None:
    X, y, _ = build_features(*load_raw())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    rows = []
    for name, model in MODELS.items():
        res = cross_validate(model, X, y, cv=cv, scoring="accuracy", return_train_score=True)
        scores = res["test_score"]
        rows.append(
            {
                "model": name,
                "cv_mean": scores.mean(),
                "cv_std": scores.std(),
                "cv_se": scores.std() / np.sqrt(len(scores)),
                "train_acc": res["train_score"].mean(),
            }
        )

    table = pd.DataFrame(rows).set_index("model")
    print("Baselines (train): majority 0.6162, gender rule 0.7868\n")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
