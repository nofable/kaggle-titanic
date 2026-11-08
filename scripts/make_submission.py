"""Step 8: fit the final model on all 891 training rows and write submission.csv.

Final model (JOURNAL.md decision): GradientBoostingClassifier with
learning_rate=0.01, n_estimators=500, max_depth=4, subsample=0.8
(CV 0.8421 +/- 0.0067, stratified 10-fold x 3).

Usage (from the repo root):
    uv run python scripts/make_submission.py
Writes submission.csv (PassengerId, Survived) in the repo root.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from kaggle_titanic.pipeline import build_features, load_raw

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
FINAL_PARAMS = {"learning_rate": 0.01, "n_estimators": 500, "max_depth": 4, "subsample": 0.8}


def main() -> None:
    train, test = load_raw()
    X, y, X_test = build_features(train, test)

    model = GradientBoostingClassifier(random_state=SEED, **FINAL_PARAMS).fit(X, y)
    pred = model.predict(X_test)

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred.astype(int)})
    out = ROOT / "submission.csv"
    submission.to_csv(out, index=False)

    print(f"Wrote {out} ({len(submission)} rows)")
    print(f"Train accuracy (fit on all rows): {model.score(X, y):.4f}")
    print(f"Predicted survival rate on test: {submission['Survived'].mean():.3f} (train: {y.mean():.3f})")

    gender_rule = (test["Sex"] == "female").astype(int)
    print(f"Agreement with gender rule on test: {(submission['Survived'] == gender_rule).mean():.3f}")


if __name__ == "__main__":
    main()
