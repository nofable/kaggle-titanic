"""Step 4 baselines: accuracy on train.csv of two rules that ignore (almost) everything.

  0a. Majority class - predict the most common outcome for every passenger.
  0b. Gender rule    - women survive, men die (the rule behind gender_submission.csv).

Usage (from the repo root):
    uv run python scripts/baselines.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    y = train["Survived"]

    majority = y.mode()[0]  # most common label
    pred_majority = pd.Series(majority, index=train.index)
    acc_majority = (pred_majority == y).mean()

    pred_gender = (train["Sex"] == "female").astype(int)
    acc_gender = (pred_gender == y).mean()

    print(f"0a. Majority class (predict {majority} for everyone): accuracy = {acc_majority:.4f}")
    print(f"0b. Gender rule (female = 1, male = 0):          accuracy = {acc_gender:.4f}")

    # Sanity check: the gender rule is what Kaggle's example submission contains.
    test = pd.read_csv(DATA_DIR / "test.csv")
    example = pd.read_csv(DATA_DIR / "gender_submission.csv")
    rule_on_test = (test["Sex"] == "female").astype(int)
    matches = (example.set_index("PassengerId")["Survived"] == rule_on_test.set_axis(test["PassengerId"])).all()
    print(f"\ngender_submission.csv matches the gender rule on test: {matches}")


if __name__ == "__main__":
    main()
