# Kaggle Titanic

A learning project working through the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) step by step: data schema → exploration and hypotheses → feature engineering → validation design → model comparison → submission.

The full reasoning is in **[JOURNAL.md](JOURNAL.md)**: every hypothesis, decision (with its reason), experiment and lesson learned.

## Results

| Model | CV accuracy (stratified 10-fold × 3) | Kaggle public LB |
|---|---|---|
| Majority class (everyone dies) | 0.616 | — |
| Gender rule (women survive) | 0.787 | 0.766 (widely reported) |
| Logistic regression | 0.807 | — |
| Random forest (defaults) | 0.810 | — |
| Gradient boosting (defaults) | 0.835 | — |
| **Gradient boosting (tuned)** | **0.842** | **0.758** |

The tuned model beat the gender rule by ~5.5 points in cross-validation but scored slightly *below* it on the leaderboard (3 passengers out of 418). Understanding that gap is the next step. See the journal.

## Features (20 model inputs)

`is_female`, `Pclass` (one-hot), `Age` (imputed by class × title median, standardised), `AgeMissing`, `FamilySize` (SibSp + Parch + 1, standardised), `FarePerPerson` (Fare ÷ passengers on the same ticket, standardised), `Embarked` (one-hot), `CabinKnown`, `Deck` (one-hot).

## Layout

```
src/kaggle_titanic/
  features.py    feature helpers and imputers
  pipeline.py    build_features(train, test) -> X_train, y, X_test
scripts/         exploration, model comparison, tuning, submission
figures/         plots from exploration
results/         grid search results
JOURNAL.md       the learning journal
```

## Running it

Requires [uv](https://docs.astral.sh/uv/). Download the competition data into `titanic-data/` (not committed, per Kaggle's rules), then:

```
uv sync
uv run python scripts/cheap_models.py      # compare models with CV
uv run python scripts/tune_gb.py           # grid search gradient boosting
uv run python scripts/make_submission.py   # write submission.csv
```
