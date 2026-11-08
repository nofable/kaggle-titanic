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
| Gradient boosting (tuned, 20 features) | 0.842 | 0.758 |
| **Decision tree, depth 3 (sex + age + pclass only)** | **0.803** | **0.761** |

## What I learned

- **CV ≠ leaderboard.** The tuned 20-feature model beat the gender rule by ~5.5 points in CV, but every model, including the gender rule itself, lost 2–8 points on the test set. All submissions landed within 3 passengers (of 418) of each other.
- **Accuracy only rewards crossing 50%.** Class strongly changes a man's survival *probability* (35% in 1st class vs 11% otherwise) but never past 50%, so it changes no predictions. The only men a model can profitably flip are young boys in 1st/2nd class.
- **Much of the outcome is irreducible noise.** For example, 3rd-class women split 72/72.
- **A feature that's strong on its own can add nothing given the others** (CabinKnown, once Sex and Pclass are in).
- **The missed unlock:** group survival (the "woman–child group" idea), using what happened to the rest of a passenger's family/ticket group. It needs label-based features computed strictly inside CV folds to avoid leakage.

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
submissions/     alternative submission files
JOURNAL.md       the learning journal
```

## Running it

Requires [uv](https://docs.astral.sh/uv/). Download the competition data into `titanic-data/` (not committed, per Kaggle's rules), then:

```
uv sync
uv run python scripts/cheap_models.py      # compare models with CV
uv run python scripts/tune_gb.py           # grid search gradient boosting
uv run python scripts/make_submission.py   # write submission.csv
uv run python scripts/simple_model.py      # sex + age + pclass models -> submissions/
```
