# Kaggle Titanic — Learning Journal

Competition: https://www.kaggle.com/competitions/titanic
Goal: predict `Survived` (0/1) for the 418 passengers in `test.csv`. Scored on **accuracy**.

## Ground rules

1. **This is a learning project.** Claude coaches; William does the thinking.
   - Claude does **not** hand over answers (feature ideas, model choices, code solutions, "the trick" for this dataset).
   - Claude asks questions, gives hints in escalating levels (nudge → bigger hint → explanation) only when asked, explains concepts, and reviews William's reasoning and code.
   - William can explicitly ask for a bigger hint or an explanation when stuck.
2. Go step by step. Don't skip ahead to modelling before we understand the data.
3. Write things down here as we go: hypotheses, decisions, results, and *why*.
4. ~~Final model is built in **PyTorch**.~~ **Changed 2026-11-08:** final model is **gradient boosting** (scikit-learn). PyTorch dropped: on 891 tabular rows a neural net has no structural advantage over trees, needs more tuning, and would likely only match GB (~0.83). William judged it contrived for this problem.
5. Finish by pushing the project to William's personal GitHub.

## Roadmap

- [x] **0. Setup** — Python env via uv (uv not yet installed; system Python is 3.9), install pandas, numpy, matplotlib, pytorch, jupyter. `git init`, `.gitignore`.
- [x] **1. Data schema** — what each column means, types, missing values, what's in train vs test.
- [x] **2. EDA & hypotheses** — explore, form hypotheses about what drives survival, test them with the data.
- [x] **3. Feature ideas** — brainstorm, then decide which to build and how to handle missing data.
- [x] **4. Validation strategy** — how we'll measure models honestly before touching the leaderboard.
- [x] **5. Baselines & cheap tests** — hypothesise which models will do well, run quick experiments, compare.
- [x] **6. Converge** — pick an approach to take forward and justify it.
- [x] ~~**7. PyTorch model**~~ — dropped (see ground rule 4).
- [x] **8. Submit** — generate `submission.csv`, submit to Kaggle, record score.
- [ ] **9. Ship** — tidy repo, README, push to GitHub.

## Data on disk

`titanic-data/` — `train.csv` (891 rows, has `Survived`), `test.csv` (418 rows, no label), `gender_submission.csv` (example submission format).

---

## Log

### Session 1 — 2026-11-08
- Kicked off. Agreed ground rules and roadmap above.
- Step 1 (schema), first pass done from column names alone. William's type analysis:
  - `PassengerId`: identifier (nominal, no ordering meaning). `Survived`: binary target.
  - `Pclass`: ordinal categorical (1 > 2 > 3). `Sex`: binary categorical. `Embarked`: nominal categorical (3 values).
  - `Age`, `Fare`: continuous, non-negative. `SibSp`, `Parch`: count variables (discrete, non-negative).
  - `Name`, `Ticket`, `Cabin`: free text / semi-structured strings, possibly with categories embedded in them.
  - Missing data: missing because the record is incomplete, or missing because there was no value (e.g. no cabin)? Each case needs different handling.
- Next: check these first impressions against the actual data (needs Python env, Step 0).
- **Observation (from `head`, unverified):** 3rd-class passengers appear not to have cabins. This would mean `Cabin` missingness is structural and tied to `Pclass`. Logged as H7; verify on the full data.
- Step 1b, column profile (`scripts/profile_columns.py`):
  - Missing values: Age (train 177 / 19.9%, test 86 / 20.6%), Cabin (77% / 78%), Fare (1 in test), **Embarked (2 in train)**.
  - Age can't be structurally missing (everyone has an age), so it needs imputing if we use it. Open question: is missing Age random, or related to other columns?
  - Cabin: William suspects not everyone had a cabin. To be checked against history and the H7 numbers.
  - Numeric summaries look similar across train and test, so no big **distribution shift** expected. (Similar means/stds are a weak check.)
  - Fare has high cardinality (248 unique values in train). William expected a few fixed price tiers. Open question: why so many?
- Follow-ups:
  - Missed `Embarked` (2 missing in train) on the first read. Lesson: read the table column by column.
  - **Cabin source (history):** the data comes from an incomplete 1st-class cabin list, plus letters and snippets for the rest. So missing Cabin is mostly *records not surviving*, **not** "no cabin". This rejects William's earlier "not everyone had a cabin" idea. It's a missingness problem (MAR/MNAR), not structural. Open question: who was more likely to have their cabin recorded afterwards?
  - **Parch:** test contains values never seen in train (max 9 in test vs 6 in train). Relevant when choosing an encoding.
  - **Fare:** min is 0, which breaks the "positive" assumption. Why would anyone pay 0? Ticket has 681 unique values for 891 passengers, so some tickets cover several people and Fare may be a *per-ticket* price, not per-person. Possible explanation for Fare's high cardinality.
- Step 2 started: correlation heatmap (`scripts/correlation_grid.py` → `figures/correlation_grid.png`).
  - Survived vs: is_female +0.54, Pclass −0.34, Fare +0.26. PassengerId ≈ 0, so dropped.
  - William's Embarked reasoning: the port shouldn't matter by itself, so any correlation is probably confounded by class. Logged as H8, to be tested by stratifying.
  - Fare–Pclass r = −0.55, so Fare is a confounding *suspect*. Not proven yet: does Fare add information within a class?
  - Correction: Pearson's weakness for Age/SibSp/Parch is **not** about normality. It's that it only captures *linear* relationships, so a non-monotonic relationship to Survived can show r ≈ 0. (SibSp/Parch are also right-skewed counts, not normal.) Plan: binned survival rates or plots instead.
  - Next: test H8 with `scripts/embarked_by_class.py`, then the Age analysis.
  - **H8 result (William):** port differences don't fully disappear within class. Cherbourg survival is higher within each class. Queenstown counts are too small to read in classes 1 and 2.
  - New idea, H9: Cherbourg passengers paid more *within* each class (richer, so better cabins or location), and that explains the port effect. Testing with `scripts/fare_by_port_class.py`.
  - Sex check (`scripts/embarked_by_class_sex.py`): Southampton boarded a lower share of women in every class, which explains part of the port effect. A gap remains within sex × class: 3rd-class women (Cherbourg 0.65, n=23, vs Southampton 0.38, n=88) and men (Cherbourg higher in every class with usable n). Some cells are small, so the size of the gap is uncertain.
  - **Decision: keep Embarked.** Validation will judge whether it actually helps.
- Age (`scripts/age_survival_by_sex.py`, `figures/age_survival_by_sex_*.png`): the near-zero Pearson r hid a strong **non-linear** effect concentrated in young boys. The effect of age depends on sex, so this is an **interaction**.
- William's idea: parse titles out of `Name` to help guess missing Age (and marital status). Parsing script written: `scripts/missing_age_and_titles.py`.
- Age missing by class (train): 1st 13.9%, 2nd 6.0%, 3rd 27.7%. So Age is not MCAR; missingness depends on class (at least MAR).
- Imputation idea (William): fill missing Age with the **median age by class**, instead of the overall median. Candidates to compare: overall median, by class, by title, by class × title. Proposed test: hide some *known* ages, impute them with each method, and measure the error (MAE).
- **Fare is per ticket, not per person.** 216 of 929 tickets are shared (596 passengers), and on 215 of them every passenger shows the same Fare. E.g. ticket 3101295: 7 passengers (6 Panulas + Riihivouri, unrelated) each show 39.69.
- William added `FarePerFamily = Fare / FamilySize` (`fare_per_family` in `src/kaggle_titanic/features.py`), plotted with `scripts/fare_survival_by_sex.py q8 FarePerFamily`. Known limitation: FamilySize ≠ ticket group size for 284 / 1309 passengers (22%). For 200 the ticket group is bigger (friends, servants), for 84 the family is spread over several tickets. Alternative to compare later: Fare / ticket group size.
- William proposed `TicketGroupSize` (passengers sharing a Ticket, counted over train + test) to replace FamilySize. Built `ticket_group_size` / `fare_per_person` in `src/kaggle_titanic/features.py`; plotted with `scripts/family_survival_by_sex.py TicketGroupSize` and `scripts/fare_survival_by_sex.py q8 FarePerPerson`. **Decided** (see Decisions).
- Title groups (`scripts/title_groups.py`): Mr 757, Miss 264, Mrs 198, Master 61, Fancy 29. The test-only title 'Dona' is handled by the mapping.
- Age imputation masking test (`scripts/age_imputation_test.py`, 5-fold × 20 repeats on 714 known ages, medians fit on the non-hidden rows only). MAE in years: class × title 8.90 (±0.48), title 9.41 (±0.51), class 10.46 (±0.59), overall 11.27 (±0.64). To interpret and decide.

### Session 1 wrap-up — 2026-11-08
- Steps 0–6 and 8 done; PyTorch dropped (Step 7). First submission: **LB 0.75837** vs CV 0.8421.
- Tidied: removed unused `torch` dependency, wrote README, first git commit.
- **Next session — pick up here:**
  1. Investigate the CV → LB gap (~8 points; 3 passengers below the gender rule). Options: (b) submit a deliberately simpler model (depth-3 tree / logistic regression) as a diagnostic; (c) inspect the ~51 test passengers where the model overrides the gender rule.
  2. Candidate explanations to weigh: noise (1 passenger = 0.24%), selection optimism from the 54-combo grid, overfitting to small train-only patterns (Deck G, Cherbourg 3rd-class women, FamilySize 5+), train/test mix differences.
  3. Housekeeping: bump `.python-version` from 3.9 to 3.12 (fixes the NotOpenSSL warning and the matmul warnings, and lets the project's `kaggle` CLI (≥1.8) read `~/.kaggle/access_token`). Until then, submit with `uvx --python 3.12 --with six kaggle@2.2.5 competitions submit -c titanic -f submission.csv -m "..."`.
  4. Step 9: push to personal GitHub.

## Checkpoint — end of Step 4 (2026-11-08)

**Where we are:** Steps 0–4 done (setup, schema, EDA, features, validation). No model trained yet.

**What drives survival (from EDA):**
- Sex dominates (gender rule alone: 0.787).
- Class: strong, and confounds almost everything else (fare, port, cabin, family size).
- Age: matters non-linearly, mainly for **boys**; little effect for women. An Age × Sex interaction.
- Family size: non-monotonic. Fine up to 4, then a cliff at 5+ (mostly large 3rd-class families).
- Per-person fare: some signal beyond class; raw Fare was distorted by shared tickets.
- Cabin recorded: predicts survival for men (survivorship bias in the records).
- Embarked: a small Cherbourg effect left after controlling for class and sex.

**Data problems handled:** Age 20% missing (not random) → class × title median + AgeMissing flag. Cabin 77% missing (MNAR) → CabinKnown + Deck. Fare is per ticket → FarePerPerson. Embarked (2) and Fare (1) gaps filled.

**Model input:** 20 columns (see the Step 3 checklist).

**Validation:** stratified 10-fold × 3 repeats. Baselines: 0.616 (majority), 0.787 (gender rule).

**Big-picture realities:**
- Headroom is small: good scores are ~0.80–0.85, so we're fighting for 2–6 points over the gender rule.
- The remaining errors are the *exceptions*: men who survived and women who died. Every feature has to help find those.
- 891 rows is small. Simple models may match complex ones, and noise can look like improvement, so trust CV ± standard error, not single numbers.
- The Kaggle leaderboard is scored on part of the 418 test rows, so one passenger is worth roughly 0.2–0.5% there. Don't chase it.
- ~~Still to build: the preprocessing pipeline~~ → built: `src/kaggle_titanic/pipeline.py` (`build_features(train, test)` → X_train 891×20, y, X_test 418×20, no NaNs).

**Methods learned:** correlation's limits (linear only), stratifying to find confounders, composition effects / Simpson's paradox, small-n caution, forking paths, missingness mechanisms, masking tests, leakage vs transductive use, encoding and scaling choices, CV design.

## Step 3 — Feature encoding checklist

| # | Feature | Type | Done so far | Open decisions |
|---|---------|------|-------------|----------------|
| 1 | Sex | binary | `is_female` 0/1, one column ✅ | — |
| 2 | Pclass | ordinal (3) | One-hot ✅ | — |
| 3 | Age | continuous, 0.42–80 | Imputed (class × title) ✅, `AgeMissing` flag ✅, no log (roughly symmetric) ✅, standardise ✅ | — |
| 4 | AgeMissing | binary | ✅ | — |
| 5 | TitleGroup | nominal (5) | **Dropped as a model input** ✅ (still computed for AgeImputer) | Step 5: test adding it back |
| 6 | FamilySize | count, 1–11 | Raw count ✅, no log ✅, standardise ✅ | — |
| 7 | FarePerPerson | continuous, skewed, 0–128 | Built ✅, 1 test NaN imputed ✅, raw (no log) ✅, standardise ✅ | Step 5: test log1p |
| 8 | Embarked | nominal (3) | 2 NaNs imputed ✅, one-hot (3 columns) ✅ | — |
| 9 | CabinKnown | binary | ✅ | — |
| 10 | Deck | nominal (A–G, U) | Built ✅, T merged into U ✅, small decks kept ✅, one-hot (8 columns) ✅, keep alongside CabinKnown ✅ | — |
| — | All scaled features | | Standardisation (z-score) ✅, fitted on train + test features ✅ | — |

## Glossary (terms picked up along the way)

- **Target / label**: the column we predict (`Survived`). **Features**: the inputs.
- **Binary classification**: predicting one of two classes.
- **Nominal vs ordinal categorical**: no order (`Embarked`) vs meaningful order (`Pclass`).
- **Count variable**: discrete non-negative integer (`SibSp`, `Parch`). **Continuous**: `Age`, `Fare`.
- **Cardinality**: number of distinct values. `Name` and `Ticket` are *high cardinality*.
- **Semi-structured**: strings with hidden structure inside (`Ticket`, `Cabin`).
- **Feature engineering / extraction**: deriving new features from raw columns (e.g. name length).
- **Missingness mechanisms**: MCAR (missing completely at random), MAR (missing at random, given other observed columns), MNAR (missing not at random, related to the missing value itself). Also **structural missingness**: the value doesn't exist (no cabin).
- **Imputation**: filling in missing values. **Missingness indicator**: an extra 0/1 feature for "this was missing".
- **Pairwise deletion**: compute each pairwise statistic using only the rows where both values are present.
- **One-hot encoding**: one 0/1 column per category (used for nominal features).
- **Point-biserial / phi**: Pearson correlation when one or both variables are binary.
- **Confounding**: a third variable (e.g. `Pclass`) drives both the feature and the outcome, creating a correlation between them that isn't causal.
- **Stratify / condition on**: analyse within each level of a variable (e.g. within each class) to control for it.
- **Collinearity / multicollinearity**: features strongly correlated with each other (e.g. Fare and Pclass). Not the same thing as confounding.
- **Spearman correlation**: correlation on ranks. Captures any *monotonic* relationship, but still misses non-monotonic ones.
- **Garden of forking paths / multiple comparisons**: slicing data many ways until something looks significant. Small cells plus many slices lead to false findings.
- **Fisher's exact test / confidence interval for a proportion**: ways to check whether a gap between two small groups is bigger than noise.
- **Interaction effect**: one feature's effect depends on the value of another (e.g. Age matters differently for males vs females).
- **Conditional (group-wise) imputation**: fill missing values using a statistic from similar rows (e.g. median Age within the same class).
- **Data leakage**: information the model shouldn't have at training time (e.g. imputation statistics computed on validation/test rows) gets into training and inflates scores.
- **Masking / simulated missingness**: hide known values, impute them, score against the truth, to evaluate an imputation method.
- **Back-off**: when a fine-grained group is empty, fall back to a coarser group (class × title → title → overall).
- **MAE (mean absolute error)**: average of |predicted − actual|.
- **Composition effect**: a group's average changes because its *mix* changes (e.g. more boys among 'males' in bigger families), not because the effect within each subgroup changes. Related: **Simpson's paradox**.
- **Target leakage**: using information derived from the label (`Survived`) that wouldn't be available at prediction time, or that comes from the rows being validated. Feature-only counts (e.g. ticket group size over train + test) are not target leakage.
- **Transductive**: using unlabelled test *features* when building features or statistics. Doesn't leak labels, but can blur how honest validation is.
- **Survivorship bias**: data recorded mainly for those who survived, so whether data exists correlates with the outcome.
- **Min-max scaling**: (x − min) / (max − min), maps to [0, 1]. Sensitive to outliers.
- **Standardisation (z-score)**: (x − mean) / std, giving mean 0 and std 1. The usual default for neural networks.
- **Redundant / perfectly collinear column**: a column fully determined by others (e.g. Sex_male = 1 − Sex_female). **Dummy variable trap**: this redundancy breaks unregularised linear/logistic regression; harmless for NNs.
- **Bias (neuron)**: the constant term in a neuron, which sets its baseline output when the inputs are 0.
- **Overfitting / memorisation**: the model learns noise specific to the training rows (e.g. a category with 4 examples) instead of patterns that generalise.
- **Majority-class baseline**: predict the most common label for everyone. Its accuracy is the share of that label; any real model must beat it.
- **Cross-validation (k-fold)**: split the data into k parts; train on k−1 and validate on the remaining part, k times; average the scores.
- **Stratified k-fold**: each fold keeps the same class balance (e.g. survival rate) as the full data.
- **Repeated CV**: re-run k-fold with different random fold assignments and average. It reduces noise in the *estimate*; it doesn't train the model more.
- **Standard error of the mean**: std / √n. How much an averaged score would wobble if you re-ran it.
- **Epoch**: one full pass through the training data. **Early stopping**: stop training when validation loss stops improving (Step 7).
- **Generalisation gap**: train accuracy − validation accuracy. A big gap means overfitting.
- **Bagging (bootstrap aggregating)**: train many models on random resamples of the data and average them. Random forests also pick random features at each split. Each tree overfits differently, so averaging cancels the noise and keeps the shared signal.
- **Bias vs variance**: bias = consistently wrong because the model is too simple; variance = predictions swing with the exact training rows because the model is too flexible. Forests reduce variance; deeper trees and boosting reduce bias.
- **Distribution shift (covariate shift)**: train and test inputs come from different distributions.
- **Skew**: an asymmetric distribution. Mean much bigger than median usually means right skew (a long tail of big values).
- **Monotonic vs non-monotonic relationship**: always goes one way vs can change direction.
- **Priors / domain knowledge**: what we believe before looking at the data.

## Hypotheses

| # | Hypothesis | How to test | Result |
|---|------------|-------------|--------|
| H1 | Women more likely to survive ("women and children first") | Correlation grid | **Supported**: is_female r = +0.54, the strongest single signal |
| H2 | Children more likely to survive | Survival by age bin × sex (`scripts/age_survival_by_sex.py`, 10y and 5y bins) | **Supported for boys, not clearly for girls.** Boys under ~10 survive far more than adult men (0–4: 0.65 vs ~0.1–0.25). Girls are about level with or below adult women. Sex × Age **interaction** |
| H10 | Men aged 25–50 survived more than men under 25 or over 50 | Class mix + survival within class by male age band (`scripts/male_age_by_class.py`) | **Partly explained by class**: 25% of men aged 25–49 were 1st class vs 7% of men aged 15–24. Men 50+ are 56% 1st class yet survive poorly (0.13), against what class would predict. Possible causes (speculation, untestable here): frailty in cold water, a sense of duty. Within-class check still to read |
| H3 | Higher class (lower `Pclass`) more likely to survive | Correlation grid | **Supported**: Pclass r = −0.34 (lower number = better class = more survival) |
| H4 | Family size affects survival, direction unknown (support vs slowed down coordinating) | Survival by FamilySize (SibSp+Parch+1) × sex (`scripts/family_survival_by_sex.py`) | **Non-monotonic.** Men: survival rises from 1 to 4 (0.16 → 0.50). Women: stable at ~0.8 up to 4, then drops sharply at 5+ (small n). Confounder checks (`scripts/family_confounders.py`): the male rise is mostly **composition**. Larger families have a higher share of boys (Master), while adult men (Mr) stay flat at ~0.14–0.21 for sizes 1–4. Families of 5+ are 77–100% 3rd class, against 31–46% for sizes 3–4, so the cliff is heavily confounded with class (small n). Boys in families of 5+ mostly died (e.g. 0/9 at size 6) |
| H5 | Longer names mean posher passengers, so may correlate with `Pclass` / survival | | |
| H6 | Whether a value is missing (e.g. `Cabin`) may itself carry signal | Survival with vs without Cabin, by class and class × sex (`scripts/cabin_checks.py`) | **Supported, mainly for men.** 1st-class men: 0.41 with cabin (n=95) vs 0.22 without (n=27). Women are near ceiling either way. 2nd/3rd class: big gaps but n=6 per cell. William predicted no correlation. Likely explanation: **survivorship bias** in how cabins were recorded (MNAR). Deck letter: almost all 1st class. William notes deck A (0.47, n=15) and C (0.59, n=59) are lower than B/D/E. Unchecked for sex mix |
| H8 | `Embarked` has no direct effect; any correlation comes through `Pclass` (unless the port decided where on the ship you stayed) | Survival by Embarked **within each Pclass** (`scripts/embarked_by_class.py`) | **Not fully supported**: Cherbourg survives better within each class. Small n for Queenstown in classes 1 and 2 |
| H9 | Cherbourg's advantage is explained by paying more within each class | Mean/median Fare by Pclass × Port (`scripts/fare_by_port_class.py`) | **Partly supported**: 1st-class Cherbourg passengers paid more (median 78 vs 52 at Southampton). Check whether it holds in 3rd class |
| H11 | Title in Name (Mr/Mrs/Miss/...) indicates marital status and can help guess missing Age | Parse titles (`scripts/missing_age_and_titles.py`) | Parsed for all 1,309 names; 18 distinct titles, many with n = 1. To interpret |
| H12 | Missing Age is not random | Class mix + survival, missing vs known Age (`scripts/missing_age_and_titles.py`) | Output produced; to interpret |
| H7 | 3rd-class passengers have no `Cabin` (missingness is structural, driven by `Pclass`) | `scripts/check_cabins_by_class.py` | **Partly true.** Strong association, not deterministic. Cabin recorded (combined): 1st 256/323 (79%), 2nd 23/277 (8%), 3rd 16/709 (2%). Train and test look similar. Pinned. |

## Decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-11-08 | Drop `PassengerId` as a feature | It's an identifier; r with Survived = −0.005 |
| 2026-11-08 | Keep `Age` as a feature | Strong non-linear effect (young boys) and Age × Sex interaction; older men survive less than their class mix predicts |
| 2026-11-08 | Collapse titles to Mr / Mrs / Miss / Master / Fancy. Mlle→Miss, Mme→Mrs, Ms→Miss; nobility, Dr (including the one female Dr), Rev and military → Fancy; unseen titles → Fancy | Rare titles (n = 1–8) are too small to learn from. Mapping in `src/kaggle_titanic/features.py` |
| 2026-11-08 | Impute missing Age with the median by Pclass × TitleGroup (back off to title, then overall). Fit on training rows only | Lowest MAE in the masking test (8.90 vs 9.41 title-only, 11.27 overall). Implemented as `AgeImputer` in `src/kaggle_titanic/features.py` |
| 2026-11-08 | Add `AgeMissing` 0/1 flag (computed **before** imputing) | Missing-Age passengers survived at 0.29 vs 0.41, and are mostly 3rd class. Imputation would hide that. Validate its value in Step 5 |
| 2026-11-08 | Use `FamilySize` (SibSp + Parch + 1) instead of SibSp and Parch separately | SibSp and Parch plots tell the same story (stable or rising to a point, then a cliff at large counts). Keep as **raw count** (cheap, keeps all the information; a neural net can learn the cliff at 5+). Note: the 'male rise to 4' is a composition effect (boys), not a reason. Possible Step 5 test: add a large-family (≥5) flag or bands |
| 2026-11-08 | Use `FarePerPerson` = Fare / ticket group size (counted over train + test) in place of raw Fare | Fare is a per-ticket price. Dividing by the people on the ticket gives a true per-person price and removes the group-size distortion. Cleaner survival pattern than FarePerFamily. Not leakage (features only). `fare_per_person()` in `src/kaggle_titanic/features.py`. Note: 1 test row has missing Fare and needs imputing |
| 2026-11-08 | Keep `FamilySize` as the group feature; TicketGroupSize used only as the fare divisor for now | Family and travel group carry different signals. Could test TicketGroupSize as a feature in Step 5 |
| 2026-11-08 | Impute missing FarePerPerson with the median for Pclass × Embarked (fit on train; back off to Pclass, then overall) | Only case: test passenger 1044 (3rd class, Southampton, alone on ticket 3701, no relatives aboard) → 7.85. Per-person median used because he travelled alone. `FarePerPersonImputer` in `src/kaggle_titanic/features.py` |
| 2026-11-08 | Add `CabinKnown` 0/1 flag | Strong for men within class (survivorship bias in records); missing rate is similar in train/test (77% / 78%). `cabin_known()` |
| 2026-11-08 | Add `Deck` letter (first char of Cabin; 'U' = unknown) | William saw lower survival on decks A and C. Accepted without the sex-mix check (A has n = 15). Almost all decks are 1st class. `deck()` |
| 2026-11-08 | Impute missing Embarked: within the passenger's class, pick the port whose median FarePerPerson is closest to theirs (fit on train) | 2 cases: Icard & Stone (1st class, shared ticket 113572, FarePerPerson 40.0, cabin B28) → C (1st-class medians C 35.48, Q 30.00 (n = 2), S 26.55). `EmbarkedImputer` |
| 2026-11-08 | One-hot encode `Pclass` | William's choice. Trade-off: drops the built-in ordering (1 > 2 > 3), but with 3 levels it's cheap and the network can learn any per-class effect |
| 2026-11-08 | Encode Sex as a single `is_female` 0/1 column | Binary features → one column (a second one-hot column is redundant). One-hot is for 3+ categories. William first reasoned that 'zeros are hard for NNs'; corrected: the bias handles the baseline, and one-hot has zeros too |
| 2026-11-08 | Age: no log transform | Age is roughly symmetric (mean 29.7, median 28). Log would stretch children but squash adult differences; the NN can learn curved effects from raw values |
| 2026-11-08 | Scale continuous features with standardisation (z-score) | NN default; less distorted by outliers than min-max |
| 2026-11-08 | **Pipeline rule:** steps that use only features (imputers, scaler, ticket counts) are fitted on **train + test features**. Anything that uses `Survived` is fitted on training rows/folds only | Uses all the feature information available (transductive); no label leakage. Small blur in CV honesty, accepted. Supersedes the earlier "imputers fit on train only" (AgeImputer, FarePerPersonImputer, EmbarkedImputer now fitted on train + test). No code change: the pipeline passes the combined frame to `fit()` |
| 2026-11-08 | Drop TitleGroup as a model input; keep it for Age imputation | Sex covers Mr vs Mrs/Miss; Age covers Master vs Mr, including missing ages via the class × title imputer. Lost: Miss vs Mrs (0.70 vs 0.79) and Fancy (n = 29). Test adding it back in Step 5 |
| 2026-11-08 | Deck: merge 'T' into 'U'; keep small decks (A, F, G) as their own categories; one-hot; keep alongside CabinKnown | T has 1 passenger (train only). Redundancy of Deck_U with CabinKnown is harmless for a NN. Small decks risk memorisation (overfitting), but grouping was judged a bigger loss of resolution. Side effect: the T passenger has Deck = U but CabinKnown = 1 |
| 2026-11-08 | Validation: **stratified 10-fold CV, repeated 3×** (30 runs per experiment), using scikit-learn's `RepeatedStratifiedKFold`. Report mean accuracy ± standard error | A single split on 891 rows is noisy. k = 10 because training is cheap. Stratified so each fold keeps ~38% survivors. Repeats reduce noise in the estimate; increase them if the standard error is bigger than the differences being compared |
| 2026-11-08 | **Final model: gradient boosting. Drop PyTorch** | Best CV (0.8354 ± 0.0070, defaults) vs LR 0.8073, RF 0.8103. A NN on small tabular data has no structural advantage and would mostly reproduce GB; learning PyTorch isn't worth a contrived use here |
| 2026-11-08 | Final settings: GB lr 0.01, 500 trees, depth 4, subsample 0.8 (top of grid) | William: highest CV; its train–CV gap and SE are similar to the simpler alternatives, so there's no compelling reason to step down. Accepts that its CV score is somewhat optimistic |
| 2026-11-08 | Keep `Embarked` as a feature | Cherbourg effect remains after stratifying by class and by sex × class (partly explained by sex mix and fare). Cheap to include; revisit with validation in Step 5 |

## Experiments

| # | Model / features | Validation score | Notes |
|---|------------------|------------------|-------|
| 0a | Majority class (everyone dies) | 0.6162 (train) | Floor. No fitting, so CV would give the same |
| 0b | Gender rule (women survive, men die) | 0.7868 (train) | = `gender_submission.csv`. The real bar to beat |

| 1 | Logistic regression (20 features) | 0.8073 ± 0.0063 | train 0.8165 |
| 2 | Decision tree, depth 3 | 0.8070 ± 0.0080 | train 0.8257 |
| 3 | Decision tree, depth 5 | 0.8216 ± 0.0064 | train 0.8642 |
| 4 | Decision tree, unlimited | 0.7868 ± 0.0071 | train 0.9863 |
| 5 | Random forest, 300 trees (defaults) | 0.8103 ± 0.0066 | train 0.9863 |
| 6 | Gradient boosting (defaults) | 0.8354 ± 0.0070 | train 0.9067 |

| 7 | GB grid search: 54 combos (lr 0.01/0.05/0.1 × trees 100/300/500 × depth 2/3/4 × subsample 0.8/1.0) | best 0.8421 ± 0.0067 (lr 0.01, 500 trees, depth 4, subsample 0.8; train 0.9124) | `scripts/tune_gb.py`, full table in `results/gb_grid.csv`. +0.7 pt over default, about 1 SE. 4 combos within 1 SE of best. Best score is optimistic (selection bias) |
| 8 | **Kaggle submission**: tuned GB (exp 7 settings) fit on all 891 rows | **Public LB 0.75837** (317/418 correct) | CV said 0.8421. ~8-point drop. Below the gender rule's widely reported LB of 0.76555 (320/418), i.e. 3 passengers worse. Submitted 2026-11-08 |

Experiments 1–6: `scripts/cheap_models.py`, stratified 10-fold × 3, untuned. ± = standard error over 30 folds (understated, since folds overlap).

**Step 5 predictions (William, before running):** logistic regression 0.79 · decision tree 0.80 · random forest 0.80 · gradient boosting 0.82 (winner).


## Open questions / parking lot

- **Step 5 tests to run:** add TitleGroup back; log1p(FarePerPerson) vs raw (raw z-scores reach +8.4; log turns zero fares into −3.7 outliers); FamilySize bands (one-hot) vs raw. Boundaries must not overlap and should follow the data: the cliff is between 4 and 5, so don't bin 4 with 5+; TicketGroupSize as a feature; AgeMissing flag on/off; Embarked on/off.


- **Cabin (from H7):** 67 1st-class passengers have no cabin, and 39 2nd/3rd-class passengers have one. Lost records or real? Does a missing Cabin add information beyond `Pclass` (e.g. does survival differ for 1st class with vs without a cabin)? Can we use the cabin letter? Revisit in Step 3.
