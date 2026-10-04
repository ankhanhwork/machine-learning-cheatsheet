# Theory Guide: Machine Learning Pipeline for Binary Classification

> This guide explains `Machine Learning Pipeline for Binary Target.ipynb` in workflow order. The notebook is a reusable practice template: some cells are defaults, while others are optional examples that require replacing placeholder column names or uncommenting code. Parameter values below describe the notebook configuration; they are not guaranteed to be optimal for a new dataset.

## Contents

1. [The full workflow](#1-the-full-workflow)
2. [Libraries and reproducibility](#2-libraries-and-reproducibility)
3. [Loading and inspecting data](#3-loading-and-inspecting-data)
4. [Defining the target, X, and y](#4-defining-the-target-x-and-y)
5. [Quality checks, imbalance, outliers, and PSI](#5-quality-checks-imbalance-outliers-and-psi)
6. [Data leakage](#6-data-leakage)
7. [Optional feature engineering](#7-optional-feature-engineering)
8. [Splitting and validation](#8-splitting-and-validation)
9. [Preprocessing and pipelines](#9-preprocessing-and-pipelines)
10. [Classification algorithms](#10-classification-algorithms)
11. [Evaluation metrics and threshold selection](#11-evaluation-metrics-and-threshold-selection)
12. [Tuning and cross-validation](#12-tuning-and-cross-validation)
13. [Model interpretation](#13-model-interpretation)
14. [Final fit, predictions, and submission](#14-final-fit-predictions-and-submission)
15. [Report outline](#15-report-outline)
16. [Before using the notebook on real data](#16-before-using-the-notebook-on-real-data)

## 1. The full workflow

This is a **binary classification** problem: use customer or loan information to predict whether an observation belongs to class 0 or class 1. The example context is bad debt: `1 = bad debt`, `0 = no bad debt`. The goal is not just to train an estimator; it is to build a **pipeline** that combines preparation, model fitting, evaluation, interpretation, and prediction output.

```text
Labelled data + rows to score
             ↓
Inspect structure, quality, labels, leakage, and distribution shift
             ↓
Create/select features and choose a suitable validation split
             ↓
Preprocessing pipeline → classifier
             ↓
Compare with AP, F1, and confusion matrix; choose a threshold
             ↓
Tune/cross-validate/interpret
             ↓
Refit on all labelled rows → predict test rows → submission/report
```

A pipeline applies the same transformations during fitting and prediction. It also learns imputation values, scaling parameters, and category mappings from each training fold only. This helps prevent **data leakage** during preprocessing.

## 2. Libraries and reproducibility

- `pandas`, `numpy`: tabular data and numerical operations, including missing/infinite values.
- `matplotlib`: PR and confusion-matrix plots.
- `scikit-learn`: data splitting, preprocessing, models, metrics, tuning, and permutation importance.
- `xgboost` and `shap`: optional packages; they must be installed with compatible versions.
- `RANDOM_STATE = 42`: fixes randomness in selected operations so runs are reproducible in the same environment. A seed does not make data representative; different versions, libraries, or hardware may still cause small differences.

Some imported tools are not used in every branch (for example, `GridSearchCV`, `RobustScaler`, and `accuracy_score`). Importing a tool does not mean the notebook applies it.

## 3. Loading and inspecting data

### Input files

The default cells read `train.csv` (labelled), `test.csv` (to score), and `submission_template.csv` (output structure). A commented alternative handles labelled rows split between train and test, with `data_without_label.csv` as the unlabelled prediction set. Only concatenate the two labelled files if both really contain the target.

`shape` reports row and column counts; `head()` previews rows. `info()` shows dtypes and non-null counts. The check table calculates data type, missing count/rate, and number of unique values. `describe(include="all").T` summarizes columns; `duplicated()` counts fully identical rows.

**Why it matters:** detect incorrect types (dates read as strings), mostly empty fields, identifiers included as predictors, duplicate rows, and implausible values before modeling.

**Strengths:** quick, accessible, and useful for forming data-quality questions.

**Limitations:** summary statistics do not reveal why an issue exists or what it means in the business context. Duplicate rows may be valid and should not be deleted automatically. `nunique(dropna=False)` treats missing as an additional distinct value.

## 4. Defining the target, X, and y

Class 1 is described as bad debt, with the example definition “more than three months overdue.” The label creation cell uses `overdue_months > 3`; exactly three months remains class 0. Run this branch only when the dataset **does not already have a target** and the overdue field has the correct meaning. If a target is supplied, do not run it: it may overwrite labels or implement a different definition.

The following values are placeholders: `target_col = "target"`, `id_cols = ["ID"]`, `drop_cols = []`. Replace them with the real schema. `X` contains predictors after removing the target, identifiers, and deliberately excluded columns; `y` contains the target. `X_test` removes identifiers/excluded columns but has no target. An ID is usually needed to join results back to records, not used as a predictor when it is an arbitrary code.

Use `value_counts()` and class percentages to understand how rare class 1 is. Confirm that the target is binary, correctly encoded, and consistently defined in every file. Label definition is a business decision: specify the observation date, prediction horizon, and period used to confirm bad debt.

## 5. Quality checks, imbalance, outliers, and PSI

### Data types, missingness, constants, and cardinality

The notebook assigns numeric dtypes to `numerical_cols` and all other dtypes to `categorical_cols`. This is convenient but may be wrong: numeric category codes may be treated as continuous, while string dates may be treated as categories. Review the split using domain knowledge.

- **Missingness:** share of values absent in a column. Missingness can be random, caused by collection processes, or meaningful in its own right.
- **Constant feature:** `unique <= 1`; it cannot distinguish records and is usually unhelpful. Confirm it is not a special business flag before dropping it.
- **High cardinality:** the notebook flags `unique_ratio >= 0.80` (unique values divided by row count). This is a heuristic, not a rule. Many values could indicate an ID or a useful continuous variable.
- **Infinity:** `+inf` or `-inf` often comes from division by zero or transformations. A standard imputer is not guaranteed to handle infinity; convert it to missing deliberately or fix the calculation.

### Class imbalance

If bad debt is rare, a classifier that always predicts 0 may still have high accuracy. The notebook therefore inspects target frequencies and emphasizes positive-class F1, Average Precision, class weights, and threshold choice. No balancing technique is always correct; evaluation should reflect the consequences of false negatives and false positives.

### IQR outlier check

For each numeric feature, the notebook computes `IQR = Q3 - Q1`, a lower bound `Q1 - 1.5 × IQR`, and upper bound `Q3 + 1.5 × IQR`, then counts/rates observations outside those bounds and calculates skewness. This is a **diagnostic flag**, not proof of bad data. Income and loan amounts can naturally be right-skewed; deleting or winsorizing them without justification can remove signal. The IQR rule is also less stable for highly skewed or discrete distributions.

### Population Stability Index (PSI)

PSI compares train and test distribution shares across bins/categories:

`PSI = Σ (p_test - p_train) × ln(p_test / p_train)`.

Numeric bin edges come from training quantiles; categorical values use the union of train/test levels. Shares are clipped at `1e-6` to avoid taking a logarithm of zero. The notebook uses heuristic bands: `<0.10` small, `0.10–0.25` moderate, `>0.25` substantial.

**When to use:** check whether labelled and unlabelled sets may represent different populations, or monitor drift.

**Strengths:** summarizes distribution change in a reportable number.

**Weaknesses:** thresholds are context-dependent; bins, sample size, missingness, and new categories affect the result. PSI does not assess the unlabelled target distribution, prove model degradation, or distinguish harmless from harmful shifts. The categorical calculation can be sensitive to very high cardinality.

## 6. Data leakage

**Leakage** occurs when a predictor contains information that would only exist after the prediction time or directly reveals the target. Examples include delinquency, collections, write-offs, and post-default status. The validation score may look strong even though the feature is unavailable in production.

The notebook provides three screening checks:

1. Search column names for words such as `target`, `default`, `overdue`, `collection`, `writeoff`, `recovery`, and `dpd`.
2. Inspect absolute correlations between numeric predictors and the target.
3. For columns with at most two values, calculate the rate at which their values match the target.

These are **investigation flags**, not verdicts. High correlation does not prove leakage, and low correlation does not prove safety. Ask whether the feature exists at scoring time, when it was created, and whether its definition depends on a future outcome. A random split can also leak time structure across periods.

## 7. Optional feature engineering

These cells are examples with placeholder names. Use them only when the columns exist and the transformation makes sense; apply the same logic to train and test. In the notebook, `X` is created before feature engineering. After adding columns to train/test, recreate `X`, `y`, `X_test`, numeric/category lists, and the split. Otherwise, the new features never reach the model.

### Date components

`pd.to_datetime(..., errors="coerce")` turns invalid dates into missing values; the notebook derives year, month, quarter, and day of week. Useful when seasonality matters. Skip when dates carry no signal, are unavailable at scoring time, or a simple component does not represent a cyclic effect well. Temporal data usually requires time-based validation.

### Log transform with `log1p`

`log1p(x) = ln(1+x)` is useful for non-negative, right-skewed values; it reduces the influence of large magnitudes. It cannot be applied directly when `x < -1`; choose another transformation for negative data. Keeping both the original and log copy can create multicollinearity for linear models.

### Ratio feature

`numerator / denominator` creates a relative quantity, such as debt-to-income. The example does not handle a zero or missing denominator, which can produce infinity or missing values. Define a meaningful treatment first. Do not create ratios between quantities with no valid unit/business relationship.

### Interaction feature

`col_a * col_b` represents a combined effect, especially useful when a linear model cannot learn interactions by itself. Trees may learn interactions through branching, though engineered interactions can still help. Creating many interactions increases dimensionality, reduces interpretability, and can overfit.

### Binning

The example groups age using `[0,25,40,60,∞]`. Binning loses detail and depends on cut points. The label `<=25` must match `pd.cut` edge behavior; its default left-open first interval may exclude zero unless `include_lowest=True`. Check missing and out-of-range values. Use bins for defensible business thresholds or step-like relationships, not just to create extra features.

## 8. Splitting and validation

### Default stratified random holdout

`train_test_split(test_size=0.20, stratify=y, random_state=42)` reserves 20% of labelled rows for validation. `stratify` approximately preserves class proportions; the seed makes the split reproducible. Use it when rows are approximately independent and deployment resembles random sampling.

It is unsuitable when customers have repeated rows, the goal is future prediction, or records are otherwise dependent. A random split can place records from the same customer/period on both sides and make validation performance optimistic.

### Time-based split

`make_time_splits` sorts unique timestamps and applies `TimeSeriesSplit`; rows with identical timestamps stay in the same fold. The example uses `n_splits=5`, `gap=0`. Validation comes after training in time. Use it to simulate future-period predictions. A `gap` leaves a buffer between train and validation, useful when labels mature late or observation windows overlap.

Weaknesses: the most recent period may have a different composition; folds are not target-stratified; enough unique times are needed. If customers recur over time, this tests later periods for a continuing portfolio, not unseen customers.

### Group split with GroupKFold

`make_group_splits` and `make_group_holdout` use `GroupKFold` to keep every row from a group (for example, a customer) in one fold. They check for missing groups and require at least `n_splits` groups. Use this when deployment targets unseen customers. It does not enforce chronology or stratify the target; inspect positive counts in each fold.

### Combined time and group split

`make_time_group_holdout` takes the final time window, samples customer groups from it for validation, and excludes all earlier rows from those groups from training. Defaults are `n_splits=5`, `gap=0`, `group_fraction=0.25`, `random_state=42`. This answers a stricter question: “future performance on customers never seen in training.” It may discard many rows and produce a small or imbalanced validation set. Check dates, row counts, groups, and positive counts. Choose the split that matches deployment; do not combine optional methods without a clear reason.

## 9. Preprocessing and pipelines

### Imputation

- `SimpleImputer(strategy="median")`: median; less affected by outliers than the mean. It does not recover true values and reduces variability.
- `mean`: arithmetic mean; more suitable for roughly symmetric data, sensitive to outliers.
- `most_frequent`: mode; convenient for categories, but makes missing records indistinguishable from the most common category.
- `constant`, `fill_value="Unknown"`: makes missingness its own value; ensure compatible dtypes/categories.

The notebook defines separate imputers for illustration. The main pipelines actually use median for numeric columns and most-frequent for categorical columns.

### Categorical encoding

- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`: creates one binary column per level. An unseen test category does not raise an error; it becomes all zeros for that encoded group. No artificial ordering is imposed. It may create many columns, and a dense matrix can use substantial memory. `sparse_output` depends on the scikit-learn version.
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`: maps levels to integers and unseen values to -1. It is compact but imposes an arbitrary order. For nominal categories, that artificial order can mislead linear models and threshold-based trees. Use when categories genuinely have a known order.

### Scaling

- `StandardScaler`: subtracts the mean and divides by standard deviation, usually giving mean 0/std 1. Useful for Logistic Regression and distance/gradient-based methods; sensitive to outliers.
- `RobustScaler`: uses median/IQR and is more resistant to outliers; the notebook defines it but does not use it in the main pipelines.
- Trees generally do not need scaling because their splits depend on ordering/cut points.

### Three preprocessing configurations

1. `preprocess_standard`: numeric median + StandardScaler; categorical most-frequent + one-hot. Used for Logistic Regression.
2. `preprocess_tree`: numeric median without scaling; categorical most-frequent + one-hot. Used for Dummy, Decision Tree, Random Forest, Extra Trees, and XGBoost.
3. `preprocess_ordinal`: numeric median; categorical most-frequent + OrdinalEncoder. Used for HistGradientBoosting.

`ColumnTransformer` sends each column group to the appropriate transformer and concatenates the outputs. `Pipeline` chains preprocessing and the estimator so each fold learns transformations only from its training rows.

## 10. Classification algorithms

The parameters below include explicit notebook settings and important parameters in the tuning space. Parameters not listed use library defaults, which may change across versions.

### DummyClassifier — baseline

The notebook uses `strategy="prior"`: predict the most frequent training class and return probabilities based on the class prior. It does not learn relationships between X and y.

**Strengths:** simple reference point to test whether real models add value. **Weaknesses:** ignores row-level signal. **Use:** include as a baseline. **Do not use:** as a decision model when predictions have value; high accuracy alone is not meaningful under imbalance.

### Logistic Regression

A linear classifier that maps a weighted feature sum through the sigmoid function to estimate probability. Coefficients describe direction on the transformed feature scale; they are not causal effects.

The standard notebook model sets `max_iter=2000`, `random_state=42`; the balanced variant adds `class_weight="balanced"`.

- `max_iter`: maximum optimization iterations. Increase if convergence warnings occur; more iterations do not fix poor scaling or unsuitable regularization.
- `class_weight="balanced"`: weights errors inversely to class frequency. It may improve positive recall, reduce precision, and make probabilities less calibrated.
- `random_state`: seed for solver randomness where applicable.
- `C` (important but not tuned in this notebook): inverse regularization strength; lower C means stronger regularization, often reducing overfit but potentially underfitting.
- `penalty`, `solver`: regularization type and optimization method; choices must be compatible.

**Strengths:** fast, useful baseline, relatively interpretable, often useful probabilities; requires scaling. **Weaknesses:** linear boundary in transformed feature space; needs feature engineering for nonlinear effects/interactions; multicollinearity destabilizes coefficients.

**Use:** compact, interpretable model with approximately linear relationships. **Less suitable:** complex nonlinear relationships without useful engineered features.

### Decision Tree

A tree asks sequential questions such as `income <= threshold?`, partitions the data, and predicts at leaves. The notebook only sets `random_state=42`; other complexity controls use library defaults.

- `max_depth`: maximum depth; a smaller value limits overfit but can underfit.
- `min_samples_split`: minimum observations required to split a node; larger values make the tree more conservative.
- `min_samples_leaf`: minimum observations per leaf; larger values stabilize predictions.
- `criterion`: split quality measure such as Gini, entropy, or log loss, depending on estimator/version.
- `class_weight`: class error weights.
- `max_features`: number of features considered per split.

**Strengths:** nonlinear effects/interactions, no scaling, shallow trees are readable. **Weaknesses:** a single tree is unstable, high variance, and prone to overfit; leaf probabilities are coarse. **Use:** simple decision rules. Avoid an uncontrolled deep tree when stability/generalization matters.

### Random Forest

Fits many trees on bootstrap samples; each split considers a feature subset; predictions are aggregated to reduce single-tree variance. The notebook uses `n_estimators=300`, `n_jobs=-1`, `random_state=42`, plus a `class_weight="balanced"` variant.

- `n_estimators`: number of trees; more often stabilizes predictions but costs time and memory, with diminishing returns.
- `n_jobs=-1`: use available CPU cores; faster but resource-intensive.
- `random_state`: reproducibility for randomized parts.
- `max_depth`: depth limit; `None` lets trees grow until stopping rules, potentially overfitting individual trees.
- `min_samples_split`: minimum rows to split a node.
- `min_samples_leaf`: minimum rows per leaf; increasing it can reduce noisy predictions.
- `max_features`: features considered per split (`sqrt`, `log2`, or `None`); sampling reduces correlation between trees, but too few features can weaken each tree.
- `class_weight="balanced"`: adjusts class influence; must be evaluated rather than assumed better.

**Strengths:** strong nonlinear tabular candidate, little scaling need, generally more stable than one tree. **Weaknesses:** large and less directly interpretable; dense one-hot features consume memory; poor extrapolation; impurity importance is biased toward high-cardinality features.

**Use:** robust nonlinear model for tabular data. **Not always needed:** when latency/model size, extrapolation, or explicit readable rules dominate.

### Extra Trees (Extremely Randomized Trees)

An ensemble like Random Forest that randomizes split thresholds more aggressively. Notebook settings: `n_estimators=300`, `n_jobs=-1`, `random_state=42`. `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `class_weight` control complexity and class weighting similarly to Random Forest.

**Strengths:** can reduce variance and may be fast; useful comparison. **Weaknesses:** added randomness can increase bias; it is not guaranteed to beat RF and is hard to interpret. **Use:** additional ensemble benchmark. **Skip:** if compute is limited and other candidates already satisfy requirements.

### HistGradientBoostingClassifier

Boosting adds trees sequentially: later trees correct errors from earlier ensemble predictions. Histogram binning makes split search efficient. Notebook settings: `learning_rate=0.08`, `max_iter=250`, `random_state=42`, with ordinal encoding.

- `learning_rate`: contribution of each tree; a lower value often requires more iterations.
- `max_iter`: number of boosting rounds/trees; more increases capacity and overfit risk without regularization/early stopping.
- `max_leaf_nodes` (important, not set): maximum leaves per tree.
- `max_depth`, `min_samples_leaf`, `l2_regularization` (version-dependent): control tree structure and regularization.
- `random_state`: seed.

**Strengths:** learns nonlinear tabular relationships, no scaling. **Weaknesses:** tuning-sensitive; ordinal encoding nominal categories imposes false order; categorical/NaN support varies by version. **Use:** try sklearn boosting on tabular data. **Less suitable:** if interpretability is paramount or the category encoding is invalid.

### XGBoost (optional)

Gradient boosting with optimized loss, regularization, and sampling. The package must be installed. Notebook settings: `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `eval_metric="logloss"`, `n_jobs=-1`, `random_state=42`.

- `n_estimators`: boosting rounds/trees; increases capacity and runtime/overfit risk.
- `max_depth`: tree depth; deeper trees learn complex interactions but overfit more easily.
- `learning_rate` (`eta`): shrinks each tree contribution; low values usually need more trees.
- `subsample`: row fraction sampled per round; below 1 may regularize.
- `colsample_bytree`: feature fraction per tree; can reduce correlation/overfit and runtime.
- `min_child_weight` (in tuning): minimum child-node sum of Hessian/weights; larger values are more conservative.
- `reg_alpha`: L1 regularization; encourages sparse weights.
- `reg_lambda`: L2 regularization; shrinks weights.
- `scale_pos_weight`: positive-class weight; negative/positive ratio is a heuristic starting point, not a guarantee of better metrics or calibrated probabilities.
- `eval_metric="logloss"`: log-loss metric inside the estimator; the notebook ranks models by AP.
- `n_jobs`, `random_state`: parallelism and reproducibility.

**Strengths:** often strong on tabular data, with several regularization mechanisms and interaction learning. **Weaknesses:** many parameters, expensive tuning, less directly interpretable, package/version dependency. **Use:** when resources allow tuning and benchmarking. **Skip:** when extra installation is disallowed, runtime is limited, or simpler models already meet the goal.

Only Random Forest and XGBoost are tuned in the notebook. Logistic Regression, Decision Tree, Extra Trees, and HistGradientBoosting are not automatically tuned. Starting values are not optima; choose parameters with appropriate validation/CV.

## 11. Evaluation metrics and threshold selection

### Confusion matrix

| Actual / predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

- **False Negative (FN):** bad debt missed, potentially causing credit losses.
- **False Positive (FP):** a good customer flagged, potentially losing an opportunity or receiving unfair treatment.

The matrix depends on the threshold and should be interpreted with business costs.

### Precision, recall, and F1

- Positive-class precision = `TP/(TP+FP)`: among predicted bad-debt cases, how many are truly positive.
- Positive-class recall = `TP/(TP+FN)`: among actual bad-debt cases, how many are found.
- F1 is the harmonic mean of precision and recall.
- `F1_0`, `F1_1`: one-vs-rest F1 for each class. `Weighted_F1`: support-weighted average, so the majority class has greater influence.

F1 balances precision/recall but does not encode separate error costs and ignores true negatives.

### Average Precision and PR curve

`average_precision_score(y, probability)` summarizes precision across recall levels over thresholds and measures positive-class ranking. It is useful when positives are rare; its baseline is near the positive prevalence, so report prevalence alongside it. The model table sorts by AP, then Weighted F1. AP uses probabilities/scores; F1 in the table uses labels at the default threshold (typically 0.5).

### Classification report

Reports precision, recall, F1, and support per class, plus macro average (equal class weighting) and weighted average. `digits=4` controls display precision; `zero_division=0` returns zero when a denominator is zero.

### Threshold tuning

The notebook tries thresholds from 0.10 to 0.90 in steps of 0.05, calculates positive-class precision/recall/F1 and Weighted F1, then selects the row with the highest positive-class F1.

A lower threshold usually predicts more positives: recall tends to rise and precision may fall. A higher threshold predicts fewer positives: precision may rise and recall may fall. Choose using costs and review capacity, not F1 alone. Select thresholds with validation/CV data; using the same validation set to choose model, threshold, and final reported performance creates optimism from repeated selection. Consider nested CV or a final untouched holdout.

## 12. Tuning and cross-validation

### RandomizedSearchCV

Samples a subset of parameter combinations from a search space, which is cheaper than a large grid. Pipeline tuning ensures preprocessing is refit inside each fold.

- `n_iter=20`: test 20 combinations; fewer is cheaper but can miss good regions.
- `scoring="average_precision"`: optimize AP, suitable when positive ranking matters under imbalance.
- `cv=3`: three folds under scikit-learn's default behavior; temporal/group tasks need appropriate supplied splits.
- `random_state=42`: reproducible sampling of configurations.
- `n_jobs=-1`: parallel work; requires CPU/RAM.
- `verbose=1`: print progress.

Random Forest search space:

| Parameter | Values tried | Meaning |
|---|---|---|
| `n_estimators` | 200, 300, 400, 500, 700 | Number of trees, stability, cost |
| `max_depth` | 6, 8, 10, 12, 16, `None` | Tree complexity limit |
| `min_samples_split` | 2, 5, 10 | Minimum rows for a split |
| `min_samples_leaf` | 1, 2, 3, 5 | Minimum rows per leaf |
| `max_features` | `sqrt`, `log2`, `None` | Features considered per split |
| `class_weight` | `None`, `balanced` | Class weighting |

The XGBoost space also includes `learning_rate [0.01,0.03,0.05,0.08,0.10]`, `subsample [0.7,0.8,0.9,1.0]`, the same values for `colsample_bytree`, `min_child_weight [1,3,5,8]`, `reg_alpha [0,0.01,0.1,0.5]`, `reg_lambda [1,2,5,10]`, and `scale_pos_weight [1, class_ratio]`, alongside the tree counts and depths described above.

`best_score_` is the best cross-validation score within the training partition. The resulting `best_estimator_` is then scored on `X_valid`. If validation is used for tuning and then presented as an independent final result, the estimate is optimistic.

### Cross-validation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` makes five folds with similar target proportions. `cross_val_score(..., scoring="average_precision")` refits the full pipeline on each fold. The mean is average performance; a large standard deviation signals split sensitivity.

The notebook's example runs this on all `X, y`; if the same data already drove model/feature/threshold selection, it is not fully independent. For temporal tasks use `make_time_splits` and replace `cv=3`; for group tasks use group folds. CV cannot repair leakage in features or labels.

## 13. Model interpretation

### Tree feature importance

`feature_importances_` aggregates impurity reduction caused by a feature. Names are post-transform, so one-hot categories may appear as separate rows.

**Strengths:** fast and built in; shows which inputs the model used. **Weaknesses:** impurity importance is biased toward high-cardinality/continuous variables, correlated features share importance, and it provides neither direction nor causal evidence.

### Logistic Regression coefficients

A positive coefficient increases positive-class log-odds as the feature rises, holding other model inputs fixed; a negative coefficient decreases them. For coefficient β, the odds ratio for a one-unit increase is `exp(β)`; the unit is after preprocessing. Interpret scaled/one-hot coefficients cautiously; correlated features make them unstable.

### Permutation importance

Shuffle one raw validation column and measure the change in AP. The notebook uses `n_repeats=5`, `random_state=42`. A large AP decrease after shuffling means the model relied on that feature.

**Strengths:** tied to an out-of-sample metric and works with many model types. **Weaknesses:** can be slow; correlated substitutes make each feature look less important; five repeats may be noisy.

### SHAP (optional)

SHAP decomposes a prediction into a baseline plus feature contributions using a game-theoretic allocation approach. The notebook samples up to 500 validation rows, preprocesses them, then uses `TreeExplainer` for XGBoost and Random Forest.

- **Bar:** global importance (usually mean absolute SHAP); no direction.
- **Beeswarm:** distribution of SHAP values; color shows feature value and horizontal position shows contribution direction/magnitude under the model.
- **Waterfall:** explains one observation from baseline to output.

Random Forest selects class 1 with `rf_shap_values[:, :, 1]`. XGBoost SHAP output formats can vary by package version/objective; verify the class and output scale. SHAP explains model behavior, not causation, and does not rule out leakage or bias.

## 14. Final fit, predictions, and submission

The notebook sets `final_model = random_forest`, `final_threshold = 0.50`, refits on all `X, y`, calls `predict_proba(X_test)[:, 1]`, thresholds probabilities, previews ten records and positive prediction rate, writes predictions to the template's `target` column, and saves `submission_final.csv`.

After model selection with train/validation, fitting on all labelled data uses more information. An unlabelled test set cannot provide a performance score; its positive prediction rate is only a sanity check. The template must match `X_test` row count/order and required output name/type. Recreate engineered features on test. If choosing `rf_best` or another threshold, update `final_model`, `final_threshold`, and final summary consistently.

## 15. Report outline

### A. Problem and label

- Population and prediction time; positive label definition and outcome horizon.
- Train/validation/test sizes and class proportions.
- Consequences of false negatives and false positives.

### B. Data and preparation

- Types, missingness, duplicates, constants, cardinality, infinity, outliers.
- Decisions and reasons; distinguish checks from transformations actually applied.
- Leakage review, especially whether each input exists at scoring time.
- PSI/distribution shift and interpretation limits.
- Only describe feature engineering actually run.

### C. Method

- Split strategy (random/time/group) matched to deployment.
- Imputation, encoding, scaling pipeline.
- Baseline, candidate models, parameters, class weighting.
- Search space, AP objective, CV design, random seed.

### D. Results

- AP, per-class F1, Weighted F1; include precision/recall when relevant.
- Selection criterion, confusion matrix, PR curve, threshold trade-off.
- CV mean/std; distinguish CV, holdout validation, and test predictions.

### E. Interpretation and limitations

- Feature/permutation/SHAP results and their limits; avoid causal claims.
- Drift, calibration, representativeness, error costs, and fairness when relevant.
- Evidence-based improvements: temporal/group validation, leakage removal, calibration, cost-based threshold, drift monitoring, more data.

The notebook contains no saved computed outputs, so fill in actual values after running it on the dataset. Do not present example settings as empirical results.

## 16. Before using the notebook on real data

1. Replace target/ID/excluded columns, file names, and submission column with the actual schema.
2. Do not create `bad_debt` if a target already exists; verify the “more than three months” rule.
3. Choose random/time/group validation to match the deployment question; the optional branches are alternatives.
4. Check leakage and whether features exist at prediction time.
5. After feature engineering, recreate `X`, `X_test`, column lists, and split; handle zero denominators, missingness, log domains, and bin edges.
6. High-cardinality one-hot with dense output can consume substantial RAM; check the encoder and scikit-learn version.
7. Ordinal encoding for unordered categories is a strong assumption.
8. Set class weights and thresholds for the objective; F1 is not a cost function.
9. After tuning, explicitly select `best_estimator_`; the final section still points to the original RF.
10. RF SHAP requires `rf_best` from tuning; XGBoost SHAP requires the package/model and compatible output format.
11. Without real data and computed results, it is not possible to conclude which model performs best.
