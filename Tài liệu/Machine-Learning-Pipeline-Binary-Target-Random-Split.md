# Machine Learning Pipeline for Binary Target - Random Split

## Tổng quan

Đây là workflow phân loại nhị phân (binary classification) sử dụng stratified random train/validation/test split. Pipeline bao gồm so sánh baseline, tuning model với stratified cross-validation, chọn threshold trên validation set, đánh giá trên test holdout, và tạo predictions cho dataset không có nhãn.

**Trước khi chạy:** Đặt `train.csv`, `test.csv`, và `submission_template.csv` trong thư mục làm việc của notebook. Kiểm tra `target_col`, `id_cols`, và các ví dụ feature engineering tùy chọn phù hợp với cấu trúc dữ liệu thực tế.

---

## Mục lục

1. [Import Libraries](#1-import-libraries)
2. [Load and Combine Datasets](#2-load-and-combine-datasets)
3. [Quick Data Check](#3-quick-data-check)
4. [Target Definition](#4-target-definition)
5. [Select Target and Features](#5-select-target-and-features)
6. [Select Numerical and Categorical Columns](#6-select-numerical-and-categorical-columns)
7. [Data Quality Tests](#7-data-quality-tests)
8. [Outlier Tests](#8-outlier-tests)
9. [Population Stability Index (PSI)](#9-population-stability-index)
10. [Feature Leakage Tests](#10-feature-leakage-tests)
11. [Optional Feature Engineering](#11-optional-feature-engineering)
12. [Stratified Random Train/Validation/Test Split](#12-stratified-random-trainvalidationtest-split)
13. [Missing Value Methods](#13-missing-value-methods)
14. [Encoding Methods](#14-encoding-methods)
15. [Scaling Methods](#15-scaling-methods)
16. [Preprocessing Pipelines](#16-preprocessing-pipelines)
17. [F1 Scoring Rule](#17-f1-scoring-rule)
18. [Fast Baseline Comparison](#18-fast-baseline-comparison)
19. [RandomizedSearchCV Setup](#19-randomizedsearchcv-setup)
20-22. [Hyperparameter Tuning](#20-22-hyperparameter-tuning)
23. [Fit Tuned Models](#23-fit-tuned-models)
24-26. [Threshold Tuning](#24-26-threshold-tuning)
27. [Ensemble Experiment](#27-ensemble-experiment)
28. [Compare Models](#28-compare-models)
29. [Refit on Train + Validation](#29-refit-on-train--validation)
30-32. [Feature Importance](#30-32-feature-importance)
33. [Select Final Ensemble](#33-select-final-ensemble)
34. [Create Submission](#34-create-submission)
35. [Final Run Summary](#35-final-run-summary)

---

## 1. Import Libraries

Load các thư viện cần thiết cho data preparation, modeling, validation, tuning, và interpretation.

```python
# Required packages: pandas numpy matplotlib scikit-learn xgboost catboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
```

---

## 2. Load and Combine Datasets

Hai file dữ liệu (labelled và unlabelled) được ghép lại ngay lập tức với internal source tag. Feature engineering row-wise chạy một lần trên combined table. Sau đó, source tag phân tách các hàng labelled cho việc split từ các hàng unlabelled để predict cuối cùng.

```python
# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("submission_template.csv")

# Mark source before combining
source_col = "_dataset_source"
train[source_col] = "labeled"
test[source_col] = "predict"

combined_data = pd.concat([train, test], ignore_index=True, sort=False)
```

**Quy trình:**
- Imputers và encoders nằm trong model pipeline và được học từ training data only
- Việc kết hợp này đảm bảo feature engineering nhất quán giữa train và test

---

## 3. Quick Data Check

Kiểm tra cấu trúc và chất lượng cơ bản của dataset.

```python
train.info()

data_check = pd.DataFrame({
    "dtype": train.dtypes,
    "missing": train.isna().sum(),
    "missing_pct": (train.isna().mean() * 100).round(2),
    "unique": train.nunique(dropna=False)
})

print("Duplicate rows:", train.duplicated().sum())
display(train.describe(include="all").T)
```

---

## 4. Target Definition

**Định nghĩa:** Bad debt là nợ quá hạn thanh toán hơn 3 tháng.

- Label 1: Bad debt
- Label 0: No bad debt

### Label Creation from Overdue Months

Transform này áp dụng khi dữ liệu chứa trường overdue-duration đo bằng tháng và không có target label được cung cấp.

```python
# If target must be derived:
if "target" not in train.columns and "overdue_months" in train.columns:
    train["target"] = (train["overdue_months"] > 3).astype(int)
```

---

## 5. Select Target and Features

Phân tách target khỏi input columns và loại trừ identifiers hoặc fields không nên dùng cho prediction.

```python
# Configuration
target_col = "target"  # labelled outcome column
id_cols = ["ID"]       # identifier columns
drop_cols = []          # add leakage columns here
TIME_COL = "application_date"  # optional

# Validate target exists
if target_col not in train.columns:
    raise KeyError(f"Target column {target_col!r} was not found.")
```

---

## 6. Select Numerical and Categorical Columns

```python
numerical_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
```

---

## 7. Data Quality Tests

### 7.1 Missingness, Cardinality, and Constant Features

```python
feature_check = pd.DataFrame({
    "dtype": X.dtypes.astype(str),
    "missing_pct": (X.isna().mean() * 100).round(2),
    "unique": X.nunique(dropna=False),
    "unique_ratio": (X.nunique(dropna=False) / len(X)).round(4)
})

constant_features = feature_check[feature_check["unique"] <= 1]
high_cardinality = feature_check[feature_check["unique_ratio"] >= 0.80]
```

### 7.2 Infinite Values

```python
numeric_data = X.select_dtypes(include="number")
infinite_count = pd.DataFrame({
    "positive_inf": np.isposinf(numeric_data).sum(),
    "negative_inf": np.isneginf(numeric_data).sum()
})
```

### 7.3 Target Imbalance

```python
target_distribution = pd.DataFrame({
    "count": y.value_counts(),
    "percent": (y.value_counts(normalize=True) * 100).round(2)
})
```

---

## 8. Outlier Tests

IQR rule: flags observations below Q1 - 1.5×IQR hoặc above Q3 + 1.5×IQR.

```python
outlier_rows = []
for col in numerical_cols:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((X[col] < lower) | (X[col] > upper)).sum()
    outlier_rows.append({
        "Feature": col, "Q1": q1, "Q3": q3,
        "Lower": lower, "Upper": upper,
        "Outliers": outlier_count,
        "Outlier_pct": outlier_count / len(X) * 100,
        "Skewness": X[col].skew()
    })
```

---

## 9. Population Stability Index (PSI)

PSI so sánh phân phối feature giữa train và test data.

**Heuristic:**
- PSI < 0.10: Small shift
- 0.10 - 0.25: Moderate shift
- > 0.25: Substantial shift

```python
def psi_numeric(train_series, test_series, bins=10):
    # Bins based on train quantiles
    edges = np.unique(np.quantile(train_series, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    
    train_bin = pd.cut(train_series, bins=edges, include_lowest=True)
    test_bin = pd.cut(test_series, bins=edges, include_lowest=True)
    
    train_pct = train_bin.value_counts(normalize=True, sort=False).clip(lower=1e-6)
    test_pct = test_bin.value_counts(normalize=True, sort=False).reindex(train_pct.index, fill_value=0).clip(lower=1e-6)
    
    return ((test_pct - train_pct) * np.log(test_pct / train_pct)).sum()
```

---

## 10. Feature Leakage Tests

Kiểm tra các feature có thể chứa target information không có sẵn tại thời điểm prediction thực tế.

```python
leakage_words = [
    "target", "label", "bad_debt", "default", "overdue",
    "delinquent", "arrears", "collection", "writeoff",
    "write_off", "chargeoff", "charge_off", "recovery", "dpd"
]

name_leakage_candidates = [
    col for col in X.columns
    if any(word in col.lower() for word in leakage_words)
]
```

### Target Correlations

```python
target_correlations = (
    train[numerical_cols + [target_col]].corr(numeric_only=True)[target_col]
    .drop(target_col).abs().sort_values(ascending=False)
)
```

### Copy Score Candidates

```python
binary_candidates = [col for col in X.columns if train[col].nunique(dropna=False) <= 2]
copy_scores = pd.Series({
    col: (train[col].astype(str) == train[target_col].astype(str)).mean()
    for col in binary_candidates
}).sort_values(ascending=False)
```

---

## 11. Optional Feature Engineering

### 11.1 Date Features

```python
if date_col in combined_data.columns:
    combined_data[date_col] = pd.to_datetime(combined_data[date_col], errors="coerce")
    combined_data["year"] = combined_data[date_col].dt.year
    combined_data["month"] = combined_data[date_col].dt.month
    combined_data["quarter"] = combined_data[date_col].dt.quarter
    combined_data["day_of_week"] = combined_data[date_col].dt.dayofweek
```

### 11.2 Log Transformation

```python
col = "income"  # replace with suitable non-negative column
if col in combined_data.columns:
    values = pd.to_numeric(combined_data[col], errors="coerce")
    if values.dropna().ge(0).all():
        combined_data[col + "_log"] = np.log1p(values)
```

### 11.3 Ratio Feature

```python
numerator_col = "numerator_col"
denominator_col = "denominator_col"
if numerator_col in combined_data.columns and denominator_col in combined_data.columns:
    denominator = pd.to_numeric(combined_data[denominator_col], errors="coerce").replace(0, np.nan)
    combined_data["new_ratio"] = pd.to_numeric(combined_data[numerator_col], errors="coerce") / denominator
```

### 11.4 Interaction Feature

```python
col_a = "col_a"
col_b = "col_b"
if col_a in combined_data.columns and col_b in combined_data.columns:
    combined_data["interaction_feature"] = combined_data[col_a] * combined_data[col_b]
```

### 11.5 Binning

```python
bins = [0, 25, 40, 60, np.inf]
labels = ["<=25", "26-40", "41-60", "60+"]
if "age" in combined_data.columns:
    combined_data["age_group"] = pd.cut(combined_data["age"], bins=bins, labels=labels)
```

---

## 12. Stratified Random Train/Validation/Test Split

Sử dụng stratified random split với proportions và seed được cấu hình.

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

def make_random_train_valid_test(X_all, y_all, train_size=0.70, valid_size=0.15, test_size=0.15, random_state=42):
    """Stratified random split into three labelled partitions."""
    X_train_part, X_holdout, y_train_part, y_holdout = train_test_split(
        X_all, y_all, train_size=train_size, random_state=random_state, stratify=y_all
    )
    valid_share_of_holdout = valid_size / (valid_size + test_size)
    X_valid_part, X_test_part, y_valid_part, y_test_part = train_test_split(
        X_holdout, y_holdout, train_size=valid_share_of_holdout,
        random_state=random_state, stratify=y_holdout
    )
    return X_train_part, X_valid_part, X_test_part, y_train_part, y_valid_part, y_test_part

# Default split: 70% train, 15% validation, 15% test
X_train, X_valid, X_test_split, y_train, y_valid, y_test_split = make_random_train_valid_test(
    X, y, train_size=0.70, valid_size=0.15, test_size=0.15, random_state=42
)
```

---

## 13. Missing Value Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Median | Less sensitive to extreme values | Numeric features |
| Mean | Simple, works best with limited outliers | Numeric features |
| Most Frequent | Fill categorical gaps | Categorical features |
| Constant ("Unknown") | Explicit level for missingness | Categorical features (when missingness carries info) |

```python
numeric_median = SimpleImputer(strategy="median")
numeric_mean = SimpleImputer(strategy="mean")
categorical_mode = SimpleImputer(strategy="most-frequent")
categorical_unknown = SimpleImputer(strategy="constant", fill_value="Unknown")
```

---

## 14. Encoding Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| One-Hot Encoding | One binary column per category | Nominal categories, tree models |
| Ordinal Encoding | Convert to integer codes | Compact representation, when order is acceptable |

```python
onehot = OneHotEncoder(handle_unknown="ignore")
ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

---

## 15. Scaling Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| StandardScaler | Center around zero, scale by std | Linear models, neural networks |
| RobustScaler | Use median and IQR | When outliers present |

```python
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()
```

---

## 16. Preprocessing Pipelines

### Standard Scaling + One-Hot Encoding

```python
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most-frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_standard = ColumnTransformer([
    ("num", numeric_pipe, numerical_cols),
    ("cat", categorical_pipe, categorical_cols)
])
```

### No Scaling + One-Hot (Tree Models)

```python
numeric_pipe_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="most-frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_tree = ColumnTransformer([
    ("num", numeric_pipe_tree, numerical_cols),
    ("cat", categorical_pipe_tree, categorical_cols)
])
```

---

## 17. F1 Scoring Rule

```python
F1_AVERAGE = "binary"  # F1 for label 1 only
POSITIVE_LABEL = 1

def calculate_metrics(y_true, probability, threshold=0.50):
    """Return ROC_AUC, PR_AUC, Accuracy, Precision, Recall, F1."""
    y_true = pd.Series(y_true).astype(int)
    prediction = (np.asarray(probability) >= threshold).astype(int)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probability, pos_label=POSITIVE_LABEL)
    
    return {
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": auc(recall_curve, precision_curve),
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(y_true, prediction, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(y_true, prediction, average="binary", pos_label=POSITIVE_LABEL, zero_division=0),
    }
```

---

## 18. Fast Baseline Comparison

Fit baseline models và report metrics trên train, validation, và test sets.

### Models Tested

1. **Logistic Regression** - Linear baseline với balanced class weights
2. **Decision Tree** - Simple tree với max_depth=5
3. **Random Forest** - Ensemble với n_estimators=120
4. **XGBoost** - Gradient boosting với regularization
5. **CatBoost** - Native categorical handling

```python
logistic_model = Pipeline([
    ("preprocess", clone(preprocess_standard)),
    ("model", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])

random_forest_baseline = Pipeline([
    ("preprocess", clone(preprocess_tree)),
    ("model", RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
])

xgboost_baseline = Pipeline([
    ("preprocess", clone(preprocess_tree)),
    ("model", XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)),
])

catboost_baseline = CatBoostClassifier(
    iterations=120, depth=5, learning_rate=0.05, l2_leaf_reg=3,
    loss_function="Logloss", verbose=False, random_seed=RANDOM_STATE,
    cat_features=categorical_cols,
)
```

---

## 19. RandomizedSearchCV Setup

```python
SEARCH_ITERATIONS = 10  # Reduce to 5 for faster pass
SEARCH_N_JOBS = -1       # Parallelize
CV_FOLDS = 3

cv_splits = list(StratifiedKFold(
    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
).split(X_train, y_train))
```

---

## 20-22. Hyperparameter Tuning

### Random Forest

```python
rf_search = RandomizedSearchCV(
    estimator=Pipeline([
        ("preprocess", clone(preprocess_tree)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ]),
    param_distributions={
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__max_depth": [4, 6, 8, 10, 12, 14, None],
        "model__min_samples_leaf": [2, 3, 5, 7, 10],
        "model__max_features": [0.4, 0.6, 0.8, 1.0],
        "model__class_weight": ["balanced", "balanced_subsample", None],
    },
    n_iter=SEARCH_ITERATIONS, scoring=F1_SCORING, cv=cv_splits,
    random_state=RANDOM_STATE, n_jobs=SEARCH_N_JOBS, verbose=1,
    refit=True, error_score="raise", return_train_score=False,
)
```

### XGBoost

```python
xgb_search = RandomizedSearchCV(
    estimator=Pipeline([
        ("preprocess", clone(preprocess_tree)),
        ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)),
    ]),
    param_distributions={
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.10, 0.15],
        "model__min_child_weight": [1, 3, 5, 7, 10],
        "model__subsample": [0.70, 0.80, 0.90, 1.0],
        "model__colsample_bytree": [0.70, 0.80, 0.90, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    },
    n_iter=SEARCH_ITERATIONS, scoring=F1_SCORING, cv=cv_splits,
    random_state=RANDOM_STATE, n_jobs=SEARCH_N_JOBS, verbose=1,
    refit=True, error_score="raise", return_train_score=False,
)
```

### CatBoost

```python
cat_search = RandomizedSearchCV(
    estimator=CatBoostClassifier(
        loss_function="Logloss", verbose=False, random_seed=RANDOM_STATE,
        cat_features=categorical_cols, thread_count=1,
    ),
    param_distributions={
        "iterations": [100, 150, 200, 250, 300],
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.05, 0.08, 0.10, 0.15],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 10.0],
    },
    n_iter=SEARCH_ITERATIONS, scoring=F1_SCORING, cv=cv_splits,
    random_state=RANDOM_STATE, n_jobs=SEARCH_N_JOBS, verbose=1,
    refit=True, error_score="raise", return_train_score=False,
)
```

### Logistic Regression

```python
logistic_search = RandomizedSearchCV(
    estimator=Pipeline([
        ("preprocess", clone(preprocess_standard)),
        ("model", LogisticRegression(solver="liblinear", max_iter=1500, random_state=RANDOM_STATE)),
    ]),
    param_distributions={
        "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"],
    },
    n_iter=SEARCH_ITERATIONS, scoring=F1_SCORING, cv=cv_splits,
    random_state=RANDOM_STATE, n_jobs=SEARCH_N_JOBS, verbose=1,
    refit=True, error_score="raise", return_train_score=False,
)
```

---

## 23. Fit Tuned Models

```python
tuned_models = {
    "Logistic Regression": Pipeline([
        ("preprocess", clone(preprocess_standard)),
        ("model", LogisticRegression(**logistic_best_params, solver="liblinear", max_iter=1500, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("preprocess", clone(preprocess_tree)),
        ("model", RandomForestClassifier(**rf_best_params, n_jobs=-1, random_state=RANDOM_STATE))
    ]),
    "XGBoost": Pipeline([
        ("preprocess", clone(preprocess_tree)),
        ("model", XGBClassifier(**xgb_best_params, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE))
    ]),
    "CatBoost": CatBoostClassifier(
        **cat_best_params, loss_function="Logloss", verbose=False,
        random_seed=RANDOM_STATE, cat_features=categorical_cols,
    ),
}

def find_best_f1_threshold(y_true, probability):
    """Search dense set of cutoffs from validation scores."""
    probability = np.asarray(probability)
    candidates = np.unique(np.r_[0.0, np.quantile(probability, np.linspace(0.01, 0.99, 199)), 1.0])
    scores = [f1_score(y_true, (probability >= cutoff).astype(int), average="binary", pos_label=1) for cutoff in candidates]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index]), float(scores[best_index])
```

---

## 24-26. Threshold Tuning

Mỗi model có threshold riêng, được chọn trên validation set bằng F1 cho label 1.

```python
# Example: Random Forest
rf_threshold, rf_valid_f1 = find_best_f1_threshold(y_valid, tuned_probabilities[("Random Forest", "Validation")])
thresholds["Random Forest"] = rf_threshold
```

**Workflow:**
1. Tìm threshold tối ưu trên validation set
2. Áp dụng threshold đó cho tất cả splits (train/validation/test)
3. Không tune threshold trên test set

---

## 27. Ensemble Experiment

### 27.1 Blend Helper

```python
def average_model_probabilities(model_names, X_part):
    """Average label-1 probabilities from named tuned models."""
    probabilities = [predict_probability(name, tuned_models[name], X_part) for name in model_names]
    return np.mean(probabilities, axis=0)
```

### 27.2 Tree-only Ensemble: RF + XGBoost + CatBoost

```python
tree_model_names = ["Random Forest", "XGBoost", "CatBoost"]
tree_blend_probability = {
    "Train": average_model_probabilities(tree_model_names, X_train),
    "Validation": average_model_probabilities(tree_model_names, X_valid),
    "Test holdout": average_model_probabilities(tree_model_names, X_test_split),
}

tree_blend_threshold, tree_blend_valid_f1 = find_best_f1_threshold(
    y_valid, tree_blend_probability["Validation"]
)
```

### 27.3 Add Logistic Regression

```python
linear_tree_model_names = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]
# ... similar to tree-only ensemble
```

---

## 28. Compare Models

```python
validation_comparison = pd.concat([
    pd.DataFrame([
        {"Model": name, "Threshold": thresholds[name], **calculate_metrics(y_valid, tuned_probabilities[(name, "Validation")], thresholds[name])}
        for name in tuned_models
    ]),
    pd.DataFrame([
        {"Model": name, "Threshold": ensemble_thresholds[name], **calculate_metrics(y_valid, ensemble_probabilities[(name, "Validation")], ensemble_thresholds[name])}
        for name in ensemble_members
    ])
], ignore_index=True).sort_values("F1", ascending=False)
```

**Selection criteria:** Sử dụng validation F1 và practical constraints. Test metrics là diagnostic cuối cùng, không phải tuning feedback.

---

## 29. Refit on Train + Validation

1. Fit models trên train + validation
2. Evaluate trên cùng test holdout
3. Giữ nguyên validation threshold đã chọn

```python
X_development = pd.concat([X_train, X_valid]).sort_index()
y_development = pd.concat([y_train, y_valid]).sort_index()

for model_name in tuned_models:
    # Fit on train + validation
    fit_model(model_name, model, X_development, y_development)
    # Evaluate on test holdout
    probability_holdout = predict_probability(model_name, model, X_test_split)
```

---

## 30-32. Feature Importance

### 30. Built-in Importance

```python
IMPORTANCE_MODEL_NAME = "Random Forest"  # or "XGBoost", "CatBoost"

if IMPORTANCE_MODEL_NAME == "CatBoost":
    importance_values = importance_model.get_feature_importance()
else:
    importance_estimator = importance_model.named_steps["model"]
    importance_values = importance_estimator.feature_importances_
```

### 31. Permutation Importance

```python
perm = permutation_importance(
    perm_model, X_perm, y_valid, scoring=F1_SCORING,
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
)
```

### 32. SHAP (Optional)

```python
try:
    import shap
    shap_explainer = shap.TreeExplainer(shap_estimator)
    shap_values = shap_explainer(shap_input)
    shap.plots.bar(shap_values, max_display=20)
    shap.plots.beeswarm(shap_values, max_display=20)
except ImportError:
    print("SHAP not installed. Install with: %pip install shap")
```

---

## 33. Select Final Ensemble

```python
FINAL_ENSEMBLE_NAME = "RF + XGBoost + CatBoost"  # or "Logistic + RF + XGBoost + CatBoost"
final_threshold = ensemble_thresholds[FINAL_ENSEMBLE_NAME]
final_member_names = ensemble_members[FINAL_ENSEMBLE_NAME]

# Fit on ALL labelled data
for model_name in final_member_names:
    fit_model(model_name, final_models[model_name], X, y)

# Predict on unlabelled data
predict_member_probabilities = [predict_probability(name, final_models[name], X_predict) for name in final_member_names]
predict_probability_final = np.mean(predict_member_probabilities, axis=0)
predict_label_final = (predict_probability_final >= final_threshold).astype(int)
```

---

## 34. Create Submission

```python
if len(submission) != len(predict_label_final):
    raise ValueError(f"Row count mismatch: submission={len(submission)}, predictions={len(predict_label_final)}")

submission[prediction_col] = predict_label_final
submission.to_csv("submission_final.csv", index=False)
print("Saved: submission_final.csv")
```

---

## 35. Final Run Summary

```python
run_summary = pd.DataFrame({
    "Item": [
        "F1 definition", "Positive label", "Final ensemble",
        "Locked validation threshold", "Labelled rows",
        "Unlabelled prediction rows", "Train rows",
        "Validation rows", "Test holdout rows", "Feature count"
    ],
    "Value": [
        F1_AVERAGE, POSITIVE_LABEL, FINAL_ENSEMBLE_NAME,
        final_threshold, len(X), len(X_predict),
        len(X_train), len(X_valid), len(X_test_split), X.shape[1]
    ],
})
```

---

## Cấu hình chính

| Parameter | Default | Mô tả |
|-----------|---------|--------|
| `target_col` | "target" | Tên cột target |
| `id_cols` | ["ID"] | Cột identifiers |
| `TIME_COL` | "application_date" | Cột thời gian (optional) |
| `train_size` | 0.70 | Tỷ lệ train split |
| `valid_size` | 0.15 | Tỷ lệ validation split |
| `test_size` | 0.15 | Tỷ lệ test split |
| `CV_FOLDS` | 3 | Số folds cho cross-validation |
| `SEARCH_ITERATIONS` | 10 | Số iterations cho RandomizedSearchCV |
| `F1_AVERAGE` | "binary" | F1 averaging method |
| `POSITIVE_LABEL` | 1 | Positive class label |
| `RANDOM_STATE` | 42 | Random seed |

---

## Dependencies

```
pandas
numpy
matplotlib
scikit-learn
xgboost
catboost
shap (optional)
```
