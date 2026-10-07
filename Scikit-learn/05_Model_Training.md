# 05 · Chọn và huấn luyện mô hình

## 1. Tạo baseline

Baseline cho biết mức tối thiểu một model cần vượt qua. Với phân loại, `DummyClassifier` có thể luôn dự đoán lớp phổ biến; với hồi quy, `DummyRegressor` dự đoán giá trị đơn giản.

```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline = DummyClassifier(strategy="prior")
scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="balanced_accuracy")
print(scores.mean())
```

Baseline không phải mục tiêu cuối; nó kiểm tra pipeline, metric và khả năng dự đoán cơ bản.

## 2. Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

classifiers = {
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=42,
        class_weight="balanced",
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
}
```

Logistic regression là baseline tuyến tính dễ giải thích; random forest bắt quan hệ phi tuyến; boosting thường mạnh trên dữ liệu bảng. Cần ghép preprocessing phù hợp cho từng model, ví dụ logistic regression thường cần scale số, cây thường không.

## 3. Regression

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

regressors = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42,
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
}
```

Ridge là tuyến tính có regularization; tree ensemble bắt quan hệ phi tuyến. Với target lệch mạnh, cân nhắc biến đổi target có kiểm chứng nghiệp vụ, chẳng hạn `TransformedTargetRegressor`.

## 4. So sánh qua cross-validation

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(
    model, X_train, y_train,
    cv=cv,
    scoring={"f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
    n_jobs=-1,
)
print(scores["test_f1"].mean(), scores["test_f1"].std())
```

Giữ cố định folds khi so sánh các model để phép so sánh công bằng hơn. Với preprocessing, truyền nguyên pipeline làm `model`.

## 5. Fit model cuối

Sau khi chọn quy trình và tham số bằng train/CV, fit lại trên toàn bộ tập train. Chỉ đánh giá một lần trên test đã giữ riêng:

```python
best_model.fit(X_train, y_train)
test_predictions = best_model.predict(X_test)
```
