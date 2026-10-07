# 06 · Đánh giá và tuning

Chọn metric theo câu hỏi nghiệp vụ. Tách test một lần; dùng cross-validation trên train để so sánh và tune.

## 1. Classification metrics

```python
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)

pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, pred))
print("balanced accuracy:", balanced_accuracy_score(y_test, pred))
print("macro F1:", f1_score(y_test, pred, average="macro"))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

- Accuracy: tỷ lệ dự đoán đúng; dễ gây hiểu nhầm nếu lớp lệch.
- Precision: trong dự đoán dương, bao nhiêu là dương thật.
- Recall: trong dương thật, bắt được bao nhiêu.
- F1: cân bằng precision/recall; chọn `macro`, `weighted` hoặc `binary` theo mục tiêu.
- ROC AUC: xếp hạng phân biệt theo xác suất/score; với dữ liệu lệch mạnh thường xem thêm PR curve/Average Precision.

```python
positive_probability = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, positive_probability))
```

## 2. Regression metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

pred = regressor.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))
```

MAE dễ diễn giải theo đơn vị target; RMSE phạt lỗi lớn hơn; R² là mức cải thiện so với baseline trung bình trên dữ liệu đánh giá và có thể âm.

## 3. Grid search trên pipeline

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
}
search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
best_model = search.best_estimator_
```

Tên tham số pipeline theo cấu trúc `step__parameter`. `RandomizedSearchCV` hữu ích khi không gian tham số lớn; search chỉ nên dùng train, không dùng test để chọn.

## 4. Threshold và xác suất

`predict` thường dùng threshold mặc định. Nếu chi phí false negative và false positive khác nhau, chọn ngưỡng trên validation/CV theo quy tắc nghiệp vụ, rồi khóa trước đánh giá test.

```python
from sklearn.metrics import precision_recall_curve

prob = best_model.predict_proba(X_valid)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_valid, prob)
# Chọn threshold theo mục tiêu nghiệp vụ trên validation, không chọn trên test.
threshold = 0.35
pred_at_threshold = (prob >= threshold).astype(int)
```

## 5. Đọc lỗi model

Luôn xem confusion matrix hoặc residuals theo phân khúc, thời gian và nhóm quan trọng. Metric trung bình có thể che giấu hiệu năng kém ở một nhóm hoặc giai đoạn cụ thể.
