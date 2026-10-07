# Scikit-learn: từ dữ liệu thô đến dự đoán

> **Mục tiêu:** tra cứu các bước xây dựng mô hình machine learning bằng scikit-learn, từ chia dữ liệu và tiền xử lý đến đánh giá, dự đoán và lưu mô hình.

## Bắt đầu nhanh

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Đoạn trên giả định mọi feature đều là số. Với dữ liệu hỗn hợp số và phân loại, xem phần `ColumnTransformer` ở file 04.

## Tìm theo việc cần làm

| Nhu cầu | Mở file | Nội dung chính |
|---|---|---|
| Hiểu estimator API, nạp dataset, tách X/y | [01_Data_And_Estimators.md](01_Data_And_Estimators.md) | `fit`, `transform`, `predict`, `score` |
| Chia train/test, cross-validation | [02_Split_And_Cross_Validation.md](02_Split_And_Cross_Validation.md) | `train_test_split`, `KFold`, `cross_validate` |
| Điền missing, scale, encode | [03_Preprocessing.md](03_Preprocessing.md) | `SimpleImputer`, scaler, `OneHotEncoder` |
| Tiền xử lý cột hỗn hợp | [04_ColumnTransformer_Pipeline.md](04_ColumnTransformer_Pipeline.md) | `ColumnTransformer`, `Pipeline`, chống leakage |
| Chọn thuật toán và fit | [05_Model_Training.md](05_Model_Training.md) | baseline, classification, regression |
| Đánh giá và tuning | [06_Evaluation_And_Tuning.md](06_Evaluation_And_Tuning.md) | metrics, `GridSearchCV`, threshold |
| Dự đoán dữ liệu mới và lưu mô hình | [07_Prediction_And_Persistence.md](07_Prediction_And_Persistence.md) | `predict`, xác suất, joblib, checklist |
| Chạy và đọc notebook random split | [08_Notebook_Random_Split_Guide.md](08_Notebook_Random_Split_Guide.md) | giải thích cell, thuật toán, cách chạy |
| Chạy và đọc notebook time split | [09_Notebook_Time_Split_Guide.md](09_Notebook_Time_Split_Guide.md) | chia thời gian, forward CV, cách chạy |

## Quy tắc quan trọng

- Chia train/test trước khi học bất kỳ thống kê tiền xử lý nào. `fit` trên train; test chỉ đi qua `transform`/`predict`.
- Dùng `Pipeline` để gắn preprocessing với model; trong cross-validation, pipeline được fit lại bên trong từng fold.
- Chọn metric theo mục tiêu và chi phí sai lầm; accuracy không đủ khi lớp mất cân bằng.
- Giữ nguyên thứ tự, tên và kiểu cột lúc dự đoán dữ liệu mới. Lưu toàn pipeline thay vì chỉ lưu estimator cuối.
- Ví dụ hướng đến scikit-learn stable hiện hành. Hãy kiểm tra tài liệu theo phiên bản đã cài nếu API có khác biệt.

## Tài liệu chính thức

- [Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html)
