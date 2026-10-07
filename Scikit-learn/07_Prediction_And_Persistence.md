# 07 · Dự đoán dữ liệu mới và lưu model

## 1. Dự đoán bằng pipeline đã fit

```python
import pandas as pd

new_customers = pd.DataFrame([
    {"age": 34, "income": 42000, "city": "Hanoi", "plan": "basic"},
    {"age": 51, "income": 78000, "city": "Da Nang", "plan": "premium"},
])

labels = best_model.predict(new_customers)
probabilities = best_model.predict_proba(new_customers)
```

Dữ liệu mới cần đúng schema feature: tên cột, kiểu dữ liệu và ý nghĩa giống lúc train. Không truyền target hoặc ID nếu không dùng khi huấn luyện.

## 2. Gắn kết quả vào bản ghi

```python
result = new_customers.copy()
result["prediction"] = labels
result["probability_class_1"] = probabilities[:, 1]
```

Kiểm tra `best_model.classes_` để xác nhận cột xác suất tương ứng với nhãn nào:

```python
print(best_model.classes_)
```

## 3. Lưu và nạp toàn pipeline

```python
import joblib

joblib.dump(best_model, "customer_churn_pipeline.joblib")
loaded_model = joblib.load("customer_churn_pipeline.joblib")
predictions = loaded_model.predict(new_customers)
```

Chỉ nạp file model từ nguồn tin cậy: các định dạng pickle/joblib có thể thực thi mã khi load. Ghi lại phiên bản Python, scikit-learn và dependency để tái tạo môi trường.

```python
import sklearn
import sys

print(sys.version)
print(sklearn.__version__)
```

## 4. Checklist trước khi dùng prediction

- Model và toàn bộ preprocessing đã fit chỉ trên train.
- Dữ liệu inference có đủ cột; kiểu và đơn vị được kiểm tra.
- Không có feature chỉ biết được sau thời điểm cần dự đoán.
- Nhãn lớp và thứ tự cột xác suất đã xác nhận.
- Threshold được chọn trên validation, không điều chỉnh theo kết quả test.
- Theo dõi chất lượng dữ liệu và hiệu năng sau triển khai; model có thể giảm chất lượng khi dữ liệu đổi.

## 5. Phác thảo batch prediction

```python
import pandas as pd
import joblib

model = joblib.load("customer_churn_pipeline.joblib")
incoming = pd.read_csv("new_customers.csv")
required = ["age", "income", "city", "plan"]
missing = set(required) - set(incoming.columns)
if missing:
    raise ValueError(f"Thiếu cột đầu vào: {sorted(missing)}")

incoming["prediction"] = model.predict(incoming[required])
incoming.to_csv("predictions.csv", index=False)
```
