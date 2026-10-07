# 01 · Dữ liệu và estimator API

## 1. Cài đặt và import

```bash
python -m pip install scikit-learn pandas
```

Tên package khi cài là `scikit-learn`; tên import thường là `sklearn`.

```python
import sklearn
print(sklearn.__version__)
```

## 2. X và y

- `X`: ma trận feature, thường có shape `(n_samples, n_features)`; với pandas, dùng DataFrame để giữ tên cột.
- `y`: target, thường có shape `(n_samples,)`.
- Mỗi hàng của `X` phải tương ứng cùng một quan sát với phần tử cùng vị trí trong `y`.

```python
import pandas as pd

df = pd.read_csv("customers.csv")
target = "churned"
X = df.drop(columns=target)
y = df[target]

print(X.shape, y.shape)
print(X.dtypes)
print(y.value_counts(dropna=False))
```

Không đưa ID ngẫu nhiên hay thông tin chỉ có sau kết quả vào feature mà không có lý do nghiệp vụ. Kiểm tra missing, trùng lặp và nhãn trước khi học.

## 3. Estimator API

Estimator là đối tượng học từ dữ liệu qua `fit`. Transformer có thêm `transform`; predictor có `predict`.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit học mean/std, sau đó transform
```

Trong supervised learning, model nhận cả feature và target:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)
labels = clf.predict(X_scaled[:3])
probabilities = clf.predict_proba(X_scaled[:3])
```

## 4. Bộ dữ liệu mẫu tích hợp

```python
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target
print(X.head())
print(dataset.target_names)
```

`load_*` dùng để học và thử nghiệm. Trong bài toán thật, thay bằng nguồn dữ liệu của dự án và xác định target/features theo định nghĩa nghiệp vụ.

## 5. Các phương thức thường gặp

| Phương thức | Tác dụng |
|---|---|
| `fit(X, y)` | Học tham số từ dữ liệu; transformer unsupervised có thể chỉ cần `fit(X)` |
| `transform(X)` | Biến đổi feature bằng tham số đã học |
| `fit_transform(X)` | Fit rồi transform train; thường tiện và hiệu quả cho transformer |
| `predict(X)` | Trả nhãn hoặc giá trị dự đoán |
| `predict_proba(X)` | Trả xác suất lớp nếu estimator hỗ trợ |
| `score(X, y)` | Metric mặc định của estimator; cần biết rõ metric đó là gì |

Không gọi `fit_transform` trên test hoặc dữ liệu production. Hãy fit trên train và chỉ transform dữ liệu mới.
