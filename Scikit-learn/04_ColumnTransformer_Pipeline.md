# 04 · ColumnTransformer và Pipeline

`ColumnTransformer` áp dụng quy trình khác nhau theo nhóm cột; `Pipeline` xếp preprocessing nối tiếp với estimator. Kết hợp chúng để có một đối tượng duy nhất cho CV, tuning và dự đoán.

## 1. Ví dụ dữ liệu hỗn hợp

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_features = ["age", "income"]
categorical_features = ["city", "plan"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

## 2. Fit và đánh giá

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = df[numeric_features + categorical_features]
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
```

Khi `model.fit` chạy, các bước preprocessing chỉ học từ `X_train`. Khi `model.predict(X_test)` chạy, pipeline tự áp dụng đúng các biến đổi đã fit.

## 3. Thay estimator và truy cập bước

```python
from sklearn.ensemble import RandomForestClassifier

model.set_params(classifier=RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
))
model.fit(X_train, y_train)

print(model.named_steps["preprocess"])
```

Tham số lồng nhau dùng dạng `tên_bước__tên_tham_số`, ví dụ `classifier__C` hoặc `preprocess__num__imputer__strategy`.

## 4. Feature names sau biến đổi

```python
model.fit(X_train, y_train)
feature_names = model.named_steps["preprocess"].get_feature_names_out()
print(feature_names[:10])
```

`ColumnTransformer` có thể tạo ma trận sparse từ One-Hot Encoding. Nhiều estimator hỗ trợ sparse; không ép sang array dense nếu dữ liệu rộng, vì có thể tốn nhiều bộ nhớ.

## 5. Ghi nhớ

- Fit toàn pipeline trên train; giữ nguyên pipeline đã fit để dự đoán.
- Dùng danh sách cột rõ ràng để tránh đưa nhầm ID, target hoặc cột thời gian không hợp lệ.
- `remainder="drop"` là mặc định; dùng `remainder="passthrough"` khi đã chủ động muốn giữ các cột còn lại.
- Thứ tự cột lúc inference nên giống dữ liệu huấn luyện; DataFrame giúp pipeline chọn theo tên cột.
