# 03 · Tiền xử lý feature

Scikit-learn transformer có cùng quy trình: `fit` học cách biến đổi từ train, `transform` áp dụng cách đó lên train/validation/test.

## 1. Missing values

```python
from sklearn.impute import SimpleImputer

numeric_imputer = SimpleImputer(strategy="median")
X_num_train = numeric_imputer.fit_transform(X_train)
X_num_test = numeric_imputer.transform(X_test)
```

Các chiến lược phổ biến: `mean`, `median`, `most_frequent`, `constant`. Median bền hơn mean khi số liệu lệch/outlier; cần chọn theo ý nghĩa cột. Có thể thêm cờ missing bằng `add_indicator=True`.

## 2. Scaling cho cột số

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()  # mean 0, std 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- `StandardScaler`: thường hữu ích cho linear model, SVM, KNN và PCA.
- `MinMaxScaler`: đưa giá trị về khoảng thường là `[0, 1]`; nhạy với outlier.
- `RobustScaler`: dùng median/IQR, ít nhạy outlier hơn.
- Cây quyết định và rừng cây thường không cần scale feature.

## 3. Encode biến phân loại

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

one_hot = OneHotEncoder(handle_unknown="ignore")
X_cat_train = one_hot.fit_transform(X_cat_train_raw)
X_cat_test = one_hot.transform(X_cat_test_raw)
```

`OneHotEncoder` tạo cột nhị phân, không áp đặt thứ tự giả giữa các nhãn. `handle_unknown="ignore"` cho phép category mới ở dữ liệu sau này (các cột one-hot của category đó là 0).

`OrdinalEncoder` chỉ dùng khi thứ tự category có ý nghĩa hoặc model/cách mã hóa phù hợp:

```python
ordinal = OrdinalEncoder(
    categories=[["low", "medium", "high"]],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
```

## 4. Biến đổi phân phối và vector hóa text

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transform = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
```

Chỉ dùng `log1p` khi giá trị không âm và phép biến đổi có ý nghĩa. Với văn bản thô, `TfidfVectorizer` thường biến chuỗi thành ma trận đặc trưng:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_text_train = vectorizer.fit_transform(text_train)
X_text_test = vectorizer.transform(text_test)
```

## 5. Tạo feature và giảm chiều

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

Polynomial features có thể tăng mạnh số cột; dùng trong pipeline và cross-validation, cân nhắc scale và regularization.

## 6. Lỗi thường gặp

- `could not convert string to float`: encode cột phân loại hoặc loại cột không phải feature.
- Khác số cột giữa train/test: không encode hai tập độc lập; fit encoder trên train rồi transform test.
- NaN không được hỗ trợ bởi estimator: thêm imputer phù hợp.
- Leakage: không fit scaler, imputer, encoder, selector hoặc PCA trên toàn bộ dữ liệu trước split.
