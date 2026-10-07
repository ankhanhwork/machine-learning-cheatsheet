# 02 · Chia dữ liệu và cross-validation

## 1. Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # thường dùng cho phân loại để giữ tỷ lệ lớp
)
```

Tập train dùng để học; validation/CV dùng để chọn model và tham số; test được giữ kín để ước lượng hiệu quả cuối cùng. Không lựa chọn model dựa đi dựa lại trên test.

Với hồi quy, thường bỏ `stratify`. Dữ liệu theo thời gian cần chia theo thời gian, không shuffle ngẫu nhiên:

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
```

Nếu nhiều hàng thuộc cùng một khách hàng, bệnh nhân hay cửa hàng, dùng group split để các nhóm không lọt cả vào train và validation.

## 2. K-fold cross-validation

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
result = cross_validate(
    LogisticRegression(max_iter=1000),
    X_train, y_train,
    cv=cv,
    scoring={"accuracy": "accuracy", "f1": "f1_macro"},
    return_train_score=True,
)

print(result["test_f1"].mean(), result["test_f1"].std())
```

`StratifiedKFold` phù hợp phân loại; `KFold` thường dùng hồi quy. Với dữ liệu lệch lớp nghiêm trọng hoặc nhóm/thời gian, chọn splitter đúng cấu trúc dữ liệu.

## 3. Vì sao split trước preprocessing?

Scaler, imputer và feature selection đều học thông tin từ dữ liệu. Nếu fit chúng trước khi chia, thông tin từ test đã ảnh hưởng đến quá trình học, khiến đánh giá lạc quan. Gói chúng trong `Pipeline` để mỗi fold chỉ fit preprocessing trên phần train của fold đó.

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
# Truyền model nguyên pipeline vào cross_validate hoặc search CV.
```

## 4. Checklist chia tập

- Tách `X` và `y` trước; loại cột target khỏi `X`.
- Dùng `random_state` để kết quả chia có thể lặp lại.
- Dùng `stratify=y` cho phân loại thông thường.
- Đảm bảo không có bản ghi trùng hoặc cùng một thực thể ở cả hai tập nếu điều đó gây rò rỉ.
- Với chuỗi thời gian, validation phải nằm sau dữ liệu train theo thời gian.
