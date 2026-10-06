# 04 · Biến đổi dữ liệu và tạo feature

Các ví dụ bên dưới giả sử đã chạy `import numpy as np`, `import pandas as pd` và có DataFrame `df` với các cột được nhắc tới.

## 1. Tạo, sửa và xóa cột

```python
df["revenue"] = df["price"] * df["quantity"]
df["discounted"] = df["revenue"] * (1 - df["discount_rate"])
df["city"] = df["city"].str.strip().str.title()
df = df.drop(columns=["temporary"])
```

Gán cột mới bằng phép tính vector hóa. Tránh vòng lặp Python cho từng dòng khi toán tử Series giải quyết được.

### `assign`: chuỗi tạo cột dễ đọc

```python
df = df.assign(
    revenue=lambda x: x["price"] * x["quantity"],
    log_revenue=lambda x: np.log1p(x["price"] * x["quantity"]),
)
```

Trả DataFrame mới. Lambda cho phép cột sau dùng cột được tạo trước trong cùng `assign`.

## 2. Ánh xạ giá trị: `map` và `replace`

```python
df["gender_code"] = df["gender"].map({"Female": 0, "Male": 1})
df["tier"] = df["score"].map(lambda x: "high" if x >= 80 else "standard")
```

`map(dict)` đổi từng giá trị trong một Series; khóa không có trong dict thành missing. Nếu muốn giữ nguyên giá trị không được map, cân nhắc `replace(dict)`.

```python
df["status"] = df["status"].replace({"done": "complete", "wip": "in_progress"})
```

## 3. Điều kiện: `where`, `mask`, `np.where`

```python
df["passed"] = np.where(df["score"] >= 50, "pass", "fail")
df["score_valid"] = df["score"].where(df["score"].between(0, 100))
df["score_capped"] = df["score"].clip(lower=0, upper=100)
```

`np.where(condition, x, y)` chọn x nếu đúng và y nếu sai. `Series.where(condition)` giữ giá trị nơi điều kiện đúng, thay nơi sai bằng NA hoặc `other=`. `mask` ngược logic: thay nơi điều kiện đúng.

```python
df["status"] = df["status"].mask(df["score"] < 50, "needs_review")
df["band"] = np.select(
    [df["score"] < 50, df["score"] < 80],
    ["low", "mid"],
    default="high",
)
```

`np.select` đánh giá danh sách điều kiện theo thứ tự; trường hợp đầu tiên đúng được chọn.

## 4. `apply`: khi cần hàm tùy chỉnh

```python
df["full_name"] = df[["first_name", "last_name"]].apply(
    lambda row: f"{row['first_name']} {row['last_name']}", axis=1
)
```

`axis=1` gọi hàm từng dòng (thường chậm hơn phép toán vector hóa); `axis=0` áp dụng theo cột. Trước khi dùng `apply`, thử `.str`, `.dt`, toán tử Series hoặc `np.where`.

```python
df["row_total"] = df[["q1", "q2", "q3", "q4"]].sum(axis=1, min_count=1)
df["z_score"] = df.groupby("group")["value"].transform(lambda s: (s - s.mean()) / s.std())
```

`min_count=1` giữ NA nếu cả hàng không có giá trị hợp lệ thay vì biến thành 0.

## 5. Chuẩn hóa và biến đổi số

```python
df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
df["minmax"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
df["zscore"] = (df["x"] - df["x"].mean()) / df["x"].std()
```

Đây là công thức minh họa; nếu min=max hoặc std=0, mẫu số bằng 0. Trong ML, hãy fit scaler trên tập train rồi áp dụng lên validation/test để tránh leakage.

```python
denom = df["x"].max() - df["x"].min()
df["minmax"] = 0 if denom == 0 else (df["x"] - df["x"].min()) / denom
```

## 6. Gộp dữ liệu text và binning

```python
df["full_name"] = df["first"].fillna("") + " " + df["last"].fillna("")
df["age_group"] = pd.cut(df["age"], bins=[0, 18, 35, 60, np.inf], labels=["child", "young", "adult", "senior"])
df["spend_band"] = pd.qcut(df["spend"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
```

`cut` dùng ngưỡng cố định; `qcut` chia theo phân vị. Ranh giới mặc định cần hiểu kỹ (`right=True` mặc định). Nếu nhiều giá trị giống nhau khiến phân vị trùng, `duplicates="drop"` giảm số nhóm.

## 7. Thống kê theo nhóm để tạo cột

```python
df["customer_mean"] = df.groupby("customer_id")["amount"].transform("mean")
df["share_of_group"] = df["amount"] / df.groupby("city")["amount"].transform("sum")
```

`transform` trả kết quả có cùng số dòng và căn chỉnh với index ban đầu, rất phù hợp tạo feature theo nhóm.

## 8. Gán có điều kiện tránh chained assignment

```python
df.loc[df["score"] < 0, "score"] = np.nan
df.loc[df["city"] == "Hanoi", "region"] = "North"
```

Dùng `.loc[mask, column] = value` cho cập nhật có điều kiện. Tránh `df[df["score"] < 0]["score"] = ...`, vốn là chained assignment và dễ không cập nhật đúng DataFrame mong muốn.

## Tham khảo

- [pandas: Essential basic functionality](https://pandas.pydata.org/docs/user_guide/basics.html)
- [pandas: GroupBy transform](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [NumPy: conditional selection](https://numpy.org/doc/stable/reference/routines.logic.html)
