# 08 · Preprocessing recipes, quy trình mẫu và sửa lỗi

## 1. Quy trình CSV từ đầu đến cuối

```python
import numpy as np
import pandas as pd

# 1) Nạp đúng kiểu ban đầu: ID nên là string để giữ số 0 đầu
df = pd.read_csv(
    "orders.csv",
    dtype={"customer_id": "string", "order_id": "string"},
    na_values=["", "NA", "N/A", "NULL"],
)

# 2) Chuẩn hóa tên cột
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# 3) Làm sạch kiểu và text
df["customer_id"] = df["customer_id"].str.strip()
df["order_date"] = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
df["amount"] = pd.to_numeric(df["amount"].str.replace(",", "", regex=False), errors="coerce")
df["status"] = df["status"].astype("string").str.strip().str.lower()

# 4) Quy tắc missing / sentinel
df["amount"] = df["amount"].replace(-1, np.nan)
df = df.dropna(subset=["order_id", "customer_id"])

# 5) Loại đơn trùng, giữ bản cập nhật cuối
df = df.sort_values("updated_at").drop_duplicates("order_id", keep="last")

# 6) Tạo feature
df["year_month"] = df["order_date"].dt.to_period("M").astype("string")
df["is_large_order"] = df["amount"].ge(1000).fillna(False)

# 7) Kiểm tra rồi lưu
assert df["order_id"].is_unique
print(df.shape, df.dtypes, df.isna().sum())
df.to_csv("orders_clean.csv", index=False, encoding="utf-8-sig")
```

Điều chỉnh sentinel, định dạng ngày và quy tắc loại trùng theo định nghĩa dữ liệu thật. Đoạn mã trên là mẫu quy trình, không phải quy tắc nghiệp vụ phổ quát.

## 2. Ghép bảng lookup an toàn

```python
orders["customer_id"] = orders["customer_id"].astype("string").str.strip()
customers["customer_id"] = customers["customer_id"].astype("string").str.strip()

# Bảng lookup cần một dòng cho mỗi customer_id
duplicates = customers[customers["customer_id"].duplicated(keep=False)]
if not duplicates.empty:
    raise ValueError("customers có customer_id trùng; cần quyết định cách xử lý trước khi merge")

enriched = orders.merge(
    customers[["customer_id", "segment", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)
print(enriched["_merge"].value_counts(dropna=False))
```

`many_to_one` phát biểu mỗi đơn hàng có thể cùng khách hàng nhưng lookup chỉ có một dòng mỗi khách. Nếu `left_only` lớn, kiểm tra khoảng trắng, dtype, số 0 đầu, viết hoa/thường và thời điểm snapshot.

## 3. Gộp nhiều file cùng schema

```python
from pathlib import Path

paths = sorted(Path("monthly").glob("*.csv"))
frames = [pd.read_csv(path) for path in paths]
all_months = pd.concat(frames, ignore_index=True)
```

Nếu cần giữ nguồn:

```python
frames = {path.stem: pd.read_csv(path) for path in paths}
all_months = pd.concat(frames, names=["source", "row"])
```

Trước concat, so sánh `columns` và `dtypes` giữa các file; concat outer có thể âm thầm thêm cột thiếu và điền null.

## 4. Gộp nhóm và tạo feature tổng hợp

```python
customer_stats = (
    orders.groupby("customer_id", as_index=False)
    .agg(
        order_count=("order_id", "nunique"),
        total_spend=("amount", "sum"),
        average_order=("amount", "mean"),
        last_order=("order_date", "max"),
    )
)
```

Sau đó có thể merge `customer_stats` vào bảng khách hàng bằng `validate="one_to_one"` hoặc `one_to_many` phù hợp.

## 5. Mẫu xử lý cột số lẫn ký hiệu

```python
raw = df["revenue"].astype("string").str.strip()
normalized = raw.str.replace("₫", "", regex=False).str.replace(",", "", regex=False)
parsed = pd.to_numeric(normalized, errors="coerce")

bad = df.loc[raw.notna() & parsed.isna(), "revenue"]
print("Giá trị không parse được:\n", bad.drop_duplicates().head(20))
df["revenue"] = parsed
```

Chỉ bỏ ký hiệu đã biết. Không xóa mọi ký tự không phải số nếu dữ liệu có dấu âm, dấu thập phân hoặc quy ước locale.

## 6. Mẫu kiểm tra dữ liệu trước/sau

```python
def profile(frame):
    return pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "missing": frame.isna().sum(),
        "missing_pct": (frame.isna().mean() * 100).round(1),
        "unique": frame.nunique(dropna=True),
    }).sort_values("missing_pct", ascending=False)

print(profile(df))
```

## 7. Lỗi thường gặp và hướng xử lý

### `ValueError` hoặc merge không match

1. Kiểm tra tên khóa và dtype hai bên: `left[key].dtype`, `right[key].dtype`.
2. Chuẩn hóa khoảng trắng/chữ hoa; với ID dạng mã, chuyển sang string trước khi mất số 0 đầu.
3. Kiểm tra khóa null và duplicate.
4. Dùng `indicator=True` để thấy tỷ lệ `left_only`/`both`.
5. Xác nhận đúng `how` và `on`; kiểm tra `validate` để phát hiện many-to-many ngoài dự kiến.

### Merge làm số dòng tăng bất ngờ

```python
print("before:", len(left), len(right))
print("duplicate keys:", right[key].duplicated().sum())
result = left.merge(right, on=key, how="left", validate="many_to_one")
```

Nếu `validate` báo lỗi, không bỏ qua để cho chạy; xác định quy tắc chọn/aggregate dòng lookup trước.

### `astype(int)` báo lỗi hoặc có missing

```python
df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
```

Kiểm tra các giá trị bị biến thành missing trước khi tiếp tục; không chuyển thẳng sang integer thường khi có NA.

### `.dt` chỉ dùng được với datetime

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
```

`dayfirst=True` không thay thế format rõ ràng khi biết chính xác quy ước. Kiểm tra số dòng thành NaT sau parse.

### `KeyError: 'column'`

```python
print(df.columns.tolist())
df.columns = df.columns.str.strip()
```

Tên cột có thể chứa khoảng trắng, khác hoa/thường hoặc ký tự BOM. Đọc lại CSV với encoding phù hợp nếu có ký tự lạ.

### `SettingWithCopyWarning` / chained assignment

```python
subset = df.loc[df["score"] > 0].copy()
subset.loc[:, "score"] = subset["score"].astype(float)
```

Nếu muốn sửa bản lọc độc lập, thêm `.copy()` rồi gán qua `.loc`.

### Concat tạo nhiều missing hoặc index kỳ lạ

- Dùng `ignore_index=True` nếu cần index mới khi xếp dòng.
- So sánh `columns` trước khi ghép; cân nhắc chuẩn hóa schema.
- Với `axis=1`, index căn theo nhãn; `reset_index(drop=True)` chỉ làm nếu chắc chắn thứ tự dòng hai bên tương ứng.

## 8. Checklist preprocessing trước khi dùng mô hình

- [ ] Xác nhận grain của bảng: một dòng là một giao dịch, khách hàng hay ngày?
- [ ] ID được giữ nguyên dạng string nếu có số 0 đầu.
- [ ] Các giá trị placeholder đã chuyển thành missing theo quy tắc nguồn.
- [ ] Dtypes hợp lý; ngày giờ parse được; số lỗi parse đã được xem.
- [ ] Missing được thống kê và xử lý có lý do.
- [ ] Duplicates được kiểm tra theo khóa nghiệp vụ, không chỉ duplicate toàn dòng.
- [ ] Join có khóa rõ ràng, cardinality được xác minh, số dòng trước/sau được ghi nhận.
- [ ] Các phép tính không chia cho 0; ngoại lệ/sentinel được xử lý có chủ đích.
- [ ] Không dùng thông tin tương lai hoặc target để tạo feature (data leakage).
- [ ] Scaler/imputer/encoder được fit trên tập train, không fit toàn bộ dữ liệu trước khi split.
- [ ] File xuất không chứa cột index ngoài ý muốn; tên và dtype cột được kiểm tra lại.

## Tham khảo chính thức

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Merge, join and concat](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [IO tools](https://pandas.pydata.org/docs/user_guide/io.html)
