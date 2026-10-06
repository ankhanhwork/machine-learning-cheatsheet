# 03 · Đổi kiểu dữ liệu, ngày tháng và chuỗi

```python
import pandas as pd

df = pd.DataFrame({"amount": ["1,200", "850", "bad"], "date": ["01/02/2025", "15/03/2025", None], "name": [" An ", "BÌNH", "Chi"]})
```

## 1. Kiểm tra dtype

```python
df.dtypes
df["amount"].dtype
df.select_dtypes(include="number")
df.select_dtypes(include=["object", "string"])
```

Kiểu object có thể chứa chuỗi hoặc đối tượng trộn lẫn; đừng giả định cột số đang là số chỉ vì nhìn thấy chữ số.

## 2. `astype`: chuyển kiểu có kiểm soát

```python
df["customer_id"] = df["customer_id"].astype("string")
df["quantity"] = df["quantity"].astype("int64")
df = df.astype({"city": "category", "active": "boolean"})
```

`astype` báo lỗi nếu giá trị không thể chuyển, thường phù hợp khi dữ liệu phải hợp lệ. Integer thông thường (`int64`) không chứa null; pandas nullable integer là `Int64` (chữ I viết hoa).

```python
df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
```

## 3. Chuỗi số: `to_numeric`

```python
df["amount"] = df["amount"].str.replace(",", "", regex=False)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
```

`errors="coerce"` đổi giá trị lỗi thành NaN; kiểm tra lại để không che mất lỗi nguồn:

```python
raw = df["amount"]
parsed = pd.to_numeric(raw, errors="coerce")
bad_rows = df.loc[raw.notna() & parsed.isna()]
df["amount"] = parsed
```

`errors="raise"` dừng tại dữ liệu không hợp lệ; `downcast` có thể chọn kiểu số nhỏ hơn khi phù hợp.

## 4. Ngày tháng: `to_datetime`

```python
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
```

Khai báo `format` giúp tránh hiểu nhầm ngày/tháng và thường nhanh hơn. Ví dụ `01/02/2025` là 1 tháng 2 với `%d/%m/%Y`, nhưng là 2 tháng 1 với `%m/%d/%Y`.

| Mã | Ý nghĩa | Ví dụ |
|---|---|---|
| `%Y` | năm 4 chữ số | 2025 |
| `%m` | tháng 2 chữ số | 03 |
| `%d` | ngày 2 chữ số | 15 |
| `%H:%M:%S` | giờ phút giây | 14:05:00 |
| `%Y-%m-%d` | ngày ISO phổ biến | 2025-03-15 |

### Tách đặc trưng ngày

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.dayofweek  # Mon=0 ... Sun=6
df["quarter"] = df["date"].dt.quarter
df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
```

`dt` là accessor cho dữ liệu datetime. Chuỗi chưa parse datetime sẽ không có `.dt`.

## 5. Chuỗi: accessor `.str`

```python
s = df["name"].astype("string")
df["name_clean"] = s.str.strip().str.lower()
df["name_clean"] = s.str.replace(r"\s+", " ", regex=True).str.strip()
df["has_a"] = s.str.contains("a", case=False, na=False)
df["email_domain"] = df["email"].str.split("@").str[-1]
```

Thao tác chuỗi áp dụng từng phần tử theo vector hóa. `na=False` trả False tại ô thiếu khi tìm kiếm.

```python
parts = df["full_name"].str.split(" ", n=1, expand=True)
df[["first", "rest"]] = parts
df["code"] = df["code"].str.slice(0, 3)
```

`expand=True` trả DataFrame nhiều cột. Khi gán, số cột bên trái phải khớp số cột trả về.

## 6. Boolean, category và nullable dtypes

```python
df["active"] = df["active"].map({"yes": True, "no": False}).astype("boolean")
df["segment"] = df["segment"].astype("category")
df["id"] = df["id"].astype("string")
```

Nullable pandas dtypes (`Int64`, `boolean`, `string`) biểu diễn missing tốt hơn dtype NumPy thuần trong nhiều trường hợp. `category` hữu ích cho nhãn lặp lại và có thể tiết kiệm bộ nhớ.

## 7. Lưu ý dấu thập phân và phân cách hàng nghìn

```python
df = pd.read_csv("europe.csv", decimal=",", thousands=".")
```

Ví dụ văn bản `1.234,50` có thể biểu diễn 1234.50 với thiết lập trên. Định dạng tùy hệ thống nguồn; không xóa dấu phẩy hàng loạt nếu nó là dấu thập phân.

## Tham khảo

- [pandas: Working with text data](https://pandas.pydata.org/docs/user_guide/text.html)
- [pandas: Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [to_numeric API](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html)
- [to_datetime API](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
- [pandas: Nullable integer data type](https://pandas.pydata.org/docs/user_guide/integer_na.html)
