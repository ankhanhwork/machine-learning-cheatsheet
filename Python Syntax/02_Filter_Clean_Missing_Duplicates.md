# 02 · Lọc, làm sạch, missing và dòng trùng

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"name": [" An ", "Binh", "Binh", None], "city": ["HN", "HCM", "HCM", "HN"], "score": [8, -1, -1, np.nan]})
```

## 1. Nhận diện giá trị thiếu

```python
df.isna()                     # mask True/False từng ô
df.isna().sum()               # số ô thiếu theo cột
df.isna().mean().mul(100)     # tỷ lệ thiếu (%)
df.notna().sum()
df[df["score"].isna()]
```

Pandas nhận diện `None`, `NaN`, `NaT`, `pd.NA` là thiếu tùy dtype. Chuỗi rỗng `""`, dấu gạch `"-"`, `"N/A"` thường **không** tự động là null nếu lúc nhập không cấu hình `na_values`.

## 2. Điền missing: `fillna`

```python
df["score"] = df["score"].fillna(0)
df["city"] = df["city"].fillna("unknown")
df = df.fillna({"score": 0, "city": "unknown"})
df["score"] = df["score"].fillna(df["score"].median())
```

Điền từng cột giúp áp dụng giá trị phù hợp theo ý nghĩa. Median bền hơn mean khi dữ liệu có ngoại lệ; với biến phân loại có thể dùng mode, nhưng mode cần xử lý trường hợp cột rỗng.

```python
mode_value = df["city"].mode(dropna=True)
if not mode_value.empty:
    df["city"] = df["city"].fillna(mode_value.iloc[0])
```

Điền theo giá trị trước/sau (chuỗi thời gian):

```python
df = df.sort_values("date")
df["price"] = df["price"].ffill()  # lấy giá trị non-null gần nhất trước đó
df["price"] = df["price"].bfill()  # lấy giá trị non-null gần nhất sau đó
```

`ffill`/`bfill` có thể làm rò thông tin tương lai nếu dùng sai trong bài toán dự báo. Xác định quy tắc theo nghiệp vụ.

## 3. Loại dòng/cột thiếu: `dropna`

```python
df.dropna()                              # bỏ dòng có ít nhất một ô thiếu
df.dropna(subset=["customer_id"])        # chỉ yêu cầu khóa không thiếu
df.dropna(subset=["score", "city"], how="all")
df.dropna(axis=1, how="all")            # bỏ cột toàn null
df.dropna(thresh=3)                      # giữ dòng có >=3 giá trị non-null
```

`how="any"` là mặc định; `how="all"` chỉ bỏ khi mọi ô trong phạm vi đều thiếu. Gán kết quả lại để thay đổi `df`.

## 4. Dòng trùng: `duplicated` và `drop_duplicates`

```python
df.duplicated().sum()                       # số dòng trùng hoàn toàn
df[df.duplicated(keep=False)]               # xem tất cả thành viên nhóm trùng
df.drop_duplicates()                         # giữ lần xuất hiện đầu
df.drop_duplicates(subset=["customer_id"])  # trùng theo khóa
df.drop_duplicates(subset=["id"], keep="last")
```

`keep="first"` mặc định; `keep="last"` giữ bản cuối; `keep=False` đánh dấu/bỏ tất cả bản thuộc nhóm trùng. Nếu muốn giữ bản cập nhật mới nhất, hãy sort theo timestamp trước.

## 5. Thay toàn bộ giá trị: `replace`

```python
df["score"] = df["score"].replace(-1, np.nan)
df = df.replace({"N/A": pd.NA, "n/a": pd.NA, "?": pd.NA})
df["status"] = df["status"].replace({"done": "completed", "in progress": "active"})
```

`Series.replace` / `DataFrame.replace` thay khớp **giá trị ô** (hoặc regex nếu bật). Dùng dictionary để nhiều map cùng lúc. Khác `Series.str.replace`, xem [03: text](03_Data_Types_Dates_Text.md).

```python
df["phone"] = df["phone"].replace(r"[^0-9]", "", regex=True)
```

Regex này bỏ mọi ký tự không phải chữ số. Luôn kiểm tra kiểu cột và giữ `+` nếu mã quốc gia có ý nghĩa.

## 6. Lọc dòng không hợp lệ

```python
valid = df.loc[df["age"].ge(0) & df["age"].le(120)]
valid = df.loc[df["email"].str.contains("@", na=False)]
```

Phương thức so sánh tương đương: `.gt` `>`; `.ge` `>=`; `.lt` `<`; `.le` `<=`; `.eq` `==`; `.ne` `!=`.

## 7. Kiểm tra sau làm sạch

```python
assert df["customer_id"].notna().all(), "customer_id đang thiếu"
print(df.isna().sum())
print("exact duplicates:", df.duplicated().sum())
print(df.shape)
```

`assert` dừng chương trình nếu điều kiện không đạt, rất hữu ích trước bước join hoặc export.

## Tham khảo

- [pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [pandas: Duplicate labels / duplicate data](https://pandas.pydata.org/docs/user_guide/indexing.html#duplicate-labels)
- [DataFrame.replace API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html)
- [DataFrame.drop_duplicates API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)
