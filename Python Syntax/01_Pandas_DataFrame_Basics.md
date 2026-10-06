# 01 · pandas DataFrame: đọc, xem, chọn và lọc

## 1. Khởi tạo và đọc dữ liệu

```python
import pandas as pd

df = pd.DataFrame({"name": ["An", "Binh"], "age": [20, 21]})
```

`pd.DataFrame(data)` tạo bảng hai chiều có index và tên cột. `pd.Series(data)` tạo một cột có nhãn.

### Đọc CSV

```python
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", encoding="utf-8", sep=",", na_values=["", "NA", "N/A"])
df = pd.read_csv("data.csv", usecols=["id", "date", "sales"], dtype={"id": "string"})
df = pd.read_csv("data.csv", parse_dates=["date"])
```

**Thường dùng:** `sep`, `encoding`, `usecols`, `dtype`, `na_values`, `parse_dates`, `nrows`, `skiprows`. File CSV dùng dấu `;` thì đặt `sep=";"`. Nếu mã tiếng Việt lỗi, thử đúng encoding do nguồn file cung cấp, ví dụ `utf-8-sig`.

### Excel và các định dạng khác

```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1", usecols="A:D")
book = pd.read_excel("data.xlsx", sheet_name=None)  # dict tên sheet -> DataFrame
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json")
```

Excel có thể cần engine thích hợp được cài sẵn. Khi đọc nhiều sheet bằng `sheet_name=None`, `book["Sheet1"]` lấy một sheet.

## 2. Kiểm tra nhanh cấu trúc

```python
df.head()             # 5 dòng đầu
df.head(10)
df.tail(3)
df.sample(5, random_state=42)
df.shape               # (n_rows, n_columns)
df.columns.tolist()
df.index
df.dtypes
df.info()
df.describe()          # thống kê cột số
df.describe(include="all")
df.nunique()           # số giá trị riêng biệt mỗi cột
df.memory_usage(deep=True)
```

`info()` in số non-null và dtype; `describe()` cho thống kê nhanh. `sample` hữu ích khi dữ liệu đã được sắp xếp theo thời gian và `head()` không đại diện. Đặt `random_state` để lấy mẫu có thể lặp lại.

## 3. Chọn cột

```python
df["sales"]                       # Series
df[["customer_id", "sales"]]     # DataFrame
df.loc[:, ["customer_id", "sales"]]
```

Một cặp ngoặc vuông trả Series; danh sách tên cột trả DataFrame. Nếu cần DataFrame một cột, dùng `df[["sales"]]`.

## 4. Chọn dòng: `.loc` và `.iloc`

```python
df.loc[0]                         # dòng có nhãn index 0
df.loc[0:4, ["name", "age"]]     # lát nhãn: đầu/cuối đều được tính
df.iloc[0]                        # dòng ở vị trí thứ nhất
df.iloc[0:5, 0:2]                 # vị trí: cuối lát không tính
df.iloc[-1]                       # dòng cuối
df.at[3, "age"]                   # một giá trị theo nhãn
df.iat[0, 1]                      # một giá trị theo vị trí
```

`.loc` là nhãn, `.iloc` là vị trí nguyên bắt đầu từ 0. Với index ngày hoặc index không liên tục, `.loc[0]` không có nghĩa là dòng đầu.

## 5. Lọc dòng bằng điều kiện

```python
adults = df[df["age"] >= 18]
subset = df.loc[(df["age"] >= 18) & (df["city"] == "Hanoi"), ["name", "age"]]
```

Toán tử điều kiện pandas là `&` (AND), `|` (OR), `~` (NOT); bọc **từng điều kiện** bằng ngoặc. Không dùng `and/or` cho Series.

```python
df[df["city"].isin(["Hanoi", "HCMC"])]
df[df["age"].between(18, 25, inclusive="both")]
df[df["name"].str.contains("nguyen", case=False, na=False)]
df[df["date"].isna()]
```

`isin` kiểm tra thuộc tập; `between` lọc khoảng; `na=False` quy định chuỗi null không được tính là match.

### `query()` cho điều kiện dễ đọc

```python
adults = df.query("age >= 18 and city == 'Hanoi'")
minimum = 18
adults = df.query("age >= @minimum")
```

`@variable` tham chiếu biến Python bên ngoài. Tên cột có dấu cách hoặc ký tự đặc biệt cần backtick: `df.query("`net sales` > 0")`.

## 6. Sắp xếp và index

```python
df.sort_values("sales", ascending=False)
df.sort_values(["city", "sales"], ascending=[True, False], na_position="last")
df.sort_index()
df.reset_index(drop=True)
df.set_index("customer_id")
```

Các phép trên trả kết quả mới, nên gán lại. `reset_index(drop=True)` tạo index 0..n-1 và bỏ index cũ. `set_index` thường hữu ích khi khóa là định danh, nhưng `merge` theo cột thường dễ đọc hơn.

## 7. Đổi tên cột, xóa cột/dòng

```python
df = df.rename(columns={"old_name": "new_name", "Amt": "amount"})
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
df = df.drop(columns=["unused", "comment"])
df = df.drop(index=[0, 1])
```

Chuẩn hóa tên cột giúp tránh lỗi do khoảng trắng/case. Xóa bằng `columns=` và `index=` làm ý định rõ ràng.

## 8. Xuất dữ liệu

```python
df.to_csv("clean.csv", index=False, encoding="utf-8-sig")
df.to_excel("clean.xlsx", index=False, sheet_name="clean")
df.to_parquet("clean.parquet", index=False)
```

Nếu không muốn cột index thừa trong file, dùng `index=False`. Parquet thường gọn và giữ dtype tốt hơn CSV, nhưng cần thư viện engine tương ứng.

## 9. Mẫu khám phá ban đầu

```python
print("shape:", df.shape)
print("duplicates:", df.duplicated().sum())
print("missing:\n", df.isna().sum().sort_values(ascending=False))
print("types:\n", df.dtypes)
print(df.head(3))
```

**Output:** kích thước dữ liệu, số dòng trùng hoàn toàn, số missing theo cột, kiểu dữ liệu và ba dòng mẫu. Chạy đoạn này ngay sau khi đọc file để xác định bước làm sạch.

## Tham khảo chính thức

- [pandas: IO tools](https://pandas.pydata.org/docs/user_guide/io.html)
- [pandas: Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [pandas: Essential basic functionality](https://pandas.pydata.org/docs/user_guide/basics.html)
