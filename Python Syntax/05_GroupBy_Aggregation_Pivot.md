# 05 · GroupBy, tổng hợp và đổi hình dạng bảng

Các ví dụ giả sử đã chạy `import pandas as pd` và có DataFrame `df` với các cột được nhắc tới.

## 1. `groupby` + một phép tổng hợp

```python
sales_by_city = df.groupby("city")["sales"].sum()
sales_by_city = df.groupby("city", as_index=False)["sales"].sum()
```

`as_index=False` giữ khóa nhóm thành cột, tiện nối bảng hoặc xuất file. Mặc định nhóm được sắp theo khóa; dùng `sort=False` để giữ thứ tự xuất hiện khi phù hợp.

## 2. Tổng hợp nhiều phép: `agg`

```python
summary = df.groupby("city").agg(
    orders=("order_id", "nunique"),
    revenue=("sales", "sum"),
    average_order=("sales", "mean"),
    latest=("date", "max"),
).reset_index()
```

Named aggregation `output_name=(source_column, function)` tạo tên cột dễ đọc.

```python
summary = df.groupby("city")["sales"].agg(["count", "sum", "mean", "median", "min", "max"])
```

## 3. Nhiều khóa nhóm

```python
monthly = df.groupby(["city", "month"], as_index=False).agg(revenue=("sales", "sum"))
```

Nhiều khóa tạo nhóm kết hợp. Với categorical, `observed` kiểm soát nhóm category chưa xuất hiện; hành vi mặc định có thể phụ thuộc phiên bản pandas, nên đặt rõ nếu điều này ảnh hưởng báo cáo.

## 4. `transform`: giữ nguyên số dòng

```python
df["city_avg"] = df.groupby("city")["sales"].transform("mean")
df["city_rank"] = df.groupby("city")["sales"].rank(ascending=False, method="dense")
```

Khác `agg` (rút gọn mỗi nhóm thành dòng), `transform` trả một giá trị cho mỗi dòng gốc.

## 5. `filter`: giữ/bỏ cả nhóm

```python
large_cities = df.groupby("city").filter(lambda group: len(group) >= 10)
```

Giữ mọi dòng của nhóm thỏa điều kiện. Nếu cần lọc theo số lượng nhanh, tạo count bằng `transform("size")` rồi lọc.

## 6. Pivot: biến giá trị thành cột

```python
wide = df.pivot(index="date", columns="product", values="sales")
```

`pivot` yêu cầu mỗi tổ hợp `index`-`columns` duy nhất. Nếu trùng, sẽ báo lỗi; tổng hợp trùng bằng `pivot_table`.

```python
wide = df.pivot_table(
    index="date", columns="product", values="sales",
    aggfunc="sum", fill_value=0, margins=True,
)
```

`aggfunc` mặc định là mean; đặt tường minh `sum`, `mean`, `count` hoặc hàm khác. `margins=True` thêm tổng biên. Kết quả có thể có MultiIndex cột; dùng `wide.columns = wide.columns.map(str)` hoặc làm phẳng nếu cần.

## 7. Unpivot: `melt`

```python
long = wide.reset_index().melt(
    id_vars="date", var_name="product", value_name="sales"
)
```

Chuyển nhiều cột đo lường thành hai cột tên biến và giá trị, phù hợp chuẩn hóa bảng wide sang tidy/long.

## 8. Bảng chéo: `crosstab`

```python
counts = pd.crosstab(df["city"], df["product"])
shares = pd.crosstab(df["city"], df["product"], normalize="index")
```

`normalize="index"` ra tỷ trọng theo hàng; `normalize="columns"` theo cột; `normalize="all"` theo tổng toàn bảng.

## 9. Đếm và xếp hạng

```python
df["city_count"] = df.groupby("city")["order_id"].transform("count")
df["product_frequency"] = df["product"].map(df["product"].value_counts())
df["rank"] = df["sales"].rank(method="dense", ascending=False)
```

`count` đếm non-null; `size` đếm dòng kể cả giá trị thiếu trong cột được chọn. `nunique` đếm số giá trị riêng biệt.

## Tham khảo

- [pandas: GroupBy guide](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [pandas: Reshaping and pivot tables](https://pandas.pydata.org/docs/user_guide/reshaping.html)
- [DataFrame.pivot_table API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html)
