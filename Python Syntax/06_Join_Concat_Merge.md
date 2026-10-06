# 06 · Ghép bảng: merge, join, concat

Quy tắc nhớ nhanh: **`merge` ghép theo khóa**, **`join` ghép theo index**, **`concat` nối các bảng theo trục**. Ví dụ dùng `import pandas as pd` và dữ liệu minh họa được tạo bên dưới.

## Dữ liệu ví dụ

```python
import pandas as pd

orders = pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [10, 11, 99], "amount": [100, 200, 50]})
customers = pd.DataFrame({"customer_id": [10, 11, 12], "name": ["An", "Binh", "Chi"]})
```

## 1. `merge`: SQL-style join theo cột khóa

```python
result = orders.merge(customers, on="customer_id", how="left")
```

**Output khái quát:** giữ 3 đơn hàng; đơn hàng customer 99 vẫn có, `name` là missing; các cột `order_id`, `customer_id`, `amount`, `name` được ghép ngang.

### Các loại join

```python
inner = orders.merge(customers, on="customer_id", how="inner")  # chỉ khóa khớp
left = orders.merge(customers, on="customer_id", how="left")    # giữ mọi dòng trái
right = orders.merge(customers, on="customer_id", how="right")  # giữ mọi dòng phải
outer = orders.merge(customers, on="customer_id", how="outer")  # hợp khóa hai bên
```

| `how` | Dòng nào được giữ? | Dùng thường khi |
|---|---|---|
| `inner` | chỉ có khóa ở cả hai | cần bản ghi ghép thành công |
| `left` | mọi dòng bảng bên trái | làm giàu bảng chính, không bỏ quan sát |
| `right` | mọi dòng bảng bên phải | giữ toàn bộ danh mục tham chiếu |
| `outer` | mọi khóa cả hai bên | đối soát, tìm khóa không khớp |

### Tên khóa khác nhau / nhiều khóa

```python
out = transactions.merge(accounts, left_on="acct_id", right_on="account_id", how="left")
out = a.merge(b, on=["customer_id", "month"], how="left")
```

Nếu `on` bỏ trống, pandas có thể chọn các tên cột chung làm khóa. Hãy khai báo `on`/`left_on`/`right_on` rõ ràng để tránh join nhầm.

### Index làm khóa

```python
out = left.merge(right, left_on="customer_id", right_index=True, how="left")
out = left.merge(right, left_index=True, right_index=True, how="outer")
```

### Kiểm soát quan hệ khóa và tên cột trùng

```python
out = orders.merge(customers, on="customer_id", how="left", validate="many_to_one", indicator=True)
print(out["_merge"].value_counts())
```

`validate` phát hiện sai cardinality: `one_to_one`, `one_to_many`, `many_to_one`, `many_to_many`. `indicator=True` thêm `_merge` với `left_only`, `right_only`, `both` để chẩn đoán match.

```python
out = left.merge(right, on="id", suffixes=("_source", "_lookup"))
```

Tên cột không phải khóa bị trùng được gắn suffix; đặt tên giúp phân biệt nguồn.

## 2. Kiểm tra lỗi phổ biến của merge

```python
print(orders["customer_id"].dtype, customers["customer_id"].dtype)
print(orders["customer_id"].isna().sum(), customers["customer_id"].isna().sum())
print(orders["customer_id"].duplicated().sum(), customers["customer_id"].duplicated().sum())
```

Khóa cùng giá trị nhưng khác dtype (ví dụ `int64` so với string) có thể gây lỗi hoặc không khớp. Căn chỉnh kiểu trước:

```python
orders["customer_id"] = orders["customer_id"].astype("string")
customers["customer_id"] = customers["customer_id"].astype("string")
```

**Join nhân dòng:** nếu khóa xuất hiện nhiều lần cả hai bảng, many-to-many tạo tích Descartes trong mỗi khóa. Kiểm tra `duplicated(subset=key)` và `validate`.

## 3. `join`: ghép theo index

```python
customers_by_id = customers.set_index("customer_id")
out = orders.join(customers_by_id, on="customer_id", how="left")
```

`DataFrame.join` mặc định ghép index của bảng phải với index bảng trái; `on=` chỉ định cột của bảng trái đối chiếu với index bảng phải.

## 4. `concat`: nối nhiều DataFrame

### Xếp dòng (append rows)

```python
q1 = pd.DataFrame({"id": [1], "sales": [100]})
q2 = pd.DataFrame({"id": [2], "sales": [120]})
all_q = pd.concat([q1, q2], ignore_index=True)
```

**Output:** hai dòng, cột `id`, `sales`, index mới 0 và 1. `ignore_index=True` bỏ index cũ và đánh lại index.

Nếu tên cột lệch, mặc định `join="outer"` tạo hợp cột và chèn NA nơi thiếu. `join="inner"` chỉ giữ cột giao nhau:

```python
combined = pd.concat([a, b], axis=0, join="outer", ignore_index=True)
```

### Ghép ngang (append columns)

```python
wide = pd.concat([features, labels], axis=1)
```

Ghép theo **index**, không tự ghép theo khóa ID. Đảm bảo index thẳng hàng hoặc merge theo ID rõ ràng. `axis=1, join="inner"` chỉ giữ index chung.

### Nhiều file, tạo một lần

```python
frames = [pd.read_csv(path) for path in paths]
all_data = pd.concat(frames, ignore_index=True)
```

Thu thập bảng vào list rồi concat một lần; gọi concat lặp liên tục có thể tạo bản sao dữ liệu không cần thiết.

### Giữ nguồn bằng `keys`

```python
out = pd.concat({"jan": january, "feb": february}, names=["month", "old_index"])
```

Thêm MultiIndex giúp truy xuất nguồn; dùng `reset_index()` khi cần chuyển thành cột thông thường.

## 5. Cách chọn thao tác

| Ý định | Dùng |
|---|---|
| Thêm thông tin từ bảng lookup theo `id` | `merge(..., on="id", how="left")` |
| Gộp file tháng có cột giống nhau | `pd.concat([...], ignore_index=True)` |
| Gắn Series/DataFrame đã cùng index | `pd.concat([...], axis=1)` |
| Bảng phụ đã đặt khóa làm index | `.join(..., on="key")` |

## Tham khảo

- [pandas: Merge, join, concatenate and compare](https://pandas.pydata.org/docs/user_guide/merging.html)
- [pandas.merge API](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
- [pandas.concat API](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)
