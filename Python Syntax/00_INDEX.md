# Python Syntax: pandas & NumPy cho preprocessing

> **Mục tiêu:** tra đúng cú pháp trong vài giây, hiểu dữ liệu sẽ thay đổi ra sao, rồi sao chép ví dụ để điều chỉnh cho DataFrame của bạn.

## Bắt đầu nhanh

```python
import numpy as np
import pandas as pd

df = pd.read_csv("input.csv")
print(df.shape)       # (số_dòng, số_cột)
print(df.head())      # xem vài dòng
print(df.dtypes)      # kiểu từng cột
```

### Tìm theo việc cần làm

| Nhu cầu | Mở file | Cú pháp nên tìm |
|---|---|---|
| Đọc file, xem bảng, chọn dòng/cột | [01_Pandas_DataFrame_Basics.md](01_Pandas_DataFrame_Basics.md) | `read_csv`, `head`, `loc`, `iloc`, `query` |
| Xử lý null, dòng trùng, giá trị sai | [02_Filter_Clean_Missing_Duplicates.md](02_Filter_Clean_Missing_Duplicates.md) | `isna`, `fillna`, `dropna`, `drop_duplicates`, `replace` |
| Đổi kiểu, xử lý ngày và text | [03_Data_Types_Dates_Text.md](03_Data_Types_Dates_Text.md) | `astype`, `to_numeric`, `to_datetime`, `.str` |
| Tạo cột, ánh xạ, chuẩn hóa giá trị | [04_Transform_Create_Features.md](04_Transform_Create_Features.md) | `assign`, `map`, `apply`, `where`, `clip` |
| Groupby, pivot, tổng hợp | [05_GroupBy_Aggregation_Pivot.md](05_GroupBy_Aggregation_Pivot.md) | `groupby`, `agg`, `transform`, `pivot_table`, `melt` |
| Nối bảng theo khóa hoặc ghép file | [06_Join_Concat_Merge.md](06_Join_Concat_Merge.md) | `merge`, `join`, `concat`, `validate` |
| Mảng, mask, phép tính vector hóa | [07_Numpy_Array_Recipes.md](07_Numpy_Array_Recipes.md) | `array`, `where`, `select`, `concatenate`, `nan` |
| Ghép thành quy trình và sửa lỗi | [08_Preprocessing_Recipes_Troubleshooting.md](08_Preprocessing_Recipes_Troubleshooting.md) | pipeline, checklist, lỗi thường gặp |

## Ba thao tác hay nhầm

### `merge`: ghép ngang theo khóa giống SQL

```python
orders = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
```

Mỗi dòng đơn hàng giữ lại; thông tin khách hàng được ghép theo `customer_id`. Nếu khóa phía bảng khách hàng không duy nhất, `validate` giúp phát hiện thay vì âm thầm nhân bản đơn hàng.

### `concat`: xếp bảng cùng cấu trúc hoặc ghép cột

```python
all_months = pd.concat([jan, feb, mar], ignore_index=True)  # xếp dòng
with_score = pd.concat([features, score], axis=1)            # ghép cột theo index
```

`axis=0` là thêm dòng; `axis=1` là thêm cột. Ghép nhiều phần tử trong một lần, không lặp concat trong vòng lặp.

### `replace`: thay giá trị cũ bằng giá trị mới

```python
df["status"] = df["status"].replace({"N/A": pd.NA, "done": "completed"})
df["amount"] = df["amount"].replace(-1, np.nan)
```

`replace` tìm giá trị theo nội dung ô. Nó khác `.str.replace(...)`, vốn thay một đoạn ký tự bên trong chuỗi.

## Cách đọc ví dụ

- Tất cả ví dụ đều ghi import cần thiết và có thể chạy độc lập hoặc thay tên cột cho dữ liệu thật.
- Các dòng `# Output:` diễn tả kết quả chính, không phải lúc nào cũng in toàn bộ bảng.
- Pandas căn chỉnh theo **index/nhãn** ở nhiều phép toán. NumPy chủ yếu làm việc theo **vị trí/shape**.
- Trước khi ghép hoặc ghi đè cột, kiểm tra `shape`, `dtypes`, `isna().sum()` và khóa join.
- Dùng `df = df...` để thấy rõ kết quả mới. Nhiều hàm không sửa DataFrame gốc nếu không gán lại.

## Quy ước phiên bản và nguồn

Ví dụ hướng đến API pandas hiện hành và NumPy 2.x; các API cũ có thể khác. Các hàm được dẫn tới tài liệu chính thức để kiểm tra chi tiết tham số và thay đổi theo phiên bản.

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [pandas API reference](https://pandas.pydata.org/docs/reference/index.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy API reference](https://numpy.org/doc/stable/reference/index.html)
