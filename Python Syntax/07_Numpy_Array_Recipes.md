# 07 · NumPy: array, điều kiện và phép tính nhanh

```python
import numpy as np

x = np.array([10, 20, 30, 40])
```

NumPy xử lý mảng đồng nhất, tính theo vector và broadcast. Pandas có nhãn/index; NumPy chủ yếu căn theo vị trí.

Một số ví dụ chuyển đổi ở cuối giả sử có DataFrame pandas tên `df`; import bằng `import pandas as pd` nếu cần.

## 1. Tạo mảng và xem cấu trúc

```python
np.array([1, 2, 3])
np.zeros((2, 3))
np.ones((2, 2))
np.arange(0, 10, 2)          # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)         # 5 điểm đều từ 0 tới 1
x.shape                     # (4,)
x.ndim                      # số chiều
x.size                      # số phần tử
x.dtype                     # kiểu phần tử
```

## 2. Indexing, slicing và reshape

```python
x[0]                        # phần tử đầu
x[-1]                       # phần tử cuối
x[1:3]                      # vị trí 1,2; điểm cuối không tính
matrix = np.arange(6).reshape(2, 3)
matrix[0, :]                # hàng đầu
matrix[:, 1]                # cột thứ hai
matrix.reshape(-1)          # trải thành một chiều
```

Slicing cơ bản thường tạo view; sửa slice có thể tác động vào mảng gốc. `.copy()` nếu cần bản độc lập.

## 3. Tính vector hóa và broadcasting

```python
x + 5
x * 2
x ** 2
np.sqrt(x)
np.log1p(x)
matrix + np.array([100, 200, 300])
```

Broadcasting cho phép tính giữa shape tương thích mà không cần viết vòng lặp. Không tương thích shape sẽ báo lỗi.

## 4. Boolean mask và điều kiện

```python
mask = x >= 25
x[mask]                     # array([30, 40])
x[(x >= 20) & (x < 40)]
np.where(x >= 25, "high", "low")
np.select([x < 20, x < 35], ["low", "mid"], default="high")
```

Kết hợp điều kiện NumPy bằng `&`, `|`, `~`; dùng ngoặc quanh từng phép so sánh.

## 5. Missing và giá trị không hữu hạn

```python
a = np.array([1.0, np.nan, 3.0])
np.isnan(a)
np.nanmean(a)                # 2.0
np.nansum(a)                 # 4.0
np.isfinite(a)               # loại cả NaN và +/-inf
np.nan_to_num(a, nan=0.0, posinf=999)
```

`NaN` thông thường là float; integer NumPy không chứa NaN. Trong pandas có thể dùng `pd.NA` và nullable dtypes.

## 6. Tổng hợp và thống kê

```python
np.sum(x)
np.mean(x)
np.median(x)
np.min(x), np.max(x)
np.std(x, ddof=1)            # độ lệch chuẩn mẫu
np.percentile(x, [25, 50, 75])
np.unique(x, return_counts=True)
```

Với mảng nhiều chiều, `axis=0` gộp theo hàng để trả mỗi cột; `axis=1` gộp theo cột để trả mỗi hàng.

```python
matrix.sum(axis=0)
matrix.mean(axis=1)
```

## 7. Ghép mảng: `concatenate`, `stack`

```python
a = np.array([1, 2])
b = np.array([3, 4])
np.concatenate([a, b])       # [1, 2, 3, 4]
np.stack([a, b], axis=0)     # shape (2, 2)
```

`concatenate` nối theo một trục đã có; kích thước các trục còn lại phải khớp. `stack` thêm trục mới.

## 8. Chuyển pandas ↔ NumPy

```python
values = df["sales"].to_numpy()
matrix = df[["age", "income"]].to_numpy(dtype="float64")
df["scaled"] = (df["sales"] / 100).to_numpy()
```

Sau khi lấy NumPy array, nhãn index bị bỏ. Gán array vào DataFrame căn theo **vị trí**, nên đảm bảo thứ tự dòng không đổi. Nếu cần căn theo nhãn, giữ Series pandas.

```python
df["sales"] = np.where(df["sales"] < 0, np.nan, df["sales"])
```

`np.where` thường trả ndarray và có thể làm thay đổi dtype; kiểm tra `df.dtypes` sau khi gán.

## Tham khảo

- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy routines](https://numpy.org/doc/stable/reference/routines.html)
