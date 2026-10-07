# Hướng dẫn Feature Engineering cho dự báo rủi ro nợ xấu

> Mỗi đoạn code nằm ngay dưới mô tả feature tương ứng. Code được lấy trực tiếp từ notebook.

# Feature Engineering — Predicting Customer Bad Debt Risk

## Đọc trước khi chạy

Mỗi feature là **một cột mới**. Ví dụ `df\["FE\_LOAN\_TO\_INCOME"] = ...` nghĩa là thêm cột vào bảng khách hàng. Hãy đổi tên cột bên phải cho khớp dữ liệu của bạn. Cần một dòng mỗi khách hàng (hoặc mỗi khoản vay) tại ngày dự đoán. Không có cột tương ứng thì bỏ qua feature.

**Cực kỳ quan trọng:** chỉ dùng thông tin đã biết tại ngày dự đoán. Không dùng khoản trả, dư nợ hay trạng thái được ghi nhận sau ngày đó. `TARGET` không tham gia tạo feature.

## Cell 1 — Nạp dữ liệu và xem cột

Thay đường dẫn bằng file của bạn. Nếu bảng đã có sẵn trong notebook pipeline thì dùng chính DataFrame đó.

```python
import pandas as pd
import numpy as np

df = pd.read\_csv("your\_customer\_data.csv")
print(df.shape)
print(df.columns.tolist())
df.head()
```

## 1\. Tỷ lệ khoản vay trên thu nhập — `FE\_LOAN\_TO\_INCOME`

**Cần:** số tiền khoản vay và thu nhập năm. **Tạo:** `loan\_amount / annual\_income`.
**Ý nghĩa:** khoản vay lớn cỡ nào so với khả năng tạo thu nhập; tỷ lệ cao có thể báo gánh nặng lớn. Thu nhập 0/thiếu tạo giá trị không xác định, nên để NaN và xử lý trong pipeline.

Đổi hai tên cột trong ngoặc vuông theo dataset.

```python
df\["FE\_LOAN\_TO\_INCOME"] = df\["loan\_amount"] / df\["annual\_income"].replace(0, np.nan)
```

## 2\. Gánh khoản trả hàng tháng — `FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME`

**Cần:** khoản trả mỗi tháng và thu nhập năm. **Tạo:** monthly installment / (annual income / 12).
**Ý nghĩa:** phần thu nhập tháng cần dành để trả khoản vay này. Cao nghĩa là ngân sách còn lại ít hơn. Nếu dataset thu nhập đã theo tháng, bỏ `/ 12`.

```python
df\["FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME"] = (
    df\["monthly\_installment"] / (df\["annual\_income"].replace(0, np.nan) / 12)
)
```

## 3\. Tổng nợ trên thu nhập — `FE\_DEBT\_TO\_INCOME`

**Cần:** tổng dư nợ hiện có và thu nhập năm. **Tạo:** total debt / annual income.
**Ý nghĩa:** mức nợ tổng thể so với thu nhập. Chỉ dùng nếu `total\_debt` được đo tại hoặc trước ngày dự đoán; đừng cộng khoản nợ phát sinh sau đó.

```python
df\["FE\_DEBT\_TO\_INCOME"] = df\["total\_debt"] / df\["annual\_income"].replace(0, np.nan)
```

## 4\. Mức dùng hạn mức — `FE\_CREDIT\_UTILIZATION`

**Cần:** dư nợ thẻ/tín dụng quay vòng và hạn mức. **Tạo:** balance / credit limit.
**Ý nghĩa:** phần hạn mức đang sử dụng. Ví dụ 0.8 là 80%. Nếu vượt 1, đừng tự cắt bỏ trước khi hiểu cách ghi nhận hạn mức/dư nợ.

```python
df\["FE\_CREDIT\_UTILIZATION"] = df\["credit\_card\_balance"] / df\["credit\_limit"].replace(0, np.nan)
```

## Nếu có bảng lịch sử riêng

Các cell feature 5–8 dùng bảng `payments`; feature 9–10 dùng bảng `monthly`. Nếu dataset chỉ có một bảng, bỏ qua các phần này. Thay đường dẫn bằng file thực tế. Tên bảng phải có đúng các cột dùng ở bên dưới.

## 5\. Số ngày trả trễ — `FE\_DAYS\_LATE`

**Cần:** ngày đến hạn và ngày trả thực tế cho từng kỳ trả. Bảng này thường có nhiều dòng mỗi khách hàng.
**Tạo:** ngày trả trừ ngày đến hạn; số dương là trễ, số âm đặt về 0.
**Ý nghĩa:** đo mức trễ cụ thể. Với kỳ chưa trả, ngày trả bị trống không có nghĩa là đúng hạn. Hãy tính số ngày từ ngày đến hạn đến ngày dự đoán (cutoff), sau đó mới tổng hợp.

```python
payments\["due\_date"] = pd.to\_datetime(payments\["due\_date"])
payments\["payment\_date"] = pd.to\_datetime(payments\["payment\_date"])
payments\["FE\_DAYS\_LATE"] = (payments\["payment\_date"] - payments\["due\_date"]).dt.days.clip(lower=0)
```

## 6\. Tỷ lệ kỳ trả bị trễ — `FE\_LATE\_PAYMENT\_RATE`

**Cần:** `FE\_DAYS\_LATE` trên từng kỳ và ID khách hàng.
**Tạo:** số kỳ có trễ / tổng số kỳ của khách hàng.
**Ý nghĩa:** tỷ lệ trả trễ trong lịch sử quan sát. Phải chỉ tổng hợp các kỳ đã đến hạn trước ngày dự đoán.

```python
payments\["FE\_WAS\_LATE"] = (payments\["FE\_DAYS\_LATE"] > 0).astype(int)
late\_rate = payments.groupby("customer\_id")\["FE\_WAS\_LATE"].mean().rename("FE\_LATE\_PAYMENT\_RATE")
df = df.merge(late\_rate, on="customer\_id", how="left", validate="one\_to\_one")
```

## 7\. Tỷ lệ trả trễ từ 30 ngày — `FE\_LATE\_30\_RATE`

**Cần:** ngày trễ của từng kỳ và ID khách hàng. **Tạo:** cờ `days late >= 30`, rồi lấy trung bình theo khách hàng.
**Ý nghĩa:** tập trung vào trễ nghiêm trọng hơn một vài ngày. Nếu nghiệp vụ dùng ngưỡng khác (ví dụ 60/90 ngày), đổi số 30.

```python
payments\["FE\_LATE\_30"] = (payments\["FE\_DAYS\_LATE"] >= 30).astype(int)
late\_30\_rate = payments.groupby("customer\_id")\["FE\_LATE\_30"].mean().rename("FE\_LATE\_30\_RATE")
df = df.merge(late\_30\_rate, on="customer\_id", how="left", validate="one\_to\_one")
```

## 8\. Số tiền trả thiếu — `FE\_PAYMENT\_SHORTFALL\_RATE`

**Cần:** số tiền đến hạn và số tiền thực trả cho mỗi kỳ. **Tạo:** (scheduled - actual), tối thiểu bằng 0; chia cho scheduled để thành tỷ lệ thiếu.
**Ý nghĩa:** phát hiện trả một phần hoặc thiếu tiền, kể cả khi có ngày thanh toán. Tổng hợp trung bình theo khách hàng.

```python
payments\["FE\_SHORTFALL"] = (payments\["scheduled\_amount"] - payments\["actual\_amount"]).clip(lower=0)
payments\["FE\_SHORTFALL\_RATE"] = payments\["FE\_SHORTFALL"] / payments\["scheduled\_amount"].replace(0, np.nan)
shortfall\_rate = payments.groupby("customer\_id")\["FE\_SHORTFALL\_RATE"].mean().rename("FE\_PAYMENT\_SHORTFALL\_RATE")
df = df.merge(shortfall\_rate, on="customer\_id", how="left", validate="one\_to\_one")
```

## 9\. Đỉnh mức dùng hạn mức theo thời gian — `FE\_UTILIZATION\_MAX`

**Cần:** bảng nhiều dòng theo khách hàng và tháng, có balance, limit. **Tạo:** utilization mỗi tháng rồi lấy giá trị lớn nhất theo khách hàng.
**Ý nghĩa:** bắt được tháng khách hàng sử dụng hạn mức rất cao dù trung bình cả kỳ thấp. Bảng tháng phải cắt ở ngày dự đoán.

```python
monthly\["FE\_UTIL"] = monthly\["balance"] / monthly\["credit\_limit"].replace(0, np.nan)
util\_max = monthly.groupby("customer\_id")\["FE\_UTIL"].max().rename("FE\_UTILIZATION\_MAX")
df = df.merge(util\_max, on="customer\_id", how="left", validate="one\_to\_one")
```

## 10\. Biến động mức dùng hạn mức — `FE\_UTILIZATION\_STD`

**Cần:** cùng bảng tháng ở feature 9. **Tạo:** độ lệch chuẩn utilization theo khách hàng.
**Ý nghĩa:** độ biến động qua thời gian; cao có thể cho thấy sử dụng tín dụng thất thường. Chỉ có một tháng thì kết quả NaN là bình thường.

```python
util\_std = monthly.groupby("customer\_id")\["FE\_UTIL"].std().rename("FE\_UTILIZATION\_STD")
df = df.merge(util\_std, on="customer\_id", how="left", validate="one\_to\_one")
```

## 11\. Số lượng lịch sử trả — `FE\_PAYMENT\_HISTORY\_COUNT`

**Cần:** một dòng mỗi kỳ trả và ID khách hàng. **Tạo:** đếm số kỳ đủ điều kiện trước cutoff.
**Ý nghĩa:** phân biệt người không có lịch sử với người có lịch sử tốt. Không có lịch sử không đồng nghĩa ít rủi ro; nên để riêng count và cờ no-history.

```python
payment\_count = payments.groupby("customer\_id").size().rename("FE\_PAYMENT\_HISTORY\_COUNT")
df = df.merge(payment\_count, on="customer\_id", how="left", validate="one\_to\_one")
df\["FE\_NO\_PAYMENT\_HISTORY"] = df\["FE\_PAYMENT\_HISTORY\_COUNT"].isna().astype(int)
```

## Cuối cùng: kiểm tra các cột vừa tạo

Một số merge có thể tạo NaN cho khách hàng không có lịch sử. Giữ missing/no-history có chủ đích; điền giá trị trong preprocessing pipeline sau khi chia train/validation.

```python
feature\_cols = \[c for c in df.columns if c.startswith("FE\_")]
df\[\["customer\_id"] + feature\_cols].head()
```

## 12\. Số khoản tín dụng đang mở — `FE\_OPEN\_ACCOUNT\_COUNT`

**Cần:** bảng tài khoản tín dụng, một dòng mỗi tài khoản; cột trạng thái `account\_status` và `customer\_id`.
**Tạo:** lọc tài khoản đang mở rồi đếm theo khách hàng.
**Ý nghĩa:** tổng mức phơi nhiễm/độ phức tạp nghĩa vụ tín dụng. Đây là ứng viên tham khảo từ credit scoring; quan hệ có thể phi tuyến, cần kiểm chứng.

```python
open\_accounts = accounts\[accounts\["account\_status"] == "open"]
open\_count = open\_accounts.groupby("customer\_id").size().rename("FE\_OPEN\_ACCOUNT\_COUNT")
df = df.merge(open\_count, on="customer\_id", how="left", validate="one\_to\_one")
```

## 13\. Thời gian từ lần trễ gần nhất — `FE\_DAYS\_SINCE\_LAST\_LATE`

**Cần:** `payments` có ngày đến hạn/ngày trả và ngày quan sát `observation\_date` trong bảng khách hàng.
**Tạo:** lấy ngày trả gần nhất có trễ; trừ khỏi ngày quan sát.
**Ý nghĩa:** phân biệt vi phạm cũ với vi phạm gần đây. Nếu chưa từng trễ, giá trị là NaN; có thể thêm cờ chưa từng trễ riêng. Chỉ dùng kỳ thanh toán đã đến hạn trước ngày quan sát.

```python
late\_events = payments\[payments\["FE\_DAYS\_LATE"] > 0].copy()
last\_late = late\_events.groupby("customer\_id")\["payment\_date"].max().rename("\_last\_late\_date")
df = df.merge(last\_late, on="customer\_id", how="left", validate="one\_to\_one")
df\["FE\_DAYS\_SINCE\_LAST\_LATE"] = (
    pd.to\_datetime(df\["observation\_date"]) - pd.to\_datetime(df\["\_last\_late\_date"])
).dt.days
df = df.drop(columns="\_last\_late\_date")
```

## 14\. Tỷ lệ trả trễ trong 6 tháng gần nhất — `FE\_LATE\_RATE\_6M`

**Cần:** lịch trả có `due\_date`, cờ `FE\_WAS\_LATE`, `customer\_id`; cùng ngày quan sát.
**Tạo:** giữ các kỳ đến hạn trong 6 tháng trước observation date, rồi tính trung bình cờ trễ.
**Ý nghĩa:** phản ứng nhanh hơn biến lifetime nếu hành vi khách hàng đang xấu đi/cải thiện. Đổi 6 thành 3/12 để so sánh.

```python
payments\["due\_date"] = pd.to\_datetime(payments\["due\_date"])
# Gắn observation date cho mỗi khách hàng trước khi lọc cửa sổ lịch sử.
payments\_window = payments.merge(df\[\["customer\_id", "observation\_date"]], on="customer\_id", how="inner")
cutoff = pd.to\_datetime(payments\_window\["observation\_date"])
start\_6m = cutoff - pd.DateOffset(months=6)
recent\_6m = payments\_window\[(payments\_window\["due\_date"] <= cutoff) \& (payments\_window\["due\_date"] > start\_6m)]
late\_rate\_6m = recent\_6m.groupby("customer\_id")\["FE\_WAS\_LATE"].mean().rename("FE\_LATE\_RATE\_6M")
df = df.merge(late\_rate\_6m, on="customer\_id", how="left", validate="one\_to\_one")
```

## 15\. Xu hướng trả trễ — `FE\_LATE\_RATE\_TREND\_6M\_VS\_PRIOR\_6M`

**Cần:** cùng bảng payments và cờ `FE\_WAS\_LATE`.
**Tạo:** tỷ lệ trễ 6 tháng gần nhất trừ tỷ lệ 6 tháng liền trước.
**Ý nghĩa:** số dương cho thấy tỷ lệ trễ tăng gần đây; số âm là giảm. Đây là giả thuyết hữu ích từ phân tích hành vi theo thời gian, cần kiểm tra độ ổn định khi có đủ lịch sử.

```python
start\_prior\_6m = cutoff - pd.DateOffset(months=12)
prior\_6m = payments\_window\[(payments\_window\["due\_date"] <= start\_6m) \& (payments\_window\["due\_date"] > start\_prior\_6m)]
late\_rate\_prior\_6m = prior\_6m.groupby("customer\_id")\["FE\_WAS\_LATE"].mean().rename("\_late\_rate\_prior\_6m")
trend = late\_rate\_6m.to\_frame().join(late\_rate\_prior\_6m, how="outer")
trend\["FE\_LATE\_RATE\_TREND\_6M\_VS\_PRIOR\_6M"] = trend\["FE\_LATE\_RATE\_6M"] - trend\["\_late\_rate\_prior\_6m"]
df = df.merge(trend\[\["FE\_LATE\_RATE\_TREND\_6M\_VS\_PRIOR\_6M"]], on="customer\_id", how="left", validate="one\_to\_one")
```

## 16\. Số yêu cầu tín dụng gần đây — `FE\_INQUIRIES\_6M`

**Cần:** bảng inquiry có ngày yêu cầu và ID khách hàng.
**Tạo:** đếm inquiry trong sáu tháng trước ngày quan sát.
**Ý nghĩa:** có thể phản ánh nhu cầu vay/đăng ký tín dụng gần đây. Nhiều inquiry trùng do một lần mua sắm có thể cần gộp theo ngày/loại; chỉ dùng nếu dữ liệu hợp lệ và được phép.

```python
inquiries\["inquiry\_date"] = pd.to\_datetime(inquiries\["inquiry\_date"])
inquiries\_window = inquiries.merge(df\[\["customer\_id", "observation\_date"]], on="customer\_id", how="inner")
inquiry\_cutoff = pd.to\_datetime(inquiries\_window\["observation\_date"])
inquiry\_start = inquiry\_cutoff - pd.DateOffset(months=6)
inquiries\_6m = inquiries\_window\[(inquiries\_window\["inquiry\_date"] <= inquiry\_cutoff) \&
                                (inquiries\_window\["inquiry\_date"] > inquiry\_start)]
inquiry\_count = inquiries\_6m.groupby("customer\_id").size().rename("FE\_INQUIRIES\_6M")
df = df.merge(inquiry\_count, on="customer\_id", how="left", validate="one\_to\_one")
```

## 17\. Tuổi lịch sử tín dụng — `FE\_CREDIT\_HISTORY\_MONTHS`

**Cần:** ngày mở tài khoản tín dụng cũ nhất, ngày quan sát.
**Tạo:** số tháng giữa ngày mở sớm nhất và observation date.
**Ý nghĩa:** lịch sử dài hơn cho thêm bằng chứng hành vi, nhưng không mặc định là rủi ro thấp. Cần có ngày mở chính xác và lịch sử đủ phạm vi.

```python
accounts\["open\_date"] = pd.to\_datetime(accounts\["open\_date"])
accounts = accounts.merge(df\[\["customer\_id", "observation\_date"]], on="customer\_id", how="inner")
first\_open = accounts.groupby("customer\_id")\["open\_date"].min().rename("\_first\_open")
df = df.merge(first\_open, on="customer\_id", how="left", validate="one\_to\_one")
df\["FE\_CREDIT\_HISTORY\_MONTHS"] = (
    (pd.to\_datetime(df\["observation\_date"]) - pd.to\_datetime(df\["\_first\_open"])).dt.days / 30.44
)
df = df.drop(columns="\_first\_open")
```

# Bổ sung từ notebook Kaggle Will Koehrsen

Các biến bên dưới dựa trên phần polynomial/domain feature engineering trong [Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction#Feature-Engineering). Ví dụ gốc dùng schema Home Credit (`AMT\_CREDIT`, `AMT\_INCOME\_TOTAL`, `AMT\_ANNUITY`, `EXT\_SOURCE\_1/2/3`, `DAYS\_BIRTH`). Chỉ chạy các feature khi bạn có cột tương đương; tên khác thì thay lại.

## 18\. Tuổi khách hàng — `FE\_AGE\_YEARS`

**Cần:** tuổi dạng ngày âm tính từ ngày sinh (`DAYS\_BIRTH`, quy ước Home Credit), hoặc ngày sinh và ngày quan sát.
**Tạo:** `-DAYS\_BIRTH / 365.25`.
**Ý nghĩa:** chuyển số ngày khó đọc thành số năm. Nếu dữ liệu có ngày sinh thật, tính tuổi tại observation date; tránh dùng tuổi hiện tại cho hồ sơ lịch sử.

```python
df\["FE\_AGE\_YEARS"] = -df\["DAYS\_BIRTH"] / 365.25
```

## 19\. Khoản vay trên thu nhập — `FE\_CREDIT\_INCOME\_RATIO`

**Cần:** số tiền tín dụng/khoản vay được đề nghị và tổng thu nhập. **Tạo:** `AMT\_CREDIT / AMT\_INCOME\_TOTAL`.
**Ý nghĩa:** gánh khoản vay so với thu nhập. Đây là biến gốc trong ví dụ Kaggle; tương tự `FE\_LOAN\_TO\_INCOME` ở trên, nên kiểm tra trùng lặp và chỉ giữ một tên rõ ràng.

```python
df\["FE\_CREDIT\_INCOME\_RATIO"] = df\["AMT\_CREDIT"] / df\["AMT\_INCOME\_TOTAL"].replace(0, np.nan)
```

## 20\. Khoản trả trên thu nhập — `FE\_ANNUITY\_INCOME\_RATIO`

**Cần:** khoản trả định kỳ `AMT\_ANNUITY` và thu nhập `AMT\_INCOME\_TOTAL`.
**Tạo:** annuity / income. **Ý nghĩa:** phần thu nhập dành cho khoản trả theo chu kỳ dữ liệu. Giữ đúng đơn vị/thời kỳ; nếu annuity theo tháng còn thu nhập theo năm thì nhân income với 1/12.

```python
df\["FE\_ANNUITY\_INCOME\_RATIO"] = df\["AMT\_ANNUITY"] / df\["AMT\_INCOME\_TOTAL"].replace(0, np.nan)
```

## 21\. Khoản vay trên khoản trả — `FE\_CREDIT\_ANNUITY\_RATIO`

**Cần:** `AMT\_CREDIT`, `AMT\_ANNUITY`. **Tạo:** credit amount / annuity.
**Ý nghĩa:** proxy số kỳ/thời hạn trả nợ khi điều kiện vay phù hợp. Không coi là số tháng chính xác nếu lãi suất, phí hoặc lịch trả thay đổi.

```python
df\["FE\_CREDIT\_ANNUITY\_RATIO"] = df\["AMT\_CREDIT"] / df\["AMT\_ANNUITY"].replace(0, np.nan)
```

## 22\. Tổng hợp điểm nguồn ngoài — `FE\_EXT\_SOURCE\_MEAN`, `FE\_EXT\_SOURCE\_STD`

**Cần:** các điểm `EXT\_SOURCE\_1`, `EXT\_SOURCE\_2`, `EXT\_SOURCE\_3` (tương tự các score bên ngoài của dataset khác).
**Tạo:** trung bình và độ lệch chuẩn theo hàng, bỏ qua điểm thiếu.
**Ý nghĩa:** mean gom tín hiệu thành một mức tổng quát; std biểu thị các nguồn đồng thuận hay khác biệt. Missing count nên xem riêng vì số điểm có sẵn giữa khách hàng có thể khác nhau.

```python
ext\_cols = \["EXT\_SOURCE\_1", "EXT\_SOURCE\_2", "EXT\_SOURCE\_3"]
df\["FE\_EXT\_SOURCE\_MEAN"] = df\[ext\_cols].mean(axis=1)
df\["FE\_EXT\_SOURCE\_STD"] = df\[ext\_cols].std(axis=1)
df\["FE\_EXT\_SOURCE\_COUNT"] = df\[ext\_cols].notna().sum(axis=1)
```

## 23\. Tương tác và phi tuyến giữa điểm ngoài/tuổi — `FE\_POLY\_\*`

**Cần:** các cột ở trên. Notebook Kaggle tạo polynomial terms tới bậc 3 từ ba điểm EXT\_SOURCE và DAYS\_BIRTH (bao gồm bình phương, tích chéo). Điều này cho phép mô hình tuyến tính học quan hệ cong/tương tác.

Code dưới dùng `PolynomialFeatures`. Fit bộ sinh feature chỉ trên train, sau đó dùng chính transformer để biến đổi validation/test; tuyệt đối không fit riêng trên validation/test. `TARGET`/label không đưa vào danh sách đầu vào.

```python
from sklearn.preprocessing import PolynomialFeatures

poly\_cols = \["EXT\_SOURCE\_1", "EXT\_SOURCE\_2", "EXT\_SOURCE\_3", "DAYS\_BIRTH"]
poly\_train\_input = train\_df\[poly\_cols].copy()
poly\_valid\_input = valid\_df\[poly\_cols].copy()
# Điền missing bằng median học từ train; giữ cùng giá trị cho validation/test.
poly\_medians = poly\_train\_input.median()
poly\_train\_input = poly\_train\_input.fillna(poly\_medians)
poly\_valid\_input = poly\_valid\_input.fillna(poly\_medians)
poly\_transformer = PolynomialFeatures(degree=3, include\_bias=False)
poly\_train\_array = poly\_transformer.fit\_transform(poly\_train\_input)
poly\_valid\_array = poly\_transformer.transform(poly\_valid\_input)
poly\_names = poly\_transformer.get\_feature\_names\_out(poly\_cols)
poly\_train\_features = pd.DataFrame(poly\_train\_array, columns=\["FE\_POLY\_" + c for c in poly\_names], index=train\_df.index)
poly\_valid\_features = pd.DataFrame(poly\_valid\_array, columns=\["FE\_POLY\_" + c for c in poly\_names], index=valid\_df.index)
train\_df = pd.concat(\[train\_df, poly\_train\_features], axis=1)
valid\_df = pd.concat(\[valid\_df, poly\_valid\_features], axis=1)
```

**Lưu ý:** Với tree boosting, polynomial expansion có thể dư thừa; thử ablation. Polynomial terms có thể làm tăng mạnh số cột và nhạy với missing/outlier. Không thêm `TARGET` vào `poly\_cols`.



Bổ sung:
**Nhóm 1. Khả năng trả nợ từ hồ sơ một bảng**

Dùng nhóm này đầu tiên khi đề có một bảng hồ sơ xin vay hoặc một snapshot khách hàng. Các cột thường là thu nhập, khoản vay đề nghị, khoản trả dự kiến và nghĩa vụ nợ hiện hữu. Đây là nhóm nhanh nhất để chạy sau baseline.

### **24. Tổng gánh trả nợ hàng tháng — FE\_TOTAL\_PAYMENT\_TO\_INCOME**

**Cần:** thu nhập tháng, tổng khoản phải trả mỗi tháng của các khoản nợ hiện hữu, khoản trả dự kiến của khoản vay mới. Nếu thu nhập theo năm, chia 12 trước khi tính.

**Tạo:** (existing\_monthly\_payment + proposed\_monthly\_payment) / monthly\_income.

**Ý nghĩa:** xem *tất cả* khoản trả hàng tháng chiếm bao nhiêu thu nhập. Ví dụ thu nhập 20 triệu, nợ cũ trả 4 triệu và khoản vay mới dự kiến trả 3 triệu thì tỷ lệ là 7/20 = 35%. Guide cũ có tỷ lệ khoản trả của **riêng khoản vay mới** trên thu nhập, nên đây là tín hiệu khác. Tên existing\_monthly\_payment thiếu chỉ được thay bằng 0 khi data dictionary nói rõ “thiếu = không có nợ cũ”.

income = pd.to\_numeric(df\["monthly\_income"], errors="coerce")  
old\_payment = pd.to\_numeric(df\["existing\_monthly\_payment"], errors="coerce")  
new\_payment = pd.to\_numeric(df\["proposed\_monthly\_payment"], errors="coerce")

total\_payment = old\_payment + new\_payment  
df\["FE\_TOTAL\_PAYMENT\_TO\_INCOME"] = total\_payment.div(income.where(income > 0))

### **25. Thu nhập còn lại sau trả nợ — FE\_RESIDUAL\_INCOME\_AFTER\_DEBT**

**Cần:** cùng ba cột ở mục 24.

**Tạo:** monthly\_income - existing\_monthly\_payment - proposed\_monthly\_payment.

**Ý nghĩa:** với ví dụ trên, còn 13 triệu đồng sau trả nợ. Đây **chưa** phải tiền còn lại sau chi phí sinh hoạt. Nếu dataset có chi phí sinh hoạt được xác minh trước khi xét vay, có thể tạo thêm một feature trừ khoản đó. Giá trị âm là tín hiệu cần kiểm tra, không tự sửa thành 0.

income = pd.to\_numeric(df\["monthly\_income"], errors="coerce")  
old\_payment = pd.to\_numeric(df\["existing\_monthly\_payment"], errors="coerce")  
new\_payment = pd.to\_numeric(df\["proposed\_monthly\_payment"], errors="coerce")

df\["FE\_RESIDUAL\_INCOME\_AFTER\_DEBT"] = income - old\_payment - new\_payment

### **26. Flagging thiếu thông tin — FE\_INCOME\_MISSING, FE\_SCORE\_MISSING**

**Cần:** cột thu nhập hoặc credit score tại thời điểm nộp hồ sơ.

**Tạo:** 1 khi giá trị thiếu hoặc không đọc được thành số, 0 khi có giá trị hợp lệ.

**Ý nghĩa:** “không có điểm tín dụng” khác “điểm tín dụng thấp”. Feature này có thể giúp mô hình xử lý hồ sơ thiếu lịch sử. Nếu cột chỉ được cập nhật sau bước thẩm định, trạng thái thiếu có thể làm lộ quy trình phê duyệt; khi đó không dùng.

income = pd.to\_numeric(df\["monthly\_income"], errors="coerce")  
score = pd.to\_numeric(df\["credit\_score"], errors="coerce")

df\["FE\_INCOME\_MISSING"] = income.isna().astype("int8")  
df\["FE\_SCORE\_MISSING"] = score.isna().astype("int8")

Nếu không có credit\_score, bỏ dòng tạo score và FE\_SCORE\_MISSING; không tạo cột điểm giả.

**Nhóm 2. Tài sản bảo đảm và tiền dự phòng**

Chỉ mở nhóm này nếu khoản vay có tài sản bảo đảm hoặc dữ liệu có số dư tiền gửi/tài sản thanh khoản trước ngày dự báo. Không có các cột này thì bỏ qua cả nhóm.

### **27. Tỷ lệ khoản vay trên giá trị tài sản — FE\_LOAN\_TO\_VALUE**

**Cần:** số tiền **xin vay**, giá trị tài sản bảo đảm đã được định giá tại thời điểm xét duyệt. Chỉ phù hợp với khoản vay có tài sản bảo đảm.

**Tạo:** requested\_loan\_amount / collateral\_value. Ví dụ xin vay 700 triệu trên tài sản giá trị 1 tỷ thì LTV là 70%.

**Ý nghĩa:** mô tả mức khoản vay so với tài sản bảo đảm. Giá trị tài sản thiếu không chứng minh khách hàng không có tài sản; có thể là chưa định giá. Không dùng số tiền *được duyệt* nếu nó chỉ xuất hiện sau quyết định.

requested = pd.to\_numeric(df\["requested\_loan\_amount"], errors="coerce")  
collateral = pd.to\_numeric(df\["collateral\_value"], errors="coerce")

df\["FE\_LOAN\_TO\_VALUE"] = requested.div(collateral.where(collateral > 0))  
df\["FE\_COLLATERAL\_VALUE\_MISSING"] = collateral.isna().astype("int8")

### **28. Tiền dự phòng đủ trả nợ bao nhiêu tháng — FE\_SAVINGS\_BUFFER\_MONTHS**

**Cần:** tiền tiết kiệm/tài sản thanh khoản đã biết tại ngày dự báo và tổng khoản phải trả hàng tháng. Không dùng số dư được cập nhật sau giải ngân.

**Tạo:** liquid\_savings / (existing\_monthly\_payment + proposed\_monthly\_payment).

**Ý nghĩa:** nếu có 60 triệu dự phòng và phải trả 6 triệu/tháng thì tỷ lệ là 10 tháng. Đây chỉ là một cách diễn giải khoản dự phòng; chưa tính tiền sinh hoạt và khả năng rút được tài sản. Khi khoản trả bằng 0 hoặc thiếu, để NaN.

savings = pd.to\_numeric(df\["liquid\_savings"], errors="coerce")  
old\_payment = pd.to\_numeric(df\["existing\_monthly\_payment"], errors="coerce")  
new\_payment = pd.to\_numeric(df\["proposed\_monthly\_payment"], errors="coerce")

total\_payment = old\_payment + new\_payment  
df\["FE\_SAVINGS\_BUFFER\_MONTHS"] = savings.div(total\_payment.where(total\_payment > 0))

**Nhóm 3. Sự ổn định của nguồn thu nhập**

Nhóm này dùng khi data dictionary có ngày bắt đầu công việc hoặc lịch sử thu nhập. Đây là tín hiệu bổ sung; ưu tiên thấp hơn lịch sử trả nợ, credit score và tổng gánh trả nợ.

### **29. Thâm niên công việc tại ngày dự báo — FE\_EMPLOYMENT\_TENURE\_MONTHS**

**Cần:** ngày bắt đầu công việc hiện tại và ngày dự báo.

**Tạo:** số tháng từ ngày bắt đầu đến ngày dự báo. Ví dụ làm việc từ tháng 1/2023 đến tháng 1/2025 là khoảng 24 tháng.

**Ý nghĩa:** mô tả độ dài của nguồn thu nhập hiện tại; không tự kết luận thâm niên ngắn là rủi ro. Ngày bắt đầu sau ngày dự báo là giá trị lỗi/không phù hợp nên để NaN.

as\_of = pd.to\_datetime(df\["observation\_date"], errors="coerce")  
started = pd.to\_datetime(df\["employment\_start\_date"], errors="coerce")  
days = (as\_of - started).dt.days

df\["FE\_EMPLOYMENT\_TENURE\_MONTHS"] = days.where(days >= 0) / 30.44

**Nếu có bảng lịch sử: chuẩn bị cutoff cho từng hồ sơ**

Các feature tiếp theo phải được tính **riêng cho từng dòng của `df`**. Một khách hàng có thể có nhiều hồ sơ ở các ngày khác nhau. Hàm dưới đây ghép sự kiện với từng dòng dự báo, rồi chỉ giữ sự kiện xảy ra **trước** ngày đó. Chạy cell này một lần trước các mục 31–33.

def history\_before\_snapshot(df, events, event\_date\_col):  
base = df\[\["customer\_id", "observation\_date"]].copy()  
base\["\_row\_id"] = np.arange(len(base))  
base\["observation\_date"] = pd.to\_datetime(base\["observation\_date"], errors="coerce")

&#x20;   hist \\= events.copy()  
hist\\\[event\\\_date\\\_col\\] \\= pd.to\\\_datetime(hist\\\[event\\\_date\\\_col\\], errors="coerce")  
joined \\= base.merge(hist, on="customer\\\_id", how="left", validate="many\\\_to\\\_many")  
joined \\= joined\\\[joined\\\[event\\\_date\\\_col\\] \\< joined\\\["observation\\\_date"\\]\\]  
return joined, base\\\["\\\_row\\\_id"\\]



Nếu có timestamp đầy đủ và chắc chắn sự kiện cùng ngày đã xảy ra **trước giờ ra quyết định**, có thể điều chỉnh dấu < thành <=. Chỉ có ngày mà không có giờ thì giữ < để tránh dùng thông tin xảy ra muộn hơn trong cùng ngày. Với bảng lớn, ghép nhiều lần quan sát của một khách hàng có thể tốn bộ nhớ.

**Nhóm 4. Lịch sử tín dụng và nghĩa vụ đang mở**

Dùng khi có bảng khoản vay, credit bureau hoặc credit card có khóa customer\_id và ngày sự kiện. Tất cả sự kiện phải xảy ra trước observation\_date. Với bài toán có lịch sử trả nợ, đây thường là nhóm nên thử sớm.

### **30. Tổng khoản trả của nợ đang mở — FE\_ACTIVE\_DEBT\_SERVICE\_MONTHLY**

**Cần:** bảng accounts với customer\_id, ngày mở, ngày đóng thực tế (có thể thiếu) và khoản trả hàng tháng. Cần biết bảng tài khoản có ghi đầy đủ các khoản vay cũ hay không.

**Tạo:** cộng khoản trả của các khoản đã mở nhưng chưa đóng ở ngày dự báo. Không cộng khoản vay mới nếu proposed\_monthly\_payment đã tính nó.

**Ý nghĩa:** guide cũ đếm số tài khoản mở; feature này đo **số tiền phải trả**, vì hai người cùng có ba tài khoản có thể có nghĩa vụ rất khác nhau. Nếu thiếu close\_date có nghĩa là chưa đóng, code sau dùng được; nếu nó có nghĩa là không rõ trạng thái thì cần sửa theo data dictionary.

base = df\[\["customer\_id", "observation\_date"]].copy()  
base\["\_row\_id"] = np.arange(len(base))  
base\["observation\_date"] = pd.to\_datetime(base\["observation\_date"], errors="coerce")

acc = accounts\[\["customer\_id", "open\_date", "close\_date", "monthly\_payment"]].copy()  
acc\["open\_date"] = pd.to\_datetime(acc\["open\_date"], errors="coerce")  
acc\["close\_date"] = pd.to\_datetime(acc\["close\_date"], errors="coerce")  
acc\["monthly\_payment"] = pd.to\_numeric(acc\["monthly\_payment"], errors="coerce")

joined = base.merge(acc, on="customer\_id", how="left", validate="many\_to\_many")  
active = joined\[  
(joined\["open\_date"] < joined\["observation\_date"])  
\& (joined\["close\_date"].isna() | (joined\["close\_date"] >= joined\["observation\_date"]))  
]  
monthly\_sum = active.groupby("\_row\_id")\["monthly\_payment"].sum(min\_count=1)  
df\["FE\_ACTIVE\_DEBT\_SERVICE\_MONTHLY"] = base\["\_row\_id"].map(monthly\_sum).to\_numpy()

Nếu registry khoản vay **được xác nhận là đầy đủ**, hồ sơ không có tài khoản mở có thể nhận 0. Nếu không, để NaN để tránh nhầm “không có nợ” với “không có dữ liệu”.

**Nhóm 5. Dấu hiệu thay đổi gần đây**

Nhóm này cần các snapshot hoặc sự kiện theo thời gian. Chỉ thử khi có đủ quan sát trước ngày dự báo; không tạo trend từ một hoặc hai dòng dữ liệu.

### **31. Độ biến động thu nhập 6 tháng — FE\_INCOME\_CV\_6M**

**Cần:** bảng income\_history có đúng một số tiền thu nhập cho mỗi khách hàng và mỗi tháng, cùng income\_date. Nếu có nhiều khoản thu trong một tháng, cộng về tháng trước khi chạy.

**Tạo:** độ lệch chuẩn thu nhập / thu nhập trung bình trong 6 tháng trước ngày dự báo. Chỉ giữ khi có ít nhất ba tháng quan sát; mẫu số 0 hoặc âm → NaN.

**Ý nghĩa:** hai người cùng thu nhập trung bình nhưng một người thu nhập lên xuống mạnh có thể khác nhau về khả năng trả nợ. Mức biến động cao không tự động có nghĩa là sẽ nợ xấu.

hist, row\_id = history\_before\_snapshot(df, income\_history, "income\_date")  
hist = hist\[  
hist\["income\_date"] >= hist\["observation\_date"] - pd.DateOffset(months=6)  
].copy()  
hist\["income\_amount"] = pd.to\_numeric(hist\["income\_amount"], errors="coerce")

summary = hist.groupby("\_row\_id")\["income\_amount"].agg(\["count", "mean", "std"])  
cv = summary\["std"].div(summary\["mean"].where(summary\["mean"] > 0))  
cv = cv.where(summary\["count"] >= 3)

df\["FE\_INCOME\_CV\_6M"] = row\_id.map(cv).to\_numpy()  
df\["FE\_INCOME\_MONTHS\_OBSERVED"] = row\_id.map(summary\["count"]).to\_numpy()

### **32. Số lần hỏi vay trong 30 ngày — FE\_INQUIRIES\_30D**

**Cần:** bảng inquiries có customer\_id và inquiry\_date. Guide cũ đã có số inquiry trong 6 tháng; mục này tập trung vào giai đoạn rất gần ngày dự báo.

**Tạo:** đếm inquiry trong 30 ngày trước ngày dự báo.

**Ý nghĩa:** phát hiện nhiều yêu cầu tín dụng trong thời gian ngắn. Nếu nhiều dòng là bản ghi trùng của cùng một lần hỏi, cần khử trùng theo mã inquiry trước. Chỉ điền 0 cho người không có inquiry khi bảng lịch sử được xác nhận là đầy đủ.

hist, row\_id = history\_before\_snapshot(df, inquiries, "inquiry\_date")  
recent = hist\[  
hist\["inquiry\_date"] >= hist\["observation\_date"] - pd.Timedelta(days=30)  
]  
counts = recent.groupby("\_row\_id").size()

\# fillna(0) giả định bảng inquiries có đầy đủ lịch sử cho mọi khách hàng.  
df\["FE\_INQUIRIES\_30D"] = row\_id.map(counts).fillna(0).astype("int32").to\_numpy()

### **33. Mức sử dụng hạn mức đang tăng hay giảm — FE\_UTILIZATION\_CHANGE\_3M**

**Cần:** bảng monthly\_credit có một snapshot mỗi khách hàng/tháng với ngày, dư nợ và hạn mức. Guide cũ có mức sử dụng *cao nhất* và độ biến động; mục này nhìn vào **chiều thay đổi**.

**Tạo:** mức sử dụng trung bình trong 45 ngày gần nhất trừ mức trung bình của khoảng 90–135 ngày trước ngày dự báo. Số dương là gần đây dùng hạn mức nhiều hơn. Cần cả hai khoảng có ít nhất một quan sát; nếu không thì NaN.

**Ý nghĩa:** một người đang dùng hạn mức tăng nhanh có thể khác người luôn duy trì mức ổn định, dù cùng mức hiện tại. Khoảng thời gian ở đây chỉ là ví dụ để bắt dữ liệu theo tháng; đổi cho khớp tần suất thật.

hist, row\_id = history\_before\_snapshot(df, monthly\_credit, "snapshot\_date")  
balance = pd.to\_numeric(hist\["balance"], errors="coerce")  
limit = pd.to\_numeric(hist\["credit\_limit"], errors="coerce")  
hist\["\_utilization"] = balance.div(limit.where(limit > 0))

recent = hist\[  
hist\["snapshot\_date"] >= hist\["observation\_date"] - pd.Timedelta(days=45)  
].groupby("\_row\_id")\["\_utilization"].mean()

prior\_rows = hist\[  
(hist\["snapshot\_date"] >= hist\["observation\_date"] - pd.Timedelta(days=135))  
\& (hist\["snapshot\_date"] < hist\["observation\_date"] - pd.Timedelta(days=90))  
]  
prior = prior\_rows.groupby("\_row\_id")\["\_utilization"].mean()

change = recent - prior  
df\["FE\_UTILIZATION\_CHANGE\_3M"] = row\_id.map(change).to\_numpy()



## Tài liệu đã tham khảo

* Muñoz-Cancino et al. (2022), [dynamics of credit history and repayment behavior](https://arxiv.org/abs/2204.06122): nghiên cứu lịch sử tín dụng và hành vi trả nợ theo thời gian.
* Hosseini et al. (2018), [DeepCredit](https://ojs.aaai.org/index.php/ICWSM/article/view/15001): chuỗi repayment và hoạt động tài chính có tín hiệu cho rủi ro khoản vay.
* [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk): ví dụ dữ liệu nhiều bảng cho lịch trả, hạn mức và lịch sử tín dụng.

Các feature là ứng viên để thử, không đảm bảo dự báo tốt trên dataset chưa biết. Bắt đầu bằng nhóm hồ sơ, sau đó thêm lịch sử trả và so sánh validation ngoài thời gian.

5. Will Koehrsen, [Start Here: A Gentle Introduction — Feature Engineering](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction#Feature-Engineering): polynomial terms từ EXT\_SOURCE và DAYS\_BIRTH cùng domain ratios.

