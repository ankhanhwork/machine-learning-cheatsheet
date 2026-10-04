# Hướng dẫn Feature Engineering cho dự báo rủi ro nợ xấu

> Mỗi đoạn code nằm ngay dưới mô tả feature tương ứng. Code được lấy trực tiếp từ notebook.

# Feature Engineering — Predicting Customer Bad Debt Risk

## Đọc trước khi chạy
Mỗi feature là **một cột mới**. Ví dụ `df["FE_LOAN_TO_INCOME"] = ...` nghĩa là thêm cột vào bảng khách hàng. Hãy đổi tên cột bên phải cho khớp dữ liệu của bạn. Cần một dòng mỗi khách hàng (hoặc mỗi khoản vay) tại ngày dự đoán. Không có cột tương ứng thì bỏ qua feature.

**Cực kỳ quan trọng:** chỉ dùng thông tin đã biết tại ngày dự đoán. Không dùng khoản trả, dư nợ hay trạng thái được ghi nhận sau ngày đó. `TARGET` không tham gia tạo feature.

## Cell 1 — Nạp dữ liệu và xem cột
Thay đường dẫn bằng file của bạn. Nếu bảng đã có sẵn trong notebook pipeline thì dùng chính DataFrame đó.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("your_customer_data.csv")
print(df.shape)
print(df.columns.tolist())
df.head()
```

## 1. Tỷ lệ khoản vay trên thu nhập — `FE_LOAN_TO_INCOME`

**Cần:** số tiền khoản vay và thu nhập năm. **Tạo:** `loan_amount / annual_income`.
**Ý nghĩa:** khoản vay lớn cỡ nào so với khả năng tạo thu nhập; tỷ lệ cao có thể báo gánh nặng lớn. Thu nhập 0/thiếu tạo giá trị không xác định, nên để NaN và xử lý trong pipeline.

Đổi hai tên cột trong ngoặc vuông theo dataset.

```python
df["FE_LOAN_TO_INCOME"] = df["loan_amount"] / df["annual_income"].replace(0, np.nan)
```

## 2. Gánh khoản trả hàng tháng — `FE_INSTALLMENT_TO_MONTHLY_INCOME`

**Cần:** khoản trả mỗi tháng và thu nhập năm. **Tạo:** monthly installment / (annual income / 12).
**Ý nghĩa:** phần thu nhập tháng cần dành để trả khoản vay này. Cao nghĩa là ngân sách còn lại ít hơn. Nếu dataset thu nhập đã theo tháng, bỏ `/ 12`.

```python
df["FE_INSTALLMENT_TO_MONTHLY_INCOME"] = (
    df["monthly_installment"] / (df["annual_income"].replace(0, np.nan) / 12)
)
```

## 3. Tổng nợ trên thu nhập — `FE_DEBT_TO_INCOME`

**Cần:** tổng dư nợ hiện có và thu nhập năm. **Tạo:** total debt / annual income.
**Ý nghĩa:** mức nợ tổng thể so với thu nhập. Chỉ dùng nếu `total_debt` được đo tại hoặc trước ngày dự đoán; đừng cộng khoản nợ phát sinh sau đó.

```python
df["FE_DEBT_TO_INCOME"] = df["total_debt"] / df["annual_income"].replace(0, np.nan)
```

## 4. Mức dùng hạn mức — `FE_CREDIT_UTILIZATION`

**Cần:** dư nợ thẻ/tín dụng quay vòng và hạn mức. **Tạo:** balance / credit limit.
**Ý nghĩa:** phần hạn mức đang sử dụng. Ví dụ 0.8 là 80%. Nếu vượt 1, đừng tự cắt bỏ trước khi hiểu cách ghi nhận hạn mức/dư nợ.

```python
df["FE_CREDIT_UTILIZATION"] = df["credit_card_balance"] / df["credit_limit"].replace(0, np.nan)
```

## Nếu có bảng lịch sử riêng

Các cell feature 5–8 dùng bảng `payments`; feature 9–10 dùng bảng `monthly`. Nếu dataset chỉ có một bảng, bỏ qua các phần này. Thay đường dẫn bằng file thực tế. Tên bảng phải có đúng các cột dùng ở bên dưới.

## 5. Số ngày trả trễ — `FE_DAYS_LATE`

**Cần:** ngày đến hạn và ngày trả thực tế cho từng kỳ trả. Bảng này thường có nhiều dòng mỗi khách hàng.
**Tạo:** ngày trả trừ ngày đến hạn; số dương là trễ, số âm đặt về 0.
**Ý nghĩa:** đo mức trễ cụ thể. Với kỳ chưa trả, ngày trả bị trống không có nghĩa là đúng hạn. Hãy tính số ngày từ ngày đến hạn đến ngày dự đoán (cutoff), sau đó mới tổng hợp.

```python
payments["due_date"] = pd.to_datetime(payments["due_date"])
payments["payment_date"] = pd.to_datetime(payments["payment_date"])
payments["FE_DAYS_LATE"] = (payments["payment_date"] - payments["due_date"]).dt.days.clip(lower=0)
```

## 6. Tỷ lệ kỳ trả bị trễ — `FE_LATE_PAYMENT_RATE`

**Cần:** `FE_DAYS_LATE` trên từng kỳ và ID khách hàng.
**Tạo:** số kỳ có trễ / tổng số kỳ của khách hàng.
**Ý nghĩa:** tỷ lệ trả trễ trong lịch sử quan sát. Phải chỉ tổng hợp các kỳ đã đến hạn trước ngày dự đoán.

```python
payments["FE_WAS_LATE"] = (payments["FE_DAYS_LATE"] > 0).astype(int)
late_rate = payments.groupby("customer_id")["FE_WAS_LATE"].mean().rename("FE_LATE_PAYMENT_RATE")
df = df.merge(late_rate, on="customer_id", how="left", validate="one_to_one")
```

## 7. Tỷ lệ trả trễ từ 30 ngày — `FE_LATE_30_RATE`

**Cần:** ngày trễ của từng kỳ và ID khách hàng. **Tạo:** cờ `days late >= 30`, rồi lấy trung bình theo khách hàng.
**Ý nghĩa:** tập trung vào trễ nghiêm trọng hơn một vài ngày. Nếu nghiệp vụ dùng ngưỡng khác (ví dụ 60/90 ngày), đổi số 30.

```python
payments["FE_LATE_30"] = (payments["FE_DAYS_LATE"] >= 30).astype(int)
late_30_rate = payments.groupby("customer_id")["FE_LATE_30"].mean().rename("FE_LATE_30_RATE")
df = df.merge(late_30_rate, on="customer_id", how="left", validate="one_to_one")
```

## 8. Số tiền trả thiếu — `FE_PAYMENT_SHORTFALL_RATE`

**Cần:** số tiền đến hạn và số tiền thực trả cho mỗi kỳ. **Tạo:** (scheduled - actual), tối thiểu bằng 0; chia cho scheduled để thành tỷ lệ thiếu.
**Ý nghĩa:** phát hiện trả một phần hoặc thiếu tiền, kể cả khi có ngày thanh toán. Tổng hợp trung bình theo khách hàng.

```python
payments["FE_SHORTFALL"] = (payments["scheduled_amount"] - payments["actual_amount"]).clip(lower=0)
payments["FE_SHORTFALL_RATE"] = payments["FE_SHORTFALL"] / payments["scheduled_amount"].replace(0, np.nan)
shortfall_rate = payments.groupby("customer_id")["FE_SHORTFALL_RATE"].mean().rename("FE_PAYMENT_SHORTFALL_RATE")
df = df.merge(shortfall_rate, on="customer_id", how="left", validate="one_to_one")
```

## 9. Đỉnh mức dùng hạn mức theo thời gian — `FE_UTILIZATION_MAX`

**Cần:** bảng nhiều dòng theo khách hàng và tháng, có balance, limit. **Tạo:** utilization mỗi tháng rồi lấy giá trị lớn nhất theo khách hàng.
**Ý nghĩa:** bắt được tháng khách hàng sử dụng hạn mức rất cao dù trung bình cả kỳ thấp. Bảng tháng phải cắt ở ngày dự đoán.

```python
monthly["FE_UTIL"] = monthly["balance"] / monthly["credit_limit"].replace(0, np.nan)
util_max = monthly.groupby("customer_id")["FE_UTIL"].max().rename("FE_UTILIZATION_MAX")
df = df.merge(util_max, on="customer_id", how="left", validate="one_to_one")
```

## 10. Biến động mức dùng hạn mức — `FE_UTILIZATION_STD`

**Cần:** cùng bảng tháng ở feature 9. **Tạo:** độ lệch chuẩn utilization theo khách hàng.
**Ý nghĩa:** độ biến động qua thời gian; cao có thể cho thấy sử dụng tín dụng thất thường. Chỉ có một tháng thì kết quả NaN là bình thường.

```python
util_std = monthly.groupby("customer_id")["FE_UTIL"].std().rename("FE_UTILIZATION_STD")
df = df.merge(util_std, on="customer_id", how="left", validate="one_to_one")
```

## 11. Số lượng lịch sử trả — `FE_PAYMENT_HISTORY_COUNT`

**Cần:** một dòng mỗi kỳ trả và ID khách hàng. **Tạo:** đếm số kỳ đủ điều kiện trước cutoff.
**Ý nghĩa:** phân biệt người không có lịch sử với người có lịch sử tốt. Không có lịch sử không đồng nghĩa ít rủi ro; nên để riêng count và cờ no-history.

```python
payment_count = payments.groupby("customer_id").size().rename("FE_PAYMENT_HISTORY_COUNT")
df = df.merge(payment_count, on="customer_id", how="left", validate="one_to_one")
df["FE_NO_PAYMENT_HISTORY"] = df["FE_PAYMENT_HISTORY_COUNT"].isna().astype(int)
```

## Cuối cùng: kiểm tra các cột vừa tạo

Một số merge có thể tạo NaN cho khách hàng không có lịch sử. Giữ missing/no-history có chủ đích; điền giá trị trong preprocessing pipeline sau khi chia train/validation.

```python
feature_cols = [c for c in df.columns if c.startswith("FE_")]
df[["customer_id"] + feature_cols].head()
```

## 12. Số khoản tín dụng đang mở — `FE_OPEN_ACCOUNT_COUNT`

**Cần:** bảng tài khoản tín dụng, một dòng mỗi tài khoản; cột trạng thái `account_status` và `customer_id`.
**Tạo:** lọc tài khoản đang mở rồi đếm theo khách hàng.
**Ý nghĩa:** tổng mức phơi nhiễm/độ phức tạp nghĩa vụ tín dụng. Đây là ứng viên tham khảo từ credit scoring; quan hệ có thể phi tuyến, cần kiểm chứng.

```python
open_accounts = accounts[accounts["account_status"] == "open"]
open_count = open_accounts.groupby("customer_id").size().rename("FE_OPEN_ACCOUNT_COUNT")
df = df.merge(open_count, on="customer_id", how="left", validate="one_to_one")
```

## 13. Thời gian từ lần trễ gần nhất — `FE_DAYS_SINCE_LAST_LATE`

**Cần:** `payments` có ngày đến hạn/ngày trả và ngày quan sát `observation_date` trong bảng khách hàng.
**Tạo:** lấy ngày trả gần nhất có trễ; trừ khỏi ngày quan sát.
**Ý nghĩa:** phân biệt vi phạm cũ với vi phạm gần đây. Nếu chưa từng trễ, giá trị là NaN; có thể thêm cờ chưa từng trễ riêng. Chỉ dùng kỳ thanh toán đã đến hạn trước ngày quan sát.

```python
late_events = payments[payments["FE_DAYS_LATE"] > 0].copy()
last_late = late_events.groupby("customer_id")["payment_date"].max().rename("_last_late_date")
df = df.merge(last_late, on="customer_id", how="left", validate="one_to_one")
df["FE_DAYS_SINCE_LAST_LATE"] = (
    pd.to_datetime(df["observation_date"]) - pd.to_datetime(df["_last_late_date"])
).dt.days
df = df.drop(columns="_last_late_date")
```

## 14. Tỷ lệ trả trễ trong 6 tháng gần nhất — `FE_LATE_RATE_6M`

**Cần:** lịch trả có `due_date`, cờ `FE_WAS_LATE`, `customer_id`; cùng ngày quan sát.
**Tạo:** giữ các kỳ đến hạn trong 6 tháng trước observation date, rồi tính trung bình cờ trễ.
**Ý nghĩa:** phản ứng nhanh hơn biến lifetime nếu hành vi khách hàng đang xấu đi/cải thiện. Đổi 6 thành 3/12 để so sánh.

```python
payments["due_date"] = pd.to_datetime(payments["due_date"])
# Gắn observation date cho mỗi khách hàng trước khi lọc cửa sổ lịch sử.
payments_window = payments.merge(df[["customer_id", "observation_date"]], on="customer_id", how="inner")
cutoff = pd.to_datetime(payments_window["observation_date"])
start_6m = cutoff - pd.DateOffset(months=6)
recent_6m = payments_window[(payments_window["due_date"] <= cutoff) & (payments_window["due_date"] > start_6m)]
late_rate_6m = recent_6m.groupby("customer_id")["FE_WAS_LATE"].mean().rename("FE_LATE_RATE_6M")
df = df.merge(late_rate_6m, on="customer_id", how="left", validate="one_to_one")
```

## 15. Xu hướng trả trễ — `FE_LATE_RATE_TREND_6M_VS_PRIOR_6M`

**Cần:** cùng bảng payments và cờ `FE_WAS_LATE`.
**Tạo:** tỷ lệ trễ 6 tháng gần nhất trừ tỷ lệ 6 tháng liền trước.
**Ý nghĩa:** số dương cho thấy tỷ lệ trễ tăng gần đây; số âm là giảm. Đây là giả thuyết hữu ích từ phân tích hành vi theo thời gian, cần kiểm tra độ ổn định khi có đủ lịch sử.

```python
start_prior_6m = cutoff - pd.DateOffset(months=12)
prior_6m = payments_window[(payments_window["due_date"] <= start_6m) & (payments_window["due_date"] > start_prior_6m)]
late_rate_prior_6m = prior_6m.groupby("customer_id")["FE_WAS_LATE"].mean().rename("_late_rate_prior_6m")
trend = late_rate_6m.to_frame().join(late_rate_prior_6m, how="outer")
trend["FE_LATE_RATE_TREND_6M_VS_PRIOR_6M"] = trend["FE_LATE_RATE_6M"] - trend["_late_rate_prior_6m"]
df = df.merge(trend[["FE_LATE_RATE_TREND_6M_VS_PRIOR_6M"]], on="customer_id", how="left", validate="one_to_one")
```

## 16. Số yêu cầu tín dụng gần đây — `FE_INQUIRIES_6M`

**Cần:** bảng inquiry có ngày yêu cầu và ID khách hàng.
**Tạo:** đếm inquiry trong sáu tháng trước ngày quan sát.
**Ý nghĩa:** có thể phản ánh nhu cầu vay/đăng ký tín dụng gần đây. Nhiều inquiry trùng do một lần mua sắm có thể cần gộp theo ngày/loại; chỉ dùng nếu dữ liệu hợp lệ và được phép.

```python
inquiries["inquiry_date"] = pd.to_datetime(inquiries["inquiry_date"])
inquiries_window = inquiries.merge(df[["customer_id", "observation_date"]], on="customer_id", how="inner")
inquiry_cutoff = pd.to_datetime(inquiries_window["observation_date"])
inquiry_start = inquiry_cutoff - pd.DateOffset(months=6)
inquiries_6m = inquiries_window[(inquiries_window["inquiry_date"] <= inquiry_cutoff) &
                                (inquiries_window["inquiry_date"] > inquiry_start)]
inquiry_count = inquiries_6m.groupby("customer_id").size().rename("FE_INQUIRIES_6M")
df = df.merge(inquiry_count, on="customer_id", how="left", validate="one_to_one")
```

## 17. Tuổi lịch sử tín dụng — `FE_CREDIT_HISTORY_MONTHS`

**Cần:** ngày mở tài khoản tín dụng cũ nhất, ngày quan sát.
**Tạo:** số tháng giữa ngày mở sớm nhất và observation date.
**Ý nghĩa:** lịch sử dài hơn cho thêm bằng chứng hành vi, nhưng không mặc định là rủi ro thấp. Cần có ngày mở chính xác và lịch sử đủ phạm vi.

```python
accounts["open_date"] = pd.to_datetime(accounts["open_date"])
accounts = accounts.merge(df[["customer_id", "observation_date"]], on="customer_id", how="inner")
first_open = accounts.groupby("customer_id")["open_date"].min().rename("_first_open")
df = df.merge(first_open, on="customer_id", how="left", validate="one_to_one")
df["FE_CREDIT_HISTORY_MONTHS"] = (
    (pd.to_datetime(df["observation_date"]) - pd.to_datetime(df["_first_open"])).dt.days / 30.44
)
df = df.drop(columns="_first_open")
```

# Bổ sung từ notebook Kaggle Will Koehrsen

Các biến bên dưới dựa trên phần polynomial/domain feature engineering trong [Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction#Feature-Engineering). Ví dụ gốc dùng schema Home Credit (`AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_ANNUITY`, `EXT_SOURCE_1/2/3`, `DAYS_BIRTH`). Chỉ chạy các feature khi bạn có cột tương đương; tên khác thì thay lại.

## 18. Tuổi khách hàng — `FE_AGE_YEARS`

**Cần:** tuổi dạng ngày âm tính từ ngày sinh (`DAYS_BIRTH`, quy ước Home Credit), hoặc ngày sinh và ngày quan sát.
**Tạo:** `-DAYS_BIRTH / 365.25`.
**Ý nghĩa:** chuyển số ngày khó đọc thành số năm. Nếu dữ liệu có ngày sinh thật, tính tuổi tại observation date; tránh dùng tuổi hiện tại cho hồ sơ lịch sử.

```python
df["FE_AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
```

## 19. Khoản vay trên thu nhập — `FE_CREDIT_INCOME_RATIO`

**Cần:** số tiền tín dụng/khoản vay được đề nghị và tổng thu nhập. **Tạo:** `AMT_CREDIT / AMT_INCOME_TOTAL`.
**Ý nghĩa:** gánh khoản vay so với thu nhập. Đây là biến gốc trong ví dụ Kaggle; tương tự `FE_LOAN_TO_INCOME` ở trên, nên kiểm tra trùng lặp và chỉ giữ một tên rõ ràng.

```python
df["FE_CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
```

## 20. Khoản trả trên thu nhập — `FE_ANNUITY_INCOME_RATIO`

**Cần:** khoản trả định kỳ `AMT_ANNUITY` và thu nhập `AMT_INCOME_TOTAL`.
**Tạo:** annuity / income. **Ý nghĩa:** phần thu nhập dành cho khoản trả theo chu kỳ dữ liệu. Giữ đúng đơn vị/thời kỳ; nếu annuity theo tháng còn thu nhập theo năm thì nhân income với 1/12.

```python
df["FE_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
```

## 21. Khoản vay trên khoản trả — `FE_CREDIT_ANNUITY_RATIO`

**Cần:** `AMT_CREDIT`, `AMT_ANNUITY`. **Tạo:** credit amount / annuity.
**Ý nghĩa:** proxy số kỳ/thời hạn trả nợ khi điều kiện vay phù hợp. Không coi là số tháng chính xác nếu lãi suất, phí hoặc lịch trả thay đổi.

```python
df["FE_CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
```

## 22. Tổng hợp điểm nguồn ngoài — `FE_EXT_SOURCE_MEAN`, `FE_EXT_SOURCE_STD`

**Cần:** các điểm `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` (tương tự các score bên ngoài của dataset khác).
**Tạo:** trung bình và độ lệch chuẩn theo hàng, bỏ qua điểm thiếu.
**Ý nghĩa:** mean gom tín hiệu thành một mức tổng quát; std biểu thị các nguồn đồng thuận hay khác biệt. Missing count nên xem riêng vì số điểm có sẵn giữa khách hàng có thể khác nhau.

```python
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
df["FE_EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
df["FE_EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
df["FE_EXT_SOURCE_COUNT"] = df[ext_cols].notna().sum(axis=1)
```

## 23. Tương tác và phi tuyến giữa điểm ngoài/tuổi — `FE_POLY_*`

**Cần:** các cột ở trên. Notebook Kaggle tạo polynomial terms tới bậc 3 từ ba điểm EXT_SOURCE và DAYS_BIRTH (bao gồm bình phương, tích chéo). Điều này cho phép mô hình tuyến tính học quan hệ cong/tương tác.

Code dưới dùng `PolynomialFeatures`. Fit bộ sinh feature chỉ trên train, sau đó dùng chính transformer để biến đổi validation/test; tuyệt đối không fit riêng trên validation/test. `TARGET`/label không đưa vào danh sách đầu vào.

```python
from sklearn.preprocessing import PolynomialFeatures

poly_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]
poly_train_input = train_df[poly_cols].copy()
poly_valid_input = valid_df[poly_cols].copy()
# Điền missing bằng median học từ train; giữ cùng giá trị cho validation/test.
poly_medians = poly_train_input.median()
poly_train_input = poly_train_input.fillna(poly_medians)
poly_valid_input = poly_valid_input.fillna(poly_medians)
poly_transformer = PolynomialFeatures(degree=3, include_bias=False)
poly_train_array = poly_transformer.fit_transform(poly_train_input)
poly_valid_array = poly_transformer.transform(poly_valid_input)
poly_names = poly_transformer.get_feature_names_out(poly_cols)
poly_train_features = pd.DataFrame(poly_train_array, columns=["FE_POLY_" + c for c in poly_names], index=train_df.index)
poly_valid_features = pd.DataFrame(poly_valid_array, columns=["FE_POLY_" + c for c in poly_names], index=valid_df.index)
train_df = pd.concat([train_df, poly_train_features], axis=1)
valid_df = pd.concat([valid_df, poly_valid_features], axis=1)
```

**Lưu ý:** Với tree boosting, polynomial expansion có thể dư thừa; thử ablation. Polynomial terms có thể làm tăng mạnh số cột và nhạy với missing/outlier. Không thêm `TARGET` vào `poly_cols`.

## Tài liệu đã tham khảo

- Muñoz-Cancino et al. (2022), [dynamics of credit history and repayment behavior](https://arxiv.org/abs/2204.06122): nghiên cứu lịch sử tín dụng và hành vi trả nợ theo thời gian.
- Hosseini et al. (2018), [DeepCredit](https://ojs.aaai.org/index.php/ICWSM/article/view/15001): chuỗi repayment và hoạt động tài chính có tín hiệu cho rủi ro khoản vay.
- [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk): ví dụ dữ liệu nhiều bảng cho lịch trả, hạn mức và lịch sử tín dụng.

Các feature là ứng viên để thử, không đảm bảo dự báo tốt trên dataset chưa biết. Bắt đầu bằng nhóm hồ sơ, sau đó thêm lịch sử trả và so sánh validation ngoài thời gian.

5. Will Koehrsen, [Start Here: A Gentle Introduction — Feature Engineering](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction#Feature-Engineering): polynomial terms từ EXT_SOURCE và DAYS_BIRTH cùng domain ratios.

