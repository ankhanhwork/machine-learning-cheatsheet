# FE\_Tổng hợp.md

\*\*Mục tiêu:\*\* Đây là file chuẩn bổ sung và tổng hợp feature engineering cho bài toán Credit Risk Prediction hoặc Loan Approval.
>
> - Feature được nhóm theo \*\*business meaning\*\*, không theo thứ tự phát hiện.
> - Các feature trùng thực sự được hợp nhất.
> - Feature chỉ cùng family nhưng khác công thức, cửa sổ thời gian hoặc ý nghĩa vẫn được giữ riêng.
> - Mọi feature lịch sử phải chỉ dùng thông tin \*\*có trước prediction / observation date\*\*.
> - `TARGET` tuyệt đối không được dùng để tạo feature.

\---

## Mục lục Feature

> Chỉ liệt kê các feature để tra cứu nhanh.

**A. Affordability**

* [A1. Loan-to-Income — FE\_LOAN\_TO\_INCOME](#a1-loan-to-income-fe_loan_to_income)
* [A2. Installment-to-Monthly-Income — FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME](#a2-installment-to-monthly-income-fe_installment_to_monthly_income)
* [A3. Outstanding Debt-to-Income — FE\_DEBT\_TO\_INCOME](#a3-outstanding-debt-to-income-fe_debt_to_income)
* [A4. DTI theo Debt Service — FE\_DTI\_DEBT\_SERVICE](#a4-dti-theo-debt-service-fe_dti_debt_service)
* [A5. Total Payment-to-Income / FOIR — FE\_TOTAL\_PAYMENT\_TO\_INCOME](#a5-total-payment-to-income-foir-fe_total_payment_to_income)
* [A6. Residual Income — FE\_RESIDUAL\_INCOME\_AFTER\_DEBT](#a6-residual-income-fe_residual_income_after_debt)
* [A7. Salary Coverage — FE\_SALARY\_COVERAGE](#a7-salary-coverage-fe_salary_coverage)
* [A8. Credit-to-Annuity — FE\_CREDIT\_ANNUITY\_RATIO](#a8-credit-to-annuity-fe_credit_annuity_ratio)

**B. Liquidity / CASA**

* [B1. CASA Balance — FE\_CASA\_BALANCE](#b1-casa-balance-fe_casa_balance)
* [B2. Customer CASA Ratio — FE\_CASA\_RATIO](#b2-customer-casa-ratio-fe_casa_ratio)
* [B3. Average CASA — FE\_AVG\_CASA\_BALANCE](#b3-average-casa-fe_avg_casa_balance)
* [B4. CASA-to-Income — FE\_CASA\_TO\_INCOME](#b4-casa-to-income-fe_casa_to_income)
* [B5. CASA Buffer Months — FE\_CASA\_BUFFER\_MONTHS](#b5-casa-buffer-months-fe_casa_buffer_months)
* [B6. Savings Buffer Months — FE\_SAVINGS\_BUFFER\_MONTHS](#b6-savings-buffer-months-fe_savings_buffer_months)
* [B7. Average Monthly Inflow — FE\_AVG\_MONTHLY\_INFLOW](#b7-average-monthly-inflow-fe_avg_monthly_inflow)
* [B8. Net Cash Flow — FE\_NET\_CASH\_FLOW](#b8-net-cash-flow-fe_net_cash_flow)
* [B9. Cashflow Coverage — FE\_CASHFLOW\_COVERAGE](#b9-cashflow-coverage-fe_cashflow_coverage)
* [B10. Average Balance — FE\_AVG\_BALANCE](#b10-average-balance-fe_avg_balance)
* [B11. Minimum Balance — FE\_MIN\_BALANCE](#b11-minimum-balance-fe_min_balance)
* [B12. Balance Volatility — FE\_BALANCE\_CV](#b12-balance-volatility-fe_balance_cv)
* [B13. Savings Rate — FE\_SAVINGS\_RATE](#b13-savings-rate-fe_savings_rate)

**C. Exposure**

* [C1. Credit Utilization — FE\_CREDIT\_UTILIZATION](#c1-credit-utilization-fe_credit_utilization)
* [C2. Utilization Max — FE\_UTILIZATION\_MAX](#c2-utilization-max-fe_utilization_max)
* [C3. Utilization Volatility — FE\_UTILIZATION\_STD](#c3-utilization-volatility-fe_utilization_std)
* [C4. Utilization Change — FE\_UTILIZATION\_CHANGE\_3M](#c4-utilization-change-fe_utilization_change_3m)
* [C5. Open Account Count — FE\_OPEN\_ACCOUNT\_COUNT](#c5-open-account-count-fe_open_account_count)
* [C6. Active Loan Count — FE\_ACTIVE\_LOAN\_COUNT](#c6-active-loan-count-fe_active_loan_count)
* [C7. Active Debt Service Monthly — FE\_ACTIVE\_DEBT\_SERVICE\_MONTHLY](#c7-active-debt-service-monthly-fe_active_debt_service_monthly)
* [C8. Total Outstanding Debt — FE\_TOTAL\_OUTSTANDING\_DEBT](#c8-total-outstanding-debt-fe_total_outstanding_debt)
* [C9. Inquiry Count 30D — FE\_INQUIRIES\_30D](#c9-inquiry-count-30d-fe_inquiries_30d)
* [C10. Inquiry Count 3M — FE\_INQUIRIES\_3M](#c10-inquiry-count-3m-fe_inquiries_3m)
* [C11. Inquiry Count 6M — FE\_INQUIRIES\_6M](#c11-inquiry-count-6m-fe_inquiries_6m)
* [C12. Inquiry Count 12M — FE\_INQUIRIES\_12M](#c12-inquiry-count-12m-fe_inquiries_12m)

**D. Repayment Behaviour**

* [D1. Days Late / DPD per payment — FE\_DAYS\_LATE](#d1-days-late-dpd-per-payment-fe_days_late)
* [D2. Late Payment Rate — FE\_LATE\_PAYMENT\_RATE](#d2-late-payment-rate-fe_late_payment_rate)
* [D3. Late 30+ Rate — FE\_LATE\_30\_RATE](#d3-late-30-rate-fe_late_30_rate)
* [D4. Payment Shortfall Rate — FE\_PAYMENT\_SHORTFALL\_RATE](#d4-payment-shortfall-rate-fe_payment_shortfall_rate)
* [D5. Payment History Count — FE\_PAYMENT\_HISTORY\_COUNT](#d5-payment-history-count-fe_payment_history_count)
* [D6. No Payment History — FE\_NO\_PAYMENT\_HISTORY](#d6-no-payment-history-fe_no_payment_history)
* [D7. Days Since Last Late — FE\_DAYS\_SINCE\_LAST\_LATE](#d7-days-since-last-late-fe_days_since_last_late)
* [D8. Recent Late Rate 6M — FE\_LATE\_RATE\_6M](#d8-recent-late-rate-6m-fe_late_rate_6m)
* [D9. Late Rate Trend — FE\_LATE\_RATE\_TREND\_6M\_VS\_PRIOR\_6M](#d9-late-rate-trend-fe_late_rate_trend_6m_vs_prior_6m)
* [D10. Max DPD — FE\_MAX\_DPD](#d10-max-dpd-fe_max_dpd)

**E. Collateral / Loan Structure**

* [E1. Loan-to-Value — FE\_LOAN\_TO\_VALUE](#e1-loan-to-value-fe_loan_to_value)
* [E2. Collateral Value Missing — FE\_COLLATERAL\_VALUE\_MISSING](#e2-collateral-value-missing-fe_collateral_value_missing)
* [E3. Age at Loan Maturity — FE\_AGE\_AT\_MATURITY](#e3-age-at-loan-maturity-fe_age_at_maturity)

**F. Stability**

* [F1. Employment Tenure — FE\_EMPLOYMENT\_TENURE\_MONTHS](#f1-employment-tenure-fe_employment_tenure_months)
* [F2. Income CV 6M — FE\_INCOME\_CV\_6M](#f2-income-cv-6m-fe_income_cv_6m)
* [F3. Income Months Observed — FE\_INCOME\_MONTHS\_OBSERVED](#f3-income-months-observed-fe_income_months_observed)
* [F4. Relationship Tenure — FE\_RELATIONSHIP\_MONTHS](#f4-relationship-tenure-fe_relationship_months)

**G. Credit Profile**

* [G1. Credit History Months — FE\_CREDIT\_HISTORY\_MONTHS](#g1-credit-history-months-fe_credit_history_months)
* [G2. External Score Mean — FE\_EXT\_SOURCE\_MEAN](#g2-external-score-mean-fe_ext_source_mean)
* [G3. External Score STD — FE\_EXT\_SOURCE\_STD](#g3-external-score-std-fe_ext_source_std)
* [G4. External Score Count — FE\_EXT\_SOURCE\_COUNT](#g4-external-score-count-fe_ext_source_count)
* [G5. Score Missing — FE\_SCORE\_MISSING](#g5-score-missing-fe_score_missing)

**H. Customer Context**

* [H1. Age Years — FE\_AGE\_YEARS](#h1-age-years-fe_age_years)
* [H2. Income Missing — FE\_INCOME\_MISSING](#h2-income-missing-fe_income_missing)
* [H3. Income per Dependent — FE\_INCOME\_PER\_DEPENDENT](#h3-income-per-dependent-fe_income_per_dependent)
* [H4. Product Holding Count — FE\_PRODUCT\_COUNT](#h4-product-holding-count-fe_product_count)

**I. SME**

* [I1. DSCR — FE\_DSCR](#i1-dscr-fe_dscr)
* [I2. Debt-to-Assets — FE\_DEBT\_TO\_ASSETS](#i2-debt-to-assets-fe_debt_to_assets)
* [I3. Debt-to-Equity — FE\_DEBT\_TO\_EQUITY](#i3-debt-to-equity-fe_debt_to_equity)

**J. Advanced**

* [J1. Polynomial / Interaction Features — FE\_POLY\_\*](#j1-polynomial-interaction-features-fe_poly_)

\---

# 0\. Cách dùng file này

Khi nhận dataset:

```text
1. Xác định target, ID, observation date
        ↓
2. Scan data dictionary
        ↓
3. Map raw columns vào các nhóm A–J
        ↓
4. Chọn feature có thể tạo
        ↓
5. Kiểm tra unit + missing + leakage
        ↓
6. Tạo feature
        ↓
7. Kiểm tra distribution / bad rate / validation
```

## Quy tắc 3 mức

### Exact feature

Có đủ đúng thành phần.

```text
monthly\_debt\_payment + monthly\_income
→ DTI Debt Service
```

### Proxy

Không có đúng thành phần nhưng có biến gần nghĩa.

```text
Không có monthly debt payment
Có total outstanding debt + annual income
→ Outstanding Debt-to-Income
```

Không gọi proxy bằng tên của feature chuẩn.

### Cannot create

Thiếu thành phần bắt buộc.

```text
Có balance
Không có credit limit
→ Không tính Credit Utilization chính xác
```

\---

# 1\. Helper code dùng chung

```python
import pandas as pd
import numpy as np
```



## Cell 1 — Nạp dữ liệu và xem cột

Thay đường dẫn bằng file của bạn. Nếu bảng đã có sẵn trong notebook pipeline thì dùng chính DataFrame đó.

```python
import pandas as pd
import numpy as np

df = pd.read\\\\\\\_csv("your\\\\\\\_customer\\\\\\\_data.csv")
print(df.shape)
print(df.columns.tolist())
df.head()
```



## Cắt lịch sử theo observation date

Dùng cho các bảng payment, inquiry, account, balance, income history.

```python
def history\_before\_snapshot(df, events, event\_date\_col):
    base = df\[\["customer\_id", "observation\_date"]].copy()
    base\["\_row\_id"] = np.arange(len(base))
    base\["observation\_date"] = pd.to\_datetime(
        base\["observation\_date"], errors="coerce"
    )

    hist = events.copy()
    hist\[event\_date\_col] = pd.to\_datetime(
        hist\[event\_date\_col], errors="coerce"
    )

    joined = base.merge(
        hist,
        on="customer\_id",
        how="left",
        validate="many\_to\_many"
    )

    joined = joined\[
        joined\[event\_date\_col] < joined\["observation\_date"]
    ]

    return joined, base\["\_row\_id"]
```

> Nếu dataset có timestamp chính xác và chắc chắn event cùng ngày xảy ra \*\*trước giờ ra quyết định\*\*, có thể cân nhắc `<=`. Nếu chỉ có date, dùng `<` thận trọng hơn.

\---

# A. AFFORDABILITY

## Khả năng gánh và trả khoản vay

\---

## A1. Loan-to-Income — `FE\_LOAN\_TO\_INCOME`

**Cần:** `loan\_amount`, `annual\_income`

**Tạo:**

\[
Loan\\ to\\ Income = \\frac{Loan\\ Amount}{Annual\\ Income}
]

```python
df\["FE\_LOAN\_TO\_INCOME"] = (
    df\["loan\_amount"]
    / df\["annual\_income"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Đo quy mô khoản vay so với khả năng tạo thu nhập trong một năm. Tỷ lệ cao nghĩa là khoản vay lớn tương đối so với income.

**Lưu ý:** `FE\_CREDIT\_INCOME\_RATIO` trong schema Home Credit là cùng concept, không cần tạo thêm nếu đã có feature này.

\---

## A2. Installment-to-Monthly-Income — `FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME`

**Cần:** `monthly\_installment`, `annual\_income` hoặc `monthly\_income`

**Tạo:**

\[
Installment\\ Burden = \\frac{Monthly\\ Installment}{Monthly\\ Income}
]

```python
df\["FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME"] = (
    df\["monthly\_installment"]
    / (df\["annual\_income"].replace(0, np.nan) / 12)
)
```

Nếu income đã theo tháng:

```python
df\["FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME"] = (
    df\["monthly\_installment"]
    / df\["monthly\_income"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Cho biết riêng khoản vay đang xét chiếm bao nhiêu phần thu nhập tháng.

**Lưu ý:** `FE\_ANNUITY\_INCOME\_RATIO` là cùng family. Chỉ giữ một tên rõ ràng nếu công thức và kỳ thời gian giống nhau.

\---

## A3. Outstanding Debt-to-Income — `FE\_DEBT\_TO\_INCOME`

**Cần:** `total\_debt`, `annual\_income`

**Tạo:**

\[
Outstanding\\ Debt\\ to\\ Income =
\\frac{Total\\ Outstanding\\ Debt}{Annual\\ Income}
]

```python
df\["FE\_DEBT\_TO\_INCOME"] = (
    df\["total\_debt"]
    / df\["annual\_income"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Đo quy mô tổng dư nợ hiện tại so với income năm.

**Lưu ý:** Đây **không phải** DTI theo monthly debt service.

\---

## A4. DTI theo Debt Service — `FE\_DTI\_DEBT\_SERVICE`

**Cần:** `monthly\_debt\_payment`, `monthly\_income`

**Tạo:**

\[
DTI =
\\frac{Monthly\\ Debt\\ Payment}{Monthly\\ Income}
]

```python
df\["FE\_DTI\_DEBT\_SERVICE"] = (
    df\["monthly\_debt\_payment"]
    / df\["monthly\_income"].replace(0, np.nan)
)
```

Nếu income theo năm:

```python
monthly\_income = df\["annual\_income"] / 12

df\["FE\_DTI\_DEBT\_SERVICE"] = (
    df\["monthly\_debt\_payment"]
    / monthly\_income.replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Đo phần thu nhập hàng tháng đang phải dùng để phục vụ nợ. Đây là measure trực tiếp về repayment burden.

**Lưu ý:** Không nhầm với `FE\_DEBT\_TO\_INCOME`.

\---

## A5. Total Payment-to-Income / FOIR — `FE\_TOTAL\_PAYMENT\_TO\_INCOME`

**Cần:**

* `monthly\_income`
* `existing\_monthly\_payment`
* `proposed\_monthly\_payment`

**Tạo:**

\[
FOIR =
\\frac{Existing\\ Payment + Proposed\\ Payment}
{Monthly\\ Income}
]

```python
income = pd.to\_numeric(df\["monthly\_income"], errors="coerce")
old\_payment = pd.to\_numeric(
    df\["existing\_monthly\_payment"], errors="coerce"
)
new\_payment = pd.to\_numeric(
    df\["proposed\_monthly\_payment"], errors="coerce"
)

total\_payment = old\_payment + new\_payment

df\["FE\_TOTAL\_PAYMENT\_TO\_INCOME"] = (
    total\_payment.div(income.where(income > 0))
)
```

**Ý nghĩa phân tích:** Đo tổng repayment burden sau khi tính cả nghĩa vụ nợ cũ và khoản vay mới.

**Lưu ý:** Chỉ fill missing payment = 0 nếu dictionary xác nhận missing nghĩa là không có nghĩa vụ.

\---

## A6. Residual Income — `FE\_RESIDUAL\_INCOME\_AFTER\_DEBT`

**Cần:** cùng các cột ở A5.

**Tạo:**

\[
Residual\\ Income =
Income - Existing\\ Payment - Proposed\\ Payment
]

```python
df\["FE\_RESIDUAL\_INCOME\_AFTER\_DEBT"] = (
    income - old\_payment - new\_payment
)
```

**Ý nghĩa phân tích:** Cho biết số tiền tuyệt đối còn lại sau debt service. Hai khách cùng DTI có thể vẫn khác nhau mạnh về số tiền còn lại.

**Lưu ý:** Chưa phải disposable income nếu chưa trừ living expenses.

\---

## A7. Salary Coverage — `FE\_SALARY\_COVERAGE`

**Cần:** `monthly\_salary`, `monthly\_installment`

**Tạo:**

\[
Salary\\ Coverage =
\\frac{Monthly\\ Salary}{Monthly\\ Installment}
]

```python
df\["FE\_SALARY\_COVERAGE"] = (
    df\["monthly\_salary"]
    / df\["monthly\_installment"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Cho biết lương tháng cover khoản trả định kỳ bao nhiêu lần.

**Lưu ý:** Nếu khách có nhiều nguồn thu, salary không đại diện toàn bộ repayment capacity.

\---

## A8. Credit-to-Annuity — `FE\_CREDIT\_ANNUITY\_RATIO`

**Cần:** `loan\_amount`/`AMT\_CREDIT`, `annuity`/`AMT\_ANNUITY`

**Tạo:**

\[
Credit\\ to\\ Annuity =
\\frac{Credit\\ Amount}{Periodic\\ Payment}
]

```python
df\["FE\_CREDIT\_ANNUITY\_RATIO"] = (
    df\["AMT\_CREDIT"]
    / df\["AMT\_ANNUITY"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Có thể là proxy thô cho số kỳ / cấu trúc thời hạn trả nợ khi payment schedule phù hợp.

**Lưu ý:** Không coi đây là số tháng chính xác nếu có lãi, phí hoặc lịch trả thay đổi.

\---

# B. LIQUIDITY / CASA

## Khả năng thanh khoản và buffer tài chính

\---

## B1. CASA Balance — `FE\_CASA\_BALANCE`

**Cần:** `current\_account\_balance`, `savings\_account\_balance`

```python
df\["FE\_CASA\_BALANCE"] = (
    df\["current\_account\_balance"].fillna(0)
    + df\["savings\_account\_balance"].fillna(0)
)
```

**Ý nghĩa phân tích:** Phản ánh lượng tiền thanh khoản cao khách đang duy trì trong current + savings account.

**Lưu ý:** Chỉ fill missing = 0 khi dictionary xác nhận không có account/balance.

\---

## B2. Customer CASA Ratio — `FE\_CASA\_RATIO`

**Cần:** current, savings, term deposit

\[
CASA\\ Ratio =
\\frac{Current + Savings}
{Current + Savings + Term\\ Deposit}
]

```python
casa = (
    df\["current\_account\_balance"]
    + df\["savings\_account\_balance"]
)

total\_deposit = casa + df\["term\_deposit\_balance"]

df\["FE\_CASA\_RATIO"] = (
    casa / total\_deposit.replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Đo tỷ trọng tiền gửi có tính thanh khoản cao trong tổng deposit của khách.

**Lưu ý:** Đây là customer-level CASA, không phải CASA ratio toàn ngân hàng.

\---

## B3. Average CASA — `FE\_AVG\_CASA\_BALANCE`

**Cần:** snapshot current + savings theo thời gian

```python
monthly\["CASA"] = (
    monthly\["current\_balance"]
    + monthly\["savings\_balance"]
)

avg\_casa = (
    monthly.groupby("customer\_id")\["CASA"]
    .mean()
    .rename("FE\_AVG\_CASA\_BALANCE")
)
```

**Ý nghĩa phân tích:** Mức CASA điển hình khách duy trì, ổn định hơn một snapshot đơn lẻ.

**Lưu ý:** Chỉ dùng snapshot trước observation date.

\---

## B4. CASA-to-Income — `FE\_CASA\_TO\_INCOME`

**Cần:** `avg\_casa\_balance`, `monthly\_income`

\[
CASA\\ to\\ Income =
\\frac{Average\\ CASA}{Monthly\\ Income}
]

```python
df\["FE\_CASA\_TO\_INCOME"] = (
    df\["avg\_casa\_balance"]
    / df\["monthly\_income"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Chuẩn hóa CASA theo quy mô thu nhập để so sánh khách hàng công bằng hơn.

\---

## B5. CASA Buffer Months — `FE\_CASA\_BUFFER\_MONTHS`

**Cần:** `FE\_CASA\_BALANCE`, `monthly\_debt\_payment`

```python
df\["FE\_CASA\_BUFFER\_MONTHS"] = (
    df\["FE\_CASA\_BALANCE"]
    / df\["monthly\_debt\_payment"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Ước lượng CASA hiện tại đủ cover bao nhiêu tháng debt service.

\---

## B6. Savings Buffer Months — `FE\_SAVINGS\_BUFFER\_MONTHS`

**Cần:** `liquid\_savings`, existing payment, proposed payment

```python
savings = pd.to\_numeric(
    df\["liquid\_savings"], errors="coerce"
)

total\_payment = old\_payment + new\_payment

df\["FE\_SAVINGS\_BUFFER\_MONTHS"] = (
    savings.div(total\_payment.where(total\_payment > 0))
)
```

**Ý nghĩa phân tích:** Đo mức buffer thanh khoản bằng số tháng repayment.

\---

## B7. Average Monthly Inflow — `FE\_AVG\_MONTHLY\_INFLOW`

**Cần:** transaction history có `credit\_amount`

Ví dụ:

```python
tx\["transaction\_date"] = pd.to\_datetime(
    tx\["transaction\_date"], errors="coerce"
)
tx\["month"] = tx\["transaction\_date"].dt.to\_period("M")

monthly\_inflow = (
    tx.groupby(\["customer\_id", "month"])\["credit\_amount"]
    .sum()
)

avg\_inflow = (
    monthly\_inflow.groupby("customer\_id")
    .mean()
    .rename("FE\_AVG\_MONTHLY\_INFLOW")
)
```

**Ý nghĩa phân tích:** Proxy dòng tiền thực tế đi vào tài khoản, hữu ích khi declared income chưa phản ánh đủ cash generation.

**Lưu ý:** Loại loan disbursement, transfer nội bộ nếu có transaction type.

\---

## B8. Net Cash Flow — `FE\_NET\_CASH\_FLOW`

\[
Net\\ Cash\\ Flow = Credits - Debits
]

```python
df\["FE\_NET\_CASH\_FLOW"] = (
    df\["monthly\_credit\_inflow"]
    - df\["monthly\_debit\_outflow"]
)
```

**Ý nghĩa phân tích:** Khách đang tạo surplus hay tiêu vượt inflow.

\---

## B9. Cashflow Coverage — `FE\_CASHFLOW\_COVERAGE`

\[
Cashflow\\ Coverage =
\\frac{Average\\ Monthly\\ Inflow}{Monthly\\ Debt\\ Payment}
]

```python
df\["FE\_CASHFLOW\_COVERAGE"] = (
    df\["average\_monthly\_inflow"]
    / df\["monthly\_debt\_payment"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Dòng tiền vào thực tế cover nghĩa vụ nợ bao nhiêu lần.

\---

## B10. Average Balance — `FE\_AVG\_BALANCE`

```python
avg\_balance = (
    monthly.groupby("customer\_id")\["balance"]
    .mean()
    .rename("FE\_AVG\_BALANCE")
)
```

**Ý nghĩa phân tích:** Mức số dư duy trì điển hình.

\---

## B11. Minimum Balance — `FE\_MIN\_BALANCE`

```python
min\_balance = (
    monthly.groupby("customer\_id")\["balance"]
    .min()
    .rename("FE\_MIN\_BALANCE")
)
```

**Ý nghĩa phân tích:** Mức liquidity buffer thấp nhất trong kỳ quan sát.

\---

## B12. Balance Volatility — `FE\_BALANCE\_CV`

\[
Balance\\ CV =
\\frac{Std(Balance)}{Mean(Balance)}
]

```python
balance\_stats = (
    monthly.groupby("customer\_id")\["balance"]
    .agg(\["mean", "std"])
)

balance\_stats\["FE\_BALANCE\_CV"] = (
    balance\_stats\["std"]
    / balance\_stats\["mean"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Đo độ ổn định của thanh khoản theo thời gian.

\---

## B13. Savings Rate — `FE\_SAVINGS\_RATE`

**Cần:** `income`, `expenses`

\[
Savings\\ Rate =
\\frac{Income - Expenses}{Income}
]

```python
df\["FE\_SAVINGS\_RATE"] = (
    (df\["income"] - df\["expenses"])
    / df\["income"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Phần income còn tạo được surplus sau chi tiêu.

\---

# C. EXPOSURE

## Mức độ sử dụng tín dụng và nghĩa vụ đang mở

\---

## C1. Credit Utilization — `FE\_CREDIT\_UTILIZATION`

**Cần:** `credit\_card\_balance`, `credit\_limit`

```python
df\["FE\_CREDIT\_UTILIZATION"] = (
    df\["credit\_card\_balance"]
    / df\["credit\_limit"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Phần hạn mức quay vòng đang được sử dụng.

\---

## C2. Utilization Max — `FE\_UTILIZATION\_MAX`

```python
monthly\["FE\_UTIL"] = (
    monthly\["balance"]
    / monthly\["credit\_limit"].replace(0, np.nan)
)

util\_max = (
    monthly.groupby("customer\_id")\["FE\_UTIL"]
    .max()
    .rename("FE\_UTILIZATION\_MAX")
)
```

**Ý nghĩa phân tích:** Bắt đỉnh sử dụng hạn mức, ngay cả khi mức trung bình thấp.

\---

## C3. Utilization Volatility — `FE\_UTILIZATION\_STD`

```python
util\_std = (
    monthly.groupby("customer\_id")\["FE\_UTIL"]
    .std()
    .rename("FE\_UTILIZATION\_STD")
)
```

**Ý nghĩa phân tích:** Độ biến động của việc sử dụng tín dụng.

\---

## C4. Utilization Change — `FE\_UTILIZATION\_CHANGE\_3M`

**Cần:** monthly credit snapshots

```python
hist, row\_id = history\_before\_snapshot(
    df, monthly\_credit, "snapshot\_date"
)

balance = pd.to\_numeric(hist\["balance"], errors="coerce")
limit\_ = pd.to\_numeric(hist\["credit\_limit"], errors="coerce")

hist\["\_utilization"] = (
    balance.div(limit\_.where(limit\_ > 0))
)

recent = hist\[
    hist\["snapshot\_date"]
    >= hist\["observation\_date"] - pd.Timedelta(days=45)
].groupby("\_row\_id")\["\_utilization"].mean()

prior\_rows = hist\[
    (
        hist\["snapshot\_date"]
        >= hist\["observation\_date"] - pd.Timedelta(days=135)
    )
    \& (
        hist\["snapshot\_date"]
        < hist\["observation\_date"] - pd.Timedelta(days=90)
    )
]

prior = (
    prior\_rows.groupby("\_row\_id")\["\_utilization"]
    .mean()
)

change = recent - prior

df\["FE\_UTILIZATION\_CHANGE\_3M"] = (
    row\_id.map(change).to\_numpy()
)
```

**Ý nghĩa phân tích:** Số dương cho thấy mức sử dụng hạn mức gần đây đang tăng.

\---

## C5. Open Account Count — `FE\_OPEN\_ACCOUNT\_COUNT`

```python
open\_accounts = accounts\[
    accounts\["account\_status"] == "open"
]

open\_count = (
    open\_accounts.groupby("customer\_id")
    .size()
    .rename("FE\_OPEN\_ACCOUNT\_COUNT")
)
```

**Ý nghĩa phân tích:** Số credit account đang mở, phản ánh độ phức tạp của exposure.

\---

## C6. Active Loan Count — `FE\_ACTIVE\_LOAN\_COUNT`

```python
active = loans\[loans\["loan\_status"] == "active"]

active\_count = (
    active.groupby("customer\_id")
    .size()
    .rename("FE\_ACTIVE\_LOAN\_COUNT")
)
```

**Ý nghĩa phân tích:** Số khoản vay active. Khác open account count nếu một account không tương đương một loan.

\---

## C7. Active Debt Service Monthly — `FE\_ACTIVE\_DEBT\_SERVICE\_MONTHLY`

**Cần:** account open date, close date, monthly payment

```python
base = df\[\["customer\_id", "observation\_date"]].copy()
base\["\_row\_id"] = np.arange(len(base))
base\["observation\_date"] = pd.to\_datetime(
    base\["observation\_date"], errors="coerce"
)

acc = accounts\[
    \["customer\_id", "open\_date", "close\_date", "monthly\_payment"]
].copy()

acc\["open\_date"] = pd.to\_datetime(
    acc\["open\_date"], errors="coerce"
)
acc\["close\_date"] = pd.to\_datetime(
    acc\["close\_date"], errors="coerce"
)
acc\["monthly\_payment"] = pd.to\_numeric(
    acc\["monthly\_payment"], errors="coerce"
)

joined = base.merge(
    acc,
    on="customer\_id",
    how="left",
    validate="many\_to\_many"
)

active = joined\[
    (joined\["open\_date"] < joined\["observation\_date"])
    \& (
        joined\["close\_date"].isna()
        | (joined\["close\_date"] >= joined\["observation\_date"])
    )
]

monthly\_sum = (
    active.groupby("\_row\_id")\["monthly\_payment"]
    .sum(min\_count=1)
)

df\["FE\_ACTIVE\_DEBT\_SERVICE\_MONTHLY"] = (
    base\["\_row\_id"].map(monthly\_sum).to\_numpy()
)
```

**Ý nghĩa phân tích:** Đo số tiền repayment đang phải trả, không chỉ đếm số khoản.

\---

## C8. Total Outstanding Debt — `FE\_TOTAL\_OUTSTANDING\_DEBT`

```python
df\["FE\_TOTAL\_OUTSTANDING\_DEBT"] = (
    df\["mortgage\_balance"]
    + df\["credit\_card\_balance"]
    + df\["personal\_loan\_balance"]
    + df\["auto\_loan\_balance"]
)
```

**Ý nghĩa phân tích:** Tổng exposure tín dụng hiện tại.

**Lưu ý:** Tránh double-count cùng một khoản nợ.

\---

## C9. Inquiry Count 30D — `FE\_INQUIRIES\_30D`

```python
hist, row\_id = history\_before\_snapshot(
    df, inquiries, "inquiry\_date"
)

recent = hist\[
    hist\["inquiry\_date"]
    >= hist\["observation\_date"] - pd.Timedelta(days=30)
]

counts = recent.groupby("\_row\_id").size()

df\["FE\_INQUIRIES\_30D"] = (
    row\_id.map(counts)
    .fillna(0)
    .astype("int32")
    .to\_numpy()
)
```

**Ý nghĩa phân tích:** Nhu cầu tín dụng rất gần prediction date.

\---

## C10. Inquiry Count 3M — `FE\_INQUIRIES\_3M`

```python
hist, row\_id = history\_before\_snapshot(
    df, inquiries, "inquiry\_date"
)

recent = hist\[
    hist\["inquiry\_date"]
    >= hist\["observation\_date"] - pd.DateOffset(months=3)
]

counts = recent.groupby("\_row\_id").size()

df\["FE\_INQUIRIES\_3M"] = (
    row\_id.map(counts).fillna(0).to\_numpy()
)
```

**Ý nghĩa phân tích:** Credit-seeking behaviour trong cửa sổ trung hạn gần.

\---

## C11. Inquiry Count 6M — `FE\_INQUIRIES\_6M`

```python
inquiries\["inquiry\_date"] = pd.to\_datetime(
    inquiries\["inquiry\_date"]
)

inquiries\_window = inquiries.merge(
    df\[\["customer\_id", "observation\_date"]],
    on="customer\_id",
    how="inner"
)

cutoff = pd.to\_datetime(
    inquiries\_window\["observation\_date"]
)

start = cutoff - pd.DateOffset(months=6)

inquiries\_6m = inquiries\_window\[
    (inquiries\_window\["inquiry\_date"] < cutoff)
    \& (inquiries\_window\["inquiry\_date"] > start)
]

inquiry\_count = (
    inquiries\_6m.groupby("customer\_id")
    .size()
    .rename("FE\_INQUIRIES\_6M")
)
```

**Ý nghĩa phân tích:** Mức độ tìm kiếm tín dụng trong nửa năm gần nhất.

\---

## C12. Inquiry Count 12M — `FE\_INQUIRIES\_12M`

```python
hist, row\_id = history\_before\_snapshot(
    df, inquiries, "inquiry\_date"
)

recent = hist\[
    hist\["inquiry\_date"]
    >= hist\["observation\_date"] - pd.DateOffset(months=12)
]

counts = recent.groupby("\_row\_id").size()

df\["FE\_INQUIRIES\_12M"] = (
    row\_id.map(counts).fillna(0).to\_numpy()
)
```

**Ý nghĩa phân tích:** Mức độ tìm kiếm tín dụng trong một horizon rộng hơn.

\---

# D. REPAYMENT BEHAVIOUR

## Hành vi trả nợ trong quá khứ

\---

## D1. Days Late / DPD per payment — `FE\_DAYS\_LATE`

```python
payments\["due\_date"] = pd.to\_datetime(
    payments\["due\_date"]
)
payments\["payment\_date"] = pd.to\_datetime(
    payments\["payment\_date"]
)

payments\["FE\_DAYS\_LATE"] = (
    payments\["payment\_date"]
    - payments\["due\_date"]
).dt.days.clip(lower=0)
```

**Ý nghĩa phân tích:** Mức trễ của từng kỳ thanh toán.

**Lưu ý:** Kỳ chưa trả không được mặc định là đúng hạn.

\---

## D2. Late Payment Rate — `FE\_LATE\_PAYMENT\_RATE`

```python
payments\["FE\_WAS\_LATE"] = (
    payments\["FE\_DAYS\_LATE"] > 0
).astype(int)

late\_rate = (
    payments.groupby("customer\_id")\["FE\_WAS\_LATE"]
    .mean()
    .rename("FE\_LATE\_PAYMENT\_RATE")
)
```

**Ý nghĩa phân tích:** Tần suất khách từng trả chậm.

\---

## D3. Late 30+ Rate — `FE\_LATE\_30\_RATE`

```python
payments\["FE\_LATE\_30"] = (
    payments\["FE\_DAYS\_LATE"] >= 30
).astype(int)

late\_30\_rate = (
    payments.groupby("customer\_id")\["FE\_LATE\_30"]
    .mean()
    .rename("FE\_LATE\_30\_RATE")
)
```

**Ý nghĩa phân tích:** Tập trung vào delinquency nghiêm trọng hơn việc chỉ trễ vài ngày.

\---

## D4. Payment Shortfall Rate — `FE\_PAYMENT\_SHORTFALL\_RATE`

```python
payments\["FE\_SHORTFALL"] = (
    payments\["scheduled\_amount"]
    - payments\["actual\_amount"]
).clip(lower=0)

payments\["FE\_SHORTFALL\_RATE"] = (
    payments\["FE\_SHORTFALL"]
    / payments\["scheduled\_amount"].replace(0, np.nan)
)

shortfall\_rate = (
    payments.groupby("customer\_id")\["FE\_SHORTFALL\_RATE"]
    .mean()
    .rename("FE\_PAYMENT\_SHORTFALL\_RATE")
)
```

**Ý nghĩa phân tích:** Phát hiện trả thiếu tiền ngay cả khi có payment date.

\---

## D5. Payment History Count — `FE\_PAYMENT\_HISTORY\_COUNT`

```python
payment\_count = (
    payments.groupby("customer\_id")
    .size()
    .rename("FE\_PAYMENT\_HISTORY\_COUNT")
)
```

**Ý nghĩa phân tích:** Độ dày lịch sử repayment có thể quan sát.

\---

## D6. No Payment History — `FE\_NO\_PAYMENT\_HISTORY`

```python
df\["FE\_NO\_PAYMENT\_HISTORY"] = (
    df\["FE\_PAYMENT\_HISTORY\_COUNT"]
    .isna()
    .astype("int8")
)
```

**Ý nghĩa phân tích:** Phân biệt khách chưa có lịch sử với khách có lịch sử tốt.

\---

## D7. Days Since Last Late — `FE\_DAYS\_SINCE\_LAST\_LATE`

```python
late\_events = payments\[
    payments\["FE\_DAYS\_LATE"] > 0
].copy()

last\_late = (
    late\_events.groupby("customer\_id")\["payment\_date"]
    .max()
    .rename("\_last\_late\_date")
)

df = df.merge(
    last\_late,
    on="customer\_id",
    how="left"
)

df\["FE\_DAYS\_SINCE\_LAST\_LATE"] = (
    pd.to\_datetime(df\["observation\_date"])
    - pd.to\_datetime(df\["\_last\_late\_date"])
).dt.days

df = df.drop(columns="\_last\_late\_date")
```

**Ý nghĩa phân tích:** Phân biệt vi phạm cũ với vi phạm mới xảy ra.

\---

## D8. Recent Late Rate 6M — `FE\_LATE\_RATE\_6M`

```python
payments\_window = payments.merge(
    df\[\["customer\_id", "observation\_date"]],
    on="customer\_id",
    how="inner"
)

cutoff = pd.to\_datetime(
    payments\_window\["observation\_date"]
)

start\_6m = cutoff - pd.DateOffset(months=6)

recent\_6m = payments\_window\[
    (payments\_window\["due\_date"] < cutoff)
    \& (payments\_window\["due\_date"] > start\_6m)
]

late\_rate\_6m = (
    recent\_6m.groupby("customer\_id")\["FE\_WAS\_LATE"]
    .mean()
    .rename("FE\_LATE\_RATE\_6M")
)
```

**Ý nghĩa phân tích:** Nhạy hơn lifetime rate khi behaviour đang xấu đi gần đây.

\---

## D9. Late Rate Trend — `FE\_LATE\_RATE\_TREND\_6M\_VS\_PRIOR\_6M`

```python
start\_prior\_6m = cutoff - pd.DateOffset(months=12)

prior\_6m = payments\_window\[
    (payments\_window\["due\_date"] <= start\_6m)
    \& (payments\_window\["due\_date"] > start\_prior\_6m)
]

late\_rate\_prior\_6m = (
    prior\_6m.groupby("customer\_id")\["FE\_WAS\_LATE"]
    .mean()
    .rename("\_late\_rate\_prior\_6m")
)

trend = (
    late\_rate\_6m.to\_frame()
    .join(late\_rate\_prior\_6m, how="outer")
)

trend\["FE\_LATE\_RATE\_TREND\_6M\_VS\_PRIOR\_6M"] = (
    trend\["FE\_LATE\_RATE\_6M"]
    - trend\["\_late\_rate\_prior\_6m"]
)
```

**Ý nghĩa phân tích:** Số dương cho thấy repayment behaviour gần đây đang xấu đi.

\---

## D10. Max DPD — `FE\_MAX\_DPD`

```python
payments\["DPD"] = (
    pd.to\_datetime(payments\["payment\_date"])
    - pd.to\_datetime(payments\["due\_date"])
).dt.days.clip(lower=0)

max\_dpd = (
    payments.groupby("customer\_id")\["DPD"]
    .max()
    .rename("FE\_MAX\_DPD")
)
```

**Ý nghĩa phân tích:** Mức delinquency nghiêm trọng nhất khách từng gặp.

\---

# E. COLLATERAL / LOAN STRUCTURE

## Đặc điểm khoản vay và tài sản bảo đảm

\---

## E1. Loan-to-Value — `FE\_LOAN\_TO\_VALUE`

**Cần:** `requested\_loan\_amount`, `collateral\_value`

\[
LTV =
\\frac{Requested\\ Loan}{Collateral\\ Value}
]

```python
requested = pd.to\_numeric(
    df\["requested\_loan\_amount"], errors="coerce"
)
collateral = pd.to\_numeric(
    df\["collateral\_value"], errors="coerce"
)

df\["FE\_LOAN\_TO\_VALUE"] = (
    requested.div(collateral.where(collateral > 0))
)
```

**Ý nghĩa phân tích:** Khoản vay lớn đến đâu so với giá trị tài sản bảo đảm.

**Lưu ý:** Chỉ phù hợp với secured loan.

\---

## E2. Collateral Value Missing — `FE\_COLLATERAL\_VALUE\_MISSING`

```python
df\["FE\_COLLATERAL\_VALUE\_MISSING"] = (
    collateral.isna().astype("int8")
)
```

**Ý nghĩa phân tích:** Phân biệt chưa có collateral valuation với giá trị collateral bằng 0.

\---

## E3. Age at Loan Maturity — `FE\_AGE\_AT\_MATURITY`

**Cần:** `age`, `loan\_term\_months`

\[
Age\\ at\\ Maturity =
Age + \\frac{Term}{12}
]

```python
df\["FE\_AGE\_AT\_MATURITY"] = (
    df\["age"]
    + df\["loan\_term\_months"] / 12
)
```

**Ý nghĩa phân tích:** Tuổi khách tại thời điểm khoản vay kết thúc, phù hợp hơn tuổi hiện tại với khoản vay dài hạn.

\---

# F. STABILITY

## Độ ổn định nguồn thu và quan hệ

\---

## F1. Employment Tenure — `FE\_EMPLOYMENT\_TENURE\_MONTHS`

```python
as\_of = pd.to\_datetime(
    df\["observation\_date"], errors="coerce"
)
started = pd.to\_datetime(
    df\["employment\_start\_date"], errors="coerce"
)

days = (as\_of - started).dt.days

df\["FE\_EMPLOYMENT\_TENURE\_MONTHS"] = (
    days.where(days >= 0) / 30.44
)
```

**Ý nghĩa phân tích:** Độ dài của nguồn thu nhập/công việc hiện tại.

\---

## F2. Income CV 6M — `FE\_INCOME\_CV\_6M`

```python
hist, row\_id = history\_before\_snapshot(
    df, income\_history, "income\_date"
)

hist = hist\[
    hist\["income\_date"]
    >= hist\["observation\_date"] - pd.DateOffset(months=6)
].copy()

hist\["income\_amount"] = pd.to\_numeric(
    hist\["income\_amount"], errors="coerce"
)

summary = (
    hist.groupby("\_row\_id")\["income\_amount"]
    .agg(\["count", "mean", "std"])
)

cv = (
    summary\["std"]
    .div(summary\["mean"].where(summary\["mean"] > 0))
)

cv = cv.where(summary\["count"] >= 3)

df\["FE\_INCOME\_CV\_6M"] = (
    row\_id.map(cv).to\_numpy()
)
```

**Ý nghĩa phân tích:** Hai khách cùng average income nhưng một người có income biến động mạnh hơn.

\---

## F3. Income Months Observed — `FE\_INCOME\_MONTHS\_OBSERVED`

```python
df\["FE\_INCOME\_MONTHS\_OBSERVED"] = (
    row\_id.map(summary\["count"]).to\_numpy()
)
```

**Ý nghĩa phân tích:** Đo độ dày dữ liệu dùng để tính income stability.

\---

## F4. Relationship Tenure — `FE\_RELATIONSHIP\_MONTHS`

```python
df\["FE\_RELATIONSHIP\_MONTHS"] = (
    (
        pd.to\_datetime(df\["observation\_date"])
        - pd.to\_datetime(df\["customer\_since"])
    ).dt.days / 30.44
)
```

**Ý nghĩa phân tích:** Khách đã có quan hệ với tổ chức tài chính bao lâu.

\---

# G. CREDIT PROFILE

## Độ dày lịch sử và điểm tín dụng

\---

## G1. Credit History Months — `FE\_CREDIT\_HISTORY\_MONTHS`

```python
accounts\["open\_date"] = pd.to\_datetime(
    accounts\["open\_date"]
)

first\_open = (
    accounts.groupby("customer\_id")\["open\_date"]
    .min()
    .rename("\_first\_open")
)

df = df.merge(
    first\_open,
    on="customer\_id",
    how="left"
)

df\["FE\_CREDIT\_HISTORY\_MONTHS"] = (
    (
        pd.to\_datetime(df\["observation\_date"])
        - pd.to\_datetime(df\["\_first\_open"])
    ).dt.days / 30.44
)

df = df.drop(columns="\_first\_open")
```

**Ý nghĩa phân tích:** Lịch sử tín dụng càng dài thì càng có nhiều evidence hành vi để quan sát, nhưng không đồng nghĩa tự động low risk.

\---

## G2. External Score Mean — `FE\_EXT\_SOURCE\_MEAN`

```python
ext\_cols = \[
    "EXT\_SOURCE\_1",
    "EXT\_SOURCE\_2",
    "EXT\_SOURCE\_3"
]

df\["FE\_EXT\_SOURCE\_MEAN"] = (
    df\[ext\_cols].mean(axis=1)
)
```

**Ý nghĩa phân tích:** Tổng hợp nhiều score bên ngoài thành một mức tín dụng tổng quát.

\---

## G3. External Score STD — `FE\_EXT\_SOURCE\_STD`

```python
df\["FE\_EXT\_SOURCE\_STD"] = (
    df\[ext\_cols].std(axis=1)
)
```

**Ý nghĩa phân tích:** Mức độ các nguồn score đồng thuận hay mâu thuẫn với nhau.

\---

## G4. External Score Count — `FE\_EXT\_SOURCE\_COUNT`

```python
df\["FE\_EXT\_SOURCE\_COUNT"] = (
    df\[ext\_cols].notna().sum(axis=1)
)
```

**Ý nghĩa phân tích:** Độ phủ của external scoring information.

\---

## G5. Score Missing — `FE\_SCORE\_MISSING`

```python
score = pd.to\_numeric(
    df\["credit\_score"], errors="coerce"
)

df\["FE\_SCORE\_MISSING"] = (
    score.isna().astype("int8")
)
```

**Ý nghĩa phân tích:** Thin-file / thiếu thông tin khác với credit score thấp.

\---

# H. CUSTOMER CONTEXT

## Bối cảnh cá nhân và quan hệ sản phẩm

\---

## H1. Age Years — `FE\_AGE\_YEARS`

Nếu schema Home Credit:

```python
df\["FE\_AGE\_YEARS"] = (
    -df\["DAYS\_BIRTH"] / 365.25
)
```

Nếu có DOB:

```python
dob = pd.to\_datetime(
    df\["date\_of\_birth"], errors="coerce"
)
obs = pd.to\_datetime(
    df\["observation\_date"], errors="coerce"
)

df\["FE\_AGE\_YEARS"] = (
    (obs - dob).dt.days / 365.25
)
```

**Ý nghĩa phân tích:** Tuổi tại đúng prediction date.

\---

## H2. Income Missing — `FE\_INCOME\_MISSING`

```python
income = pd.to\_numeric(
    df\["monthly\_income"], errors="coerce"
)

df\["FE\_INCOME\_MISSING"] = (
    income.isna().astype("int8")
)
```

**Ý nghĩa phân tích:** Thiếu income có thể là một trạng thái quy trình/thin file riêng, không phải income = 0.

\---

## H3. Income per Dependent — `FE\_INCOME\_PER\_DEPENDENT`

```python
df\["FE\_INCOME\_PER\_DEPENDENT"] = (
    df\["income"]
    / (df\["number\_of\_dependents"] + 1)
)
```

**Ý nghĩa phân tích:** Đặt income vào bối cảnh gánh nặng household.

\---

## H4. Product Holding Count — `FE\_PRODUCT\_COUNT`

```python
product\_cols = \[
    "has\_savings",
    "has\_current\_account",
    "has\_credit\_card",
    "has\_term\_deposit",
]

df\["FE\_PRODUCT\_COUNT"] = (
    df\[product\_cols].sum(axis=1)
)
```

**Ý nghĩa phân tích:** Proxy cho relationship depth.

**Lưu ý:** Không mặc định product count cao = risk thấp.

\---

# I. SME

## Feature phù hợp hơn cho doanh nghiệp / hộ kinh doanh

\---

## I1. DSCR — `FE\_DSCR`

\[
DSCR =
\\frac{Net\\ Operating\\ Income}{Debt\\ Service}
]

```python
df\["FE\_DSCR"] = (
    df\["net\_operating\_income"]
    / df\["annual\_debt\_service"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Dòng tiền hoạt động có đủ cover nghĩa vụ trả nợ hay không.

\---

## I2. Debt-to-Assets — `FE\_DEBT\_TO\_ASSETS`

\[
Debt\\ to\\ Assets =
\\frac{Total\\ Debt}{Total\\ Assets}
]

```python
df\["FE\_DEBT\_TO\_ASSETS"] = (
    df\["total\_debt"]
    / df\["total\_assets"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Mức leverage theo tài sản.

\---

## I3. Debt-to-Equity — `FE\_DEBT\_TO\_EQUITY`

\[
Debt\\ to\\ Equity =
\\frac{Total\\ Debt}{Equity}
]

```python
df\["FE\_DEBT\_TO\_EQUITY"] = (
    df\["total\_debt"]
    / df\["equity"].replace(0, np.nan)
)
```

**Ý nghĩa phân tích:** Mức sử dụng vốn vay so với vốn chủ.

**Lưu ý:** Equity âm phải được xử lý/diễn giải riêng.

\---

# J. ADVANCED

## Feature nâng cao, ưu tiên sau khi các nhóm domain chính đã ổn

\---

## J1. Polynomial / Interaction Features — `FE\_POLY\_\*`

**Cần:** một nhóm feature numeric có lý do để thử interaction.

Ví dụ từ Home Credit:

```python
from sklearn.preprocessing import PolynomialFeatures

poly\_cols = \[
    "EXT\_SOURCE\_1",
    "EXT\_SOURCE\_2",
    "EXT\_SOURCE\_3",
    "DAYS\_BIRTH"
]

poly\_train\_input = train\_df\[poly\_cols].copy()
poly\_valid\_input = valid\_df\[poly\_cols].copy()

poly\_medians = poly\_train\_input.median()

poly\_train\_input = (
    poly\_train\_input.fillna(poly\_medians)
)
poly\_valid\_input = (
    poly\_valid\_input.fillna(poly\_medians)
)

poly\_transformer = PolynomialFeatures(
    degree=3,
    include\_bias=False
)

poly\_train\_array = (
    poly\_transformer.fit\_transform(poly\_train\_input)
)

poly\_valid\_array = (
    poly\_transformer.transform(poly\_valid\_input)
)

poly\_names = (
    poly\_transformer.get\_feature\_names\_out(poly\_cols)
)
```

**Ý nghĩa phân tích:** Cho model tuyến tính khả năng học quan hệ phi tuyến hoặc tương tác giữa nhiều score/tuổi.

**Lưu ý:**

* Fit transformer trên train, không fit validation/test.
* Tree boosting thường tự học interaction, nên polynomial có thể dư thừa.
* Chỉ thử sau khi feature domain đã ổn.

\---

# 2\. Feature aliases / duplicate đã chuẩn hóa

Các tên dưới đây không cần tồn tại song song nếu công thức giống nhau:

|Alias / tên cũ|Tên chuẩn trong file này|
|-|-|
|`FE\_CREDIT\_INCOME\_RATIO`|`FE\_LOAN\_TO\_INCOME`|
|`FE\_ANNUITY\_INCOME\_RATIO`|`FE\_INSTALLMENT\_TO\_MONTHLY\_INCOME` khi kỳ thời gian tương đương|
|DTI dùng outstanding balance|`FE\_DEBT\_TO\_INCOME`|
|DTI dùng monthly debt service|`FE\_DTI\_DEBT\_SERVICE`|
|FOIR|`FE\_TOTAL\_PAYMENT\_TO\_INCOME`|

\---

# 3\. Priority Guide

## P0 — bắt buộc trước feature engineering

```text
Target đúng
Observation date đúng
Không leakage
Unit đúng
ID / row unit đúng
```

## P1 — thử đầu tiên nếu có data

### Retail

```text
A4 DTI Debt Service
A5 Total Payment-to-Income / FOIR
A6 Residual Income
A1 Loan-to-Income
C1 Credit Utilization
D2 Late Payment Rate
D10 Max DPD
D7 Days Since Last Late
G1 Credit History Length
```

### Nếu có lịch sử account / transaction

```text
B3 Average CASA
B9 Cashflow Coverage
C7 Active Debt Service
C9–C12 Recent Inquiry Counts
D8 Recent Late Rate
D9 Late Rate Trend
F2 Income CV
```

## P2 — bổ sung khi có thời gian

```text
CASA family
Balance stability
Relationship tenure
Product count
Age at maturity
External score aggregation
```

## P3 — chỉ thử sau cùng

```text
Polynomial features
Complex interactions
Nhiều bin/rule thủ công
```

\---

# 4\. Master Lookup Table

|Raw columns có sẵn|Feature có thể tạo|Nhóm|
|-|-|-|
|Loan + Annual Income|Loan-to-Income|A|
|Monthly installment + Income|Installment-to-Income|A|
|Total outstanding debt + Income|Outstanding Debt-to-Income|A|
|Monthly debt payment + Monthly income|DTI Debt Service|A|
|Existing + Proposed payment + Income|FOIR / Total Payment-to-Income|A|
|Income + Payments|Residual Income|A|
|Salary + Installment|Salary Coverage|A|
|Current + Savings|CASA Balance|B|
|CASA + Term Deposit|CASA Ratio|B|
|CASA history|Average CASA|B|
|CASA + Income|CASA-to-Income|B|
|CASA + Debt payment|CASA Buffer|B|
|Savings + Debt payment|Savings Buffer|B|
|Transactions|Average Inflow / Net Cash Flow|B|
|Inflow + Debt payment|Cashflow Coverage|B|
|Balance snapshots|Average / Minimum / Volatility|B|
|Balance + Credit Limit|Credit Utilization|C|
|Utilization history|Max / STD / Change|C|
|Account history|Open Account Count|C|
|Loan history|Active Loan Count|C|
|Accounts + Monthly Payment|Active Debt Service|C|
|Loan balances|Total Outstanding Debt|C|
|Inquiry dates|Inquiry 30D / 3M / 6M / 12M|C|
|Due date + Payment date|Days Late / Max DPD|D|
|Payment history|Late Rate / Late 30 Rate|D|
|Scheduled + Actual payment|Shortfall Rate|D|
|Payment dates|Days Since Last Late|D|
|Recent history|Recent Late Rate / Trend|D|
|Loan + Collateral|LTV|E|
|Age + Loan Term|Age at Maturity|E|
|Employment Start + Observation|Employment Tenure|F|
|Income history|Income CV|F|
|Customer Since + Observation|Relationship Tenure|F|
|Account open date|Credit History Length|G|
|External scores|Mean / STD / Count|G|
|Credit score missing|Score Missing|G|
|DOB + Observation|Age|H|
|Income + Dependents|Income per Dependent|H|
|Product flags|Product Count|H|
|Operating Income + Debt Service|DSCR|I|
|Debt + Assets|Debt-to-Assets|I|
|Debt + Equity|Debt-to-Equity|I|

\---

# 5\. Interpretation Template

Khi feature cho tín hiệu mạnh, không viết:

> "Feature này gây ra bad debt."

Viết theo cấu trúc:

### Observation

> `FE\_DTI\_DEBT\_SERVICE` có bad-debt rate cao hơn rõ rệt ở các bin trên.

### Interpretation

> Điều này cho thấy repayment burden có liên hệ với risk trong sample hiện tại.

### Business meaning

> Khách dành phần lớn income cho debt service có ít financial buffer hơn.

### Limitation

> Đây là association, không chứng minh quan hệ nhân quả hoặc universal cutoff.

\---

# 6\. Final Checklist trước khi giữ feature

Với mỗi feature, hỏi:

* \[ ] Có ý nghĩa nghiệp vụ rõ?
* \[ ] Input tồn tại trước prediction date?
* \[ ] Unit đã đồng nhất?
* \[ ] Denominator có thể bằng 0?
* \[ ] Missing nghĩa là gì?
* \[ ] Có double-count không?
* \[ ] Có trùng với feature khác không?
* \[ ] Distribution có hợp lý?
* \[ ] Có sample đủ lớn?
* \[ ] Validation có cải thiện hoặc interpretation có giá trị?
* \[ ] Không dùng target để tạo feature?

\---

# Câu nhớ cuối

> \*\*Feature Engineering = biến raw data thành một đại lượng gần với logic nghiệp vụ hơn, không phải tạo ra thông tin mà dataset không có.\*\*

```text
Income + Debt Payment
→ DTI

Income + Total Payments
→ FOIR

Current + Savings
→ CASA

Balance + Limit
→ Utilization

Loan + Collateral
→ LTV

Payment History
→ Late Rate / DPD / Recency / Trend

Transactions
→ Cashflow / Liquidity features
```

