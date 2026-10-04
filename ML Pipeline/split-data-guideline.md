# Hướng dẫn chọn cách chia dữ liệu cho bài toán nợ xấu

Tài liệu này tập trung vào cách chọn chiến lược chia dữ liệu cho notebook [Machine Learning Pipeline for Binary Target](Machine%20Learning%20Pipeline%20for%20Binary%20Target.ipynb), các tình huống minh họa và những điểm cần kiểm tra để tránh đánh giá sai. Mô tả cấu trúc toàn bộ notebook nằm trong [README.md](README.md).

## 1. Bối cảnh dữ liệu

Tài liệu này giả định nhóm có dữ liệu có nhãn để huấn luyện và kiểm định nội bộ, cùng một tệp riêng chưa có nhãn để dự đoán cuối. Tên cột trong các ví dụ cần được thay bằng tên thật; feature phải là thông tin đã biết tại ngày dự đoán.

## 2. Khi nào cần gộp các tệp dữ liệu?

Theo mặc định, `train.csv` chứa các dòng đã có nhãn, còn `test.csv` chứa các dòng chưa có nhãn cần dự đoán. Nếu các dòng có nhãn được chia giữa `train.csv` và `test.csv`, trong khi tệp dự đoán riêng là `data_without_label.csv`, hãy dùng đoạn `pd.concat` đang được ghi chú trong phần nạp dữ liệu của notebook để gộp hai tệp có nhãn thành `train`, rồi đọc tệp chưa có nhãn vào `test`. Hai tệp được gộp cần có cùng cấu trúc cột và đều phải có cột nhãn.

Sau khi gộp, hãy chia tập kiểm định từ dữ liệu có nhãn theo chiến lược phù hợp. Không dùng `data_without_label.csv` để kiểm định; tệp này dành cho bước dự đoán cuối cùng.

## 3. Nguyên tắc chọn cách chia dữ liệu

Cách chia tập cần mô phỏng đúng cách mô hình sẽ được sử dụng sau này. Cần xác định nhóm muốn ước lượng hiệu quả trên các kỳ tương lai của danh mục khách hàng hiện hữu hay trên những khách hàng mới mà mô hình chưa từng thấy.

| Mục tiêu sử dụng thực tế | Lựa chọn phù hợp ban đầu | Lý do |
|---|---|---|
| Mỗi khách hàng hoặc khoản vay chỉ có một dòng; không có yếu tố thời gian cần mô phỏng | `train_test_split(..., stratify=y)` | Chia ngẫu nhiên phù hợp; `stratify` giữ gần đúng tỷ lệ các lớp nhãn. |
| Đánh giá các kỳ thanh toán hoặc tháng tương lai của khách hàng hiện hữu | Chia theo thời gian / `TimeSeriesSplit` | Tập kiểm định có ngày muộn hơn tập huấn luyện, mô phỏng việc dự đoán tương lai. |
| Có nhiều dòng cho mỗi khách hàng, mục tiêu là áp dụng cho khách hàng mới | `GroupKFold` hoặc giữ riêng một nhóm khách hàng | Tất cả dòng của một khách hàng được giữ trong cùng một phần. |
| Đánh giá khách hàng mới xuất hiện trong kỳ tương lai | Kết hợp chia theo thời gian và nhóm khách hàng | Vừa giữ đúng thứ tự thời gian, vừa đảm bảo khách hàng kiểm định không xuất hiện trong tập huấn luyện. |

Câu hỏi quan trọng là: *Khi mô hình được dùng thực tế, lịch sử của cùng khách hàng/khoản vay/tài khoản có sẵn trong dữ liệu huấn luyện hay không?* Nếu có, chia theo thời gian thường là điều kiện chính. Nếu cần chấm điểm khách hàng hoàn toàn mới, cần tách theo nhóm khách hàng. Nếu cả hai điều kiện đều quan trọng, cân nhắc cách chia kết hợp.

## 4. Các ví dụ dữ liệu mẫu

Trong các ví dụ, `bad_debt_next_3m` cho biết có phát sinh nợ xấu trong vòng ba tháng sau thời điểm quan sát hay không (`1` = có, `0` = không). Cột này chỉ có trong dữ liệu huấn luyện. Các biến đầu vào phải là thông tin đã biết tại ngày dự đoán.

### A. Nhiều lần quan sát khoản thanh toán của cùng khách hàng — chia theo thời gian

Nếu một khách hàng xuất hiện ở nhiều tháng và mục tiêu là dự đoán rủi ro của kỳ sau, hãy đặt các dòng cùng kỳ vào một phần và dùng khoảng thời gian mới nhất làm tập kiểm định. Nếu khi triển khai có thể sử dụng lịch sử của khách hàng hiện hữu, khách hàng đó có thể xuất hiện trong cả giai đoạn huấn luyện và giai đoạn tương lai cần đánh giá.

```csv
customer_id,payment_date,days_past_due,utilization,bad_debt_next_3m
C01,2025-01-31,0,0.35,0
C02,2025-01-31,12,0.71,1
C01,2025-02-28,8,0.48,0
C03,2025-02-28,0,0.22,0
C02,2025-03-31,35,0.83,1
C03,2025-03-31,0,0.29,0
```

Chia theo `payment_date` là phù hợp nếu mục tiêu là dự đoán kỳ sau cho danh mục có thể bao gồm khách hàng cũ. Việc có nên dùng `customer_id` làm biến đầu vào hay không là quyết định riêng; thông thường không đưa mã định danh thuần túy trực tiếp vào mô hình.

### B. Ảnh chụp danh mục theo tháng — mọi khách hàng cùng ngày chụp

Nếu tệp chứa ảnh chụp danh mục theo từng tháng, hãy giữ mọi dòng của cùng tháng ở chung một phần. Dù ngày tháng bị lặp, các dòng cùng kỳ không nên bị tách giữa huấn luyện và kiểm định.

```csv
customer_id,snapshot_month,overdue_days,balance,income_band,bad_debt_next_3m
C01,2025-01-31,0,1200,medium,0
C02,2025-01-31,18,4800,low,1
C01,2025-02-28,5,1150,medium,0
C02,2025-02-28,31,4600,low,1
C03,2025-02-28,0,700,high,0
C01,2025-03-31,24,1080,medium,1
```

Hãy chuẩn hóa ngày thành các kỳ có độ dài đều, chẳng hạn năm-tháng, rồi chia theo thời gian. Cách này đánh giá hiệu quả trên các ảnh chụp tương lai. Cần xác nhận việc cùng khách hàng xuất hiện ở nhiều kỳ có phản ánh đúng tình huống áp dụng thực tế hay không.

### C. Hồ sơ đăng ký hoặc thời điểm giải ngân — chia theo thời gian

Nếu cần chấm điểm hồ sơ tại ngày đăng ký hoặc ngày giải ngân, dùng `application_date` hoặc `origination_date` để giữ các nhóm hồ sơ mới hơn làm kiểm định. Các biến chỉ biết được sau khi bắt đầu trả nợ, như số ngày quá hạn hoặc hành động thu hồi nợ, có thể gây rò rỉ nhãn nếu dùng để dự đoán ngay tại thời điểm đăng ký.

```csv
loan_id,application_date,requested_amount,debt_to_income,employment_years,bad_debt_next_3m
L101,2025-01-06,5000,0.28,4,0
L102,2025-01-19,12000,0.61,1,1
L103,2025-02-03,7000,0.34,6,0
L104,2025-02-22,15000,0.67,2,1
L105,2025-03-08,8500,0.42,3,0
L106,2025-03-27,11000,0.55,1,1
```

### D. Bảng chéo tại một thời điểm, mỗi khách hàng một dòng — chia ngẫu nhiên phân tầng

Nếu mỗi khách hàng chỉ xuất hiện một lần, không cần mô phỏng thời gian và không có nhóm lặp lại, chia ngẫu nhiên có phân tầng là lựa chọn cơ sở phù hợp.

```csv
customer_id,age,income,loan_amount,late_payment_count,bad_debt
C01,29,42000,8000,0,0
C02,51,31000,12000,3,1
C03,38,68000,15000,0,0
C04,45,39000,9000,2,1
C05,33,52000,7000,0,0
```

### E. Nhiều dòng mỗi khách hàng, nhưng mục tiêu là chấm điểm khách hàng mới — GroupKFold

Nếu cần áp dụng cho khách hàng chưa từng xuất hiện, chia ngẫu nhiên theo dòng có thể khiến một số tháng của cùng khách hàng nằm ở cả huấn luyện lẫn kiểm định. `GroupKFold` giữ mọi quan sát của một `customer_id` trong cùng một fold. Bản thân cách này không đảm bảo thứ tự thời gian.

```csv
customer_id,observation_month,days_past_due,balance,bad_debt_next_3m
C01,2025-01,0,1200,0
C01,2025-02,4,1180,0
C02,2025-01,20,4500,1
C02,2025-02,36,4300,1
C03,2025-01,0,800,0
C03,2025-02,0,760,0
```

### F. Khách hàng mới trong kỳ tương lai — kết hợp thời gian và nhóm

Nếu khi triển khai khách hàng mới sẽ xuất hiện trong tương lai và mô hình không được thấy họ khi huấn luyện, hãy chọn một cửa sổ thời gian tương lai để kiểm định, sau đó giữ riêng một số khách hàng trong cửa sổ đó. Tất cả các dòng lịch sử của những khách hàng này cũng phải bị loại khỏi tập huấn luyện. Đây là cách chia nghiêm ngặt hơn và có thể bỏ lại nhiều dòng dữ liệu; chỉ dùng khi phản ánh đúng tình huống triển khai.

```csv
customer_id,snapshot_month,days_past_due,balance,bad_debt_next_3m
C01,2025-01,0,1200,0
C02,2025-01,20,4500,1
C03,2025-02,0,800,0
C01,2025-03,8,1150,0
C02,2025-03,42,4100,1
C04,2025-04,0,900,0
C05,2025-04,28,2600,1
C06,2025-04,0,1400,0
```

Tập kiểm định ở ví dụ này có thể gồm C04–C06 của tháng cuối, trong khi lịch sử của họ không được xuất hiện trong huấn luyện. Nếu mục tiêu là dự đoán kết quả tương lai cho khách hàng đã có trong tập huấn luyện như C01–C03, cách kết hợp này quá nghiêm ngặt; chỉ chia theo thời gian sẽ phù hợp hơn.

## 5. Những điểm cần kiểm tra khi chia theo thời gian

### Ngày dự đoán và cửa sổ thời gian của nhãn

Trước hết, xác định mỗi dòng đại diện cho điều gì: hồ sơ đăng ký, khoản vay mới, ảnh chụp hàng tháng hay một sự kiện thanh toán. Sau đó xác định `prediction_date` của dòng. Nhãn phải đo sự kiện xảy ra sau ngày dự đoán. Nếu cửa sổ tạo nhãn của tập huấn luyện chồng lên thời kỳ kiểm định, thông tin tương lai có thể bị rò rỉ.

Ví dụ, nếu nhãn đo vỡ nợ trong ba tháng tiếp theo, nhãn của các ảnh chụp cuối cùng trong huấn luyện có thể kéo dài sang lúc bắt đầu kiểm định. Cần xem xét đặt `gap` đủ để tránh chồng lấn. Trong helper của notebook, `gap` được tính theo số ngày/kỳ riêng biệt, không phải số ngày lịch. Với dữ liệu hàng tháng, `gap=3` bỏ qua ba kỳ tháng; với dữ liệu hàng ngày, điều đó không tương đương 90 ngày. Nếu các ngày không đều, trước tiên hãy gom dữ liệu theo kỳ phù hợp.

### Tránh dùng thông tin từ tương lai

- Mỗi biến đầu vào phải có sẵn tại ngày dự đoán.
- Không dùng các sự kiện xảy ra sau thời điểm dự đoán để tạo biến đầu vào.
- Các bước điền khuyết, chuẩn hóa, mã hóa và chọn biến cần nằm trong `Pipeline` và chỉ được khớp trên phần huấn luyện.
- Nếu cùng một khoản vay có nhiều dòng sự kiện, cách chia cần tính đến mối liên hệ giữa các dòng đó và phù hợp với mục tiêu đánh giá.

### Hạn chế của TimeSeriesSplit

- Chia tiến về phía trước: trong mỗi fold, dữ liệu huấn luyện có thời gian trước dữ liệu kiểm định.
- Không phân tầng theo lớp. Vì vỡ nợ thường hiếm, hãy kiểm tra số trường hợp dương tính trong từng fold.
- Các cửa sổ kiểm định có thể có cùng số mốc thời gian nhưng không nhất thiết có cùng số dòng dữ liệu.
- `gap` được tính theo chỉ số của ngày/kỳ duy nhất trong helper, không phải số ngày.
- Cùng một khách hàng có thể xuất hiện trong cả huấn luyện và kiểm định. Điều này thường phù hợp khi đánh giá tương lai của khách hàng hiện hữu.

### Hạn chế của chia theo nhóm

- `GroupKFold` tách khách hàng nhưng không ép buộc thứ tự thời gian.
- Nếu số khách hàng ít, cần giảm `n_splits`.
- Khi nhãn mất cân bằng mạnh, một số fold có thể chỉ có một lớp; các chỉ số như Average Precision hoặc recall khi đó có thể không ổn định hoặc không xác định.
- Kết hợp thời gian và nhóm có thể làm giảm mạnh số dòng huấn luyện và số trường hợp dương tính ở kiểm định.

## 6. Các lựa chọn được bổ sung trong notebook

Phần chia dữ liệu hiện có bốn lựa chọn:

1. **Chia ngẫu nhiên phân tầng** — mặc định, phù hợp khi mỗi khách hàng chỉ có một dòng và không có rủi ro rò rỉ theo thời gian.
2. **TimeSeriesSplit** — đánh giá các kỳ tương lai; helper giữ mọi dòng có cùng mốc thời gian trong cùng một fold.
3. **GroupKFold** — đánh giá khách hàng mới; toàn bộ dòng của một khách hàng thuộc cùng một fold.
4. **Kết hợp thời gian và nhóm** — chọn khách hàng kiểm định từ cửa sổ tương lai và loại lịch sử của họ khỏi tập huấn luyện.

Chia ngẫu nhiên vẫn là mặc định. Các lời gọi trong phần tùy chọn đang được ghi chú để không chạy tự động. Sau khi chọn đúng chiến lược, bỏ ghi chú lời gọi tương ứng để tạo `X_train`, `X_valid`, `y_train`, `y_valid`, rồi chạy lại các bước mô hình hóa phía sau.

Khi tinh chỉnh mô hình, không đưa cửa sổ kiểm định cuối vào các fold dùng để tinh chỉnh. Nếu dùng chia theo thời gian, hãy thay `cv=3` ngẫu nhiên trong `RandomizedSearchCV` bằng các fold tiến theo thời gian được tạo từ `X_train`. Nếu mục tiêu là khách hàng mới, hãy tạo fold GroupKFold chỉ từ mã khách hàng thuộc `X_train`. Với mục tiêu kết hợp thời gian và nhóm, GroupKFold thông thường chưa đủ; cần các fold riêng giữ đúng cả thứ tự thời gian lẫn nhóm trong phạm vi huấn luyện.

## 7. Trình tự sử dụng notebook

1. Kiểm tra tên, cấu trúc và giá trị cột của các CSV; gộp các tệp có nhãn nếu cần.
2. Điền đúng `target_col`, `id_cols`, `drop_cols`; xác nhận nhãn `1` là lớp dương trong các chỉ số mặc định.
3. Xác định ý nghĩa mỗi dòng, ngày dự đoán và mục tiêu là khách hàng cũ hay khách hàng mới.
4. Chọn một chiến lược kiểm định phù hợp trong bảng trên; dùng đúng một kết quả chia cho các bước mô hình hóa tiếp theo.
5. Kiểm tra rò rỉ, dữ liệu thiếu và mất cân bằng lớp. Nếu điều chỉnh feature engineering, hãy tạo lại `X`, `X_test` và danh sách cột.
6. Đặt bước điền khuyết, mã hóa và chuẩn hóa trong các pipeline tiền xử lý; không lấy thông tin từ tập kiểm định.
7. Chạy mô hình cơ sở Dummy trước, sau đó chọn một số mô hình phù hợp để so sánh bằng cùng tập kiểm định và chỉ số.
8. Sau khi chọn mô hình, xem confusion matrix, precision/recall của lớp dương, Average Precision và tác động của ngưỡng phân loại.
9. Nếu cần tinh chỉnh, dùng đúng kiểu fold và giữ tập kiểm định cuối chưa được đụng đến cho bước đánh giá.
10. Huấn luyện lại mô hình cuối trên toàn bộ dữ liệu có nhãn, dự đoán `data_without_label.csv`, rồi kiểm tra ID, số dòng và tên cột trong tệp nộp.

## 8. Các ô ví dụ và cấu hình cần chú ý

- Ô tạo nhãn từ số tháng quá hạn chỉ là ví dụ. Nếu dữ liệu đã có nhãn, không tạo lại nhãn.
- Các ô tạo biến mới dùng tên cột minh họa nên có thể báo `KeyError` nếu chạy nguyên trạng. Hãy sửa tên và kiểm tra miền giá trị trước.
- Nếu thêm biến sau khi tạo `X`, cần tạo lại `X`, `X_test`, `numerical_cols`, `categorical_cols` và chạy lại tiền xử lý.
- Cần đồng bộ `final_model`, `final_threshold`, `selected_model` với lựa chọn thật; giá trị minh họa không tự cập nhật.
- Các ô feature importance và SHAP yêu cầu đúng đối tượng mô hình. Nếu chưa chạy tuning, `rf_best` hoặc `xgb_best` có thể chưa tồn tại. Chỉ đưa biểu đồ SHAP toàn cục đã thực sự tạo vào báo cáo.
- XGBoost, SHAP và tuning có thể tốn thời gian; hãy chạy mô hình cơ sở trước.
- Sau khi tạo tệp nộp, kiểm tra dự đoán có bị thiếu không, mã hóa 0/1, ID/thứ tự và cấu trúc có khớp mẫu yêu cầu không.

## 9. Viết kết quả báo cáo dựa trên bằng chứng

Chỉ ghi những mô hình đã chạy thật, chiến lược chia, chỉ số, ngưỡng và confusion matrix đã thu được. Tập hợp kết quả mô hình vào một bảng; không tự điền số liệu chưa đo. Có thể thảo luận số lượng trường hợp vỡ nợ ít, lịch sử thời gian ngắn, mức độ phù hợp giữa cách chia và mục tiêu triển khai, chồng lấn cửa sổ nhãn, thiếu biến cần thiết hoặc giới hạn thời gian tính toán nếu những điểm này thực sự ảnh hưởng kết quả.

## Tài liệu tham khảo chính thức

- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
