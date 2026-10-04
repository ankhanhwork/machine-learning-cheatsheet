# Hướng dẫn sử dụng notebook phân loại nhị phân

Tài liệu này mô tả notebook [Machine Learning Pipeline for Binary Target.ipynb](Machine%20Learning%20Pipeline%20for%20Binary%20Target.ipynb), cách các phần nối với nhau và những chỗ cần chỉnh trước khi chạy trên bộ dữ liệu cụ thể.

## Notebook dùng để làm gì?

Notebook là bộ khung cho một quy trình phân loại nhị phân. Nó đi từ đọc dữ liệu, kiểm tra chất lượng, chuẩn bị biến, chia validation, thử một số mô hình, so sánh kết quả, chọn mô hình và ngưỡng, rồi tạo dự đoán cho dữ liệu chưa có nhãn.

Notebook có 160 ô: 65 ô Markdown hướng dẫn và 95 ô code. Không phải ô code nào cũng cần chạy. Nhiều ô là lựa chọn thay thế hoặc ví dụ có tên cột giả định; chỉ chạy các phần phù hợp với dữ liệu thực tế.

## Các phần chính

| Phần | Mục đích |
|---|---|
| 1–3. Thư viện, nạp dữ liệu, kiểm tra nhanh | Đọc các CSV, xem kích thước, kiểu dữ liệu, giá trị thiếu và thống kê mô tả. |
| 4–6. Target, feature và kiểu cột | Chỉ định cột nhãn, ID, cột cần bỏ; tách dữ liệu đầu vào thành biến số và biến phân loại. |
| 7–10. Kiểm tra chất lượng và leakage | Xem cột hằng số/cardinality cao, vô cực, mất cân bằng, outlier, khác biệt train-test và dấu hiệu leakage. Đây là các kiểm tra để xem xét, không tự động làm sạch dữ liệu. |
| 11. Feature engineering | Ví dụ tùy chọn cho ngày tháng, log, tỷ lệ, tương tác và chia nhóm. |
| 12–16. Validation và preprocessing | Chia train/validation; xem cách điền khuyết, mã hóa, scale và ghép các bước thành pipeline. |
| 17–26. Mô hình cơ bản | Benchmark và các ứng viên gồm Dummy, Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting và XGBoost. |
| 27–29. So sánh, xem lỗi, chỉnh threshold | Sắp xếp kết quả, xem classification report, confusion matrix, Precision–Recall curve và tác động của ngưỡng. |
| 30–32. Tuning và cross-validation | Tìm siêu tham số cho Random Forest/XGBoost và đánh giá qua nhiều fold. Đây là các bước tốn thời gian hơn. |
| 33–35. Diễn giải mô hình | Xem feature importance, hệ số Logistic Regression và permutation importance. |
| 36–40. Mô hình cuối và submission | Chọn mô hình/ngưỡng, fit lại trên toàn bộ dữ liệu có nhãn, dự đoán dữ liệu không nhãn và ghi CSV. |
| 41–42. SHAP và dàn ý report | Các ví dụ SHAP cho Random Forest/XGBoost và danh sách nội dung báo cáo. |

## Cách chọn cách nạp dữ liệu

Mặc định ở phần **2. Load Data**, notebook đọc:

- `train.csv`: dữ liệu có nhãn.
- `test.csv`: dữ liệu chưa có nhãn cần dự đoán.
- `submission_template.csv`: mẫu submission, gồm ID/thứ tự và cột cần điền.

Nếu hai file `train.csv` và `test.csv` đều là các phần dữ liệu **có nhãn**, còn file dự đoán thật sự là `data_without_label.csv`, phần Load Data có sẵn một khối `pd.concat` được comment. Khi chuyển sang cấu trúc này:

1. Comment hoặc xóa hai dòng mặc định đọc `train.csv` và `test.csv`.
2. Bỏ comment khối `pd.concat` để gộp hai file có nhãn vào `train`.
3. Đọc `data_without_label.csv` vào biến `test`.
4. Giữ lại submission template nếu bài cung cấp file đó.

Hai file được gộp cần có cùng các cột và đều phải có target. Notebook sẽ chia validation nội bộ từ tập đã gộp; file `data_without_label.csv` chỉ dùng ở bước dự đoán cuối.

## Các giá trị cần kiểm tra trước khi chạy

### Target, ID và cột cần loại

Ở phần **5. Select Target and Features**, sửa cho đúng tên cột thực tế:

- `target_col`: cột nhãn. Code metric hiện giả định nhãn là `0` và `1`, trong đó `1` là lớp dương.
- `id_cols`: cột định danh cần giữ trong submission nhưng không đưa vào mô hình. Nếu không có ID, dùng danh sách rỗng.
- `drop_cols`: các cột khác cần bỏ vì không phù hợp để dự đoán, có leakage, hoặc không dùng được tại thời điểm dự đoán.

Kiểm tra `prediction_col` ở phần tạo submission khớp với tên cột target/mẫu submission. Không bỏ ID khỏi dữ liệu submission nếu file mẫu yêu cầu ID.

### Ô tạo nhãn overdue

Ô **Label Creation from Overdue Months** là ví dụ riêng cho trường hợp phải tự tạo target từ `overdue_months`. Ô này dùng cột và quy tắc cụ thể, không phải bước bắt buộc. Nếu dữ liệu đã có target, bỏ qua ô này. Nếu cần tự tạo nhãn, sửa tên cột và quy tắc đúng theo data dictionary, rồi đặt `target_col` trùng với cột vừa tạo.

### Các ô feature engineering

Các ví dụ hiện dùng tên cột mẫu như `application_date`, `income`, `numerator_col`, `denominator_col`, `col_a`, `col_b`, `age`. Chạy nguyên trạng có thể gây `KeyError` nếu dữ liệu không có các cột đó. Chỉ chạy ví dụ phù hợp sau khi sửa tên cột, kiểm tra miền giá trị và xử lý trường hợp chia cho 0 hoặc log giá trị không hợp lệ.

**Lưu ý về thứ tự:** notebook tạo `X`, `X_test` và danh sách cột số/phân loại trước các ô feature engineering. Nếu tạo thêm cột sau đó, phải tạo lại `X`, `X_test`, `numerical_cols`, `categorical_cols` và chạy lại các bước validation/preprocessing phía sau; nếu không, mô hình sẽ không nhận các feature mới.

## Trình tự chạy gợi ý

1. Đặt các CSV trong working directory của notebook và chọn đúng kiểu dữ liệu ở phần Load Data.
2. Chạy import, nạp dữ liệu và kiểm tra nhanh cấu trúc/cột.
3. Xác nhận target, ID, cột loại bỏ và quy tắc tạo target nếu cần.
4. Đọc các bảng kiểm tra chất lượng, leakage và mất cân bằng. Điều tra cột đáng ngờ trước khi tự xóa hoặc biến đổi.
5. Chỉ chạy feature engineering phù hợp; sau đó dựng lại `X` và `X_test` như lưu ý ở trên.
6. Chọn cách preprocessing thích hợp. Các ô imputation/encoding/scaling riêng lẻ chủ yếu để tham khảo; thường chọn một pipeline hoàn chỉnh cho từng mô hình.
7. Chia validation rồi chạy một baseline và một số mô hình có lý do để so sánh. Không nhất thiết chạy mọi lựa chọn.
8. Xem bảng kết quả, sau đó cập nhật `selected_model`, `selected_pred`, `selected_prob` cho cùng một ứng viên trước khi xem report, confusion matrix và PR curve.
9. Nếu chỉnh threshold, ghi nhận threshold dựa trên validation và đồng bộ giá trị đó với `final_threshold`.
10. Đặt `final_model` đúng mô hình đã chọn, fit trên toàn bộ dữ liệu có nhãn, dự đoán `X_test`, tạo submission và kiểm tra số dòng/ID/nhãn trước khi lưu.

## Những điểm cần chú ý nhất

- **Không chạy mọi ô theo thứ tự một cách mù quáng.** XGBoost, SHAP và các feature engineering mẫu có thể cần package hoặc cột dữ liệu không có sẵn.
- **Target phải thống nhất.** Các phép tính `predict_proba(... )[:, 1]` và metric giả định lớp dương có nhãn `1`; kiểm tra mã hóa target trước khi chạy.
- **Validation khác test cuối.** Dùng validation để chọn mô hình/ngưỡng. Nhãn của dữ liệu không nhãn không có trong notebook nên không thể tính hiệu năng test từ đó.
- **Mô hình và ngưỡng cuối phải khớp lựa chọn.** Mặc định hiện đặt `final_model = random_forest` và `final_threshold = 0.50`; đây chỉ là giá trị mẫu. Kết quả threshold tuning không tự cập nhật hai biến này.
- **Feature importance đang gắn với Random Forest mặc định.** Nếu chọn mô hình khác, cập nhật đúng object hoặc chỉ diễn giải phần tương ứng.
- **Các ô tuning tốn thời gian.** Randomized search cấu hình 20 lần và `cv=3`, nghĩa là khoảng 60 lượt fit cho mỗi search, chưa tính các mô hình khác. Chỉ chạy nếu còn thời gian và thực sự cần.
- **XGBoost và SHAP là phần tùy chọn.** Cần cài package trước và chỉ chạy SHAP cho mô hình đã fit tương ứng. Các ô SHAP hiện trỏ tới `rf_best`/`xgb_best`, là biến được tạo trong phần tuning; nếu chưa chạy tuning thì chúng chưa tồn tại. Có thể bỏ qua SHAP hoặc đổi sang object đã fit phù hợp.
- **Phiên bản thư viện có thể ảnh hưởng.** `OneHotEncoder(sparse_output=False)` cần phiên bản scikit-learn hỗ trợ tham số này. Ghi lại package versions và cách chạy trong tài liệu nộp theo yêu cầu môn học.
- **Kiểm tra submission cuối.** Cột dự đoán, số dòng, ID/thứ tự và nhãn 0/1 phải khớp mẫu nộp. Notebook ghi file `submission_final.csv`.

## Phần SHAP

Notebook có ví dụ biểu đồ SHAP global (bar/beeswarm) và local (waterfall). Với report ngắn, thường chỉ cần chọn **một biểu đồ global** nếu đã chạy và biểu đồ giúp giải thích mô hình. Có thể bỏ qua các ô waterfall local nếu không cần phân tích một cá thể. SHAP mô tả hành vi của mô hình, không chứng minh quan hệ nhân quả.

## Trước khi nộp

- Chạy lại các ô cần thiết trong một kernel mới theo đúng thứ tự phụ thuộc.
- Chỉ ghi vào report các mô hình, metrics và biểu đồ thật sự đã tạo.
- Đảm bảo mô hình/ngưỡng trong report khớp với mô hình tạo `submission_final.csv`.
- Kiểm tra CSV mở được, không thiếu dự đoán, đúng số dòng và đúng cấu trúc yêu cầu.
- Lưu code/notebook gọn, giữ output dễ đọc và ghi package versions cùng hướng dẫn chạy.
