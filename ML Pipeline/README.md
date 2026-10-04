# Hướng dẫn notebook Machine Learning Pipeline for Binary Target

Tài liệu này mô tả notebook [Machine Learning Pipeline for Binary Target](Machine%20Learning%20Pipeline%20for%20Binary%20Target.ipynb): notebook gồm những phần nào, mỗi phần dùng để làm gì và nên chạy theo thứ tự nào. Hướng dẫn lựa chọn cách chia dữ liệu theo thời gian/nhóm, kèm các tình huống và dataset mẫu, nằm riêng tại [split-data-guideline.md](split-data-guideline.md).

## Mục đích của notebook

Notebook là khung làm việc cho bài toán phân loại nhị phân. Luồng tổng quát là đọc dữ liệu → kiểm tra và chuẩn bị cột → chia dữ liệu để đánh giá → tiền xử lý → huấn luyện và so sánh mô hình → chọn mô hình/ngưỡng → diễn giải → tạo dự đoán cho dữ liệu chưa có nhãn.

Notebook có 168 ô: 69 ô Markdown giải thích bằng tiếng Anh và 99 ô code. Một số ô code là lựa chọn thay thế, ví dụ hoặc đòi hỏi tên cột/package cụ thể; không nên chạy toàn bộ notebook một cách máy móc.

## Nội dung theo phần

| Phần | Nội dung và mục đích |
|---|---|
| 1–3. Thư viện, nạp dữ liệu, kiểm tra nhanh | Nạp thư viện và tệp CSV; xem kích thước, kiểu dữ liệu, giá trị thiếu và một số thống kê cơ bản. |
| 4. Xác định nhãn mục tiêu | Xác định ý nghĩa nhãn 0/1. Chỉ dùng ô tạo nhãn từ số tháng quá hạn nếu dữ liệu chưa có nhãn mục tiêu. |
| 5–6. Chọn nhãn và biến đầu vào; phân loại kiểu biến | Chọn cột nhãn, mã định danh, cột cần loại và phân loại biến đầu vào theo kiểu dữ liệu. |
| 7–10. Chất lượng dữ liệu, giá trị ngoại lệ, PSI, rò rỉ | Kiểm tra dữ liệu thiếu, số lượng giá trị khác nhau, cột hằng, giá trị vô cực, mất cân bằng, giá trị ngoại lệ, dịch chuyển giữa train/test và nguy cơ rò rỉ. Đây là các bước chẩn đoán, không tự quyết định xóa dữ liệu. |
| 11. Tạo biến tùy chọn | Ví dụ tạo biến ngày, biến đổi log, tỷ lệ, tương tác và nhóm. Chỉ dùng sau khi thay tên cột và xác nhận ý nghĩa phù hợp. |
| 12. Chia tập huấn luyện và kiểm định | Mặc định là chia ngẫu nhiên có phân tầng. Có thêm TimeSeriesSplit, GroupKFold và cách kết hợp thời gian/nhóm; tài liệu riêng hướng dẫn cách chọn. |
| 13–16. Giá trị thiếu, mã hóa, chuẩn hóa, pipeline tiền xử lý | Tham khảo cách điền khuyết, mã hóa biến phân loại, chuẩn hóa biến số và ghép các bước vào pipeline. Chọn pipeline phù hợp với mô hình thay vì áp dụng mọi phương án liên tiếp. |
| 17–26. Các mô hình | Ghi kết quả đánh giá và thử Dummy làm mốc so sánh, Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting và XGBoost nếu phù hợp. |
| 27–29. So sánh, xem kết quả chi tiết, điều chỉnh ngưỡng | So sánh các mô hình, xem chỉ số và lỗi của mô hình được chọn, đánh giá tác động của các ngưỡng phân loại. |
| 30–32. Tinh chỉnh và kiểm định chéo | Tinh chỉnh Random Forest/XGBoost và đánh giá qua nhiều fold. Việc tinh chỉnh tốn thời gian; cách tạo fold phải phù hợp với chiến lược chia đã chọn. |
| 33–35. Diễn giải mô hình | Xem mức độ quan trọng của biến, hệ số Logistic Regression và mức độ quan trọng qua phép hoán vị. |
| 36–40. Mô hình cuối và tệp nộp | Khai báo mô hình/ngưỡng cuối, huấn luyện lại trên dữ liệu có nhãn, dự đoán dữ liệu chưa có nhãn và tạo tệp kết quả để nộp. |
| 41. Phân tích SHAP | Ví dụ SHAP cho XGBoost và Random Forest. Có thể dùng biểu đồ tổng quát để mô tả cách mô hình hoạt động; SHAP không chứng minh quan hệ nhân quả. |
| 42. Dàn ý báo cáo | Gợi ý nội dung báo cáo: quy trình, cách kiểm định, kết quả, đánh đổi, giới hạn và diễn giải. |

## Cách nạp dữ liệu

Mặc định, phần 2 đọc `train.csv`, `test.csv` và `submission_template.csv`. `train.csv` cần có cột nhãn; `test.csv` dùng để tạo dự đoán cuối. Nếu các dòng có nhãn nằm trong cả `train.csv` và `test.csv`, còn tệp dự đoán là `data_without_label.csv`, hãy thay các dòng đọc mặc định bằng đoạn `pd.concat` đã ghi chú trong mã. Hai tệp có nhãn phải có cùng cột và đều phải có cột nhãn. Không dùng dữ liệu chưa có nhãn làm tập kiểm định.

## Cấu hình cần làm trước khi chạy

Trong các ô cấu hình, thay các giá trị minh họa cho phù hợp:

- `target_col`: tên cột nhãn; các chỉ số mặc định xem `1` là lớp dương.
- `id_cols`: mã định danh không đưa vào mô hình nhưng có thể cần giữ lại cho tệp nộp.
- `drop_cols`: các cột không sử dụng được hoặc có nguy cơ rò rỉ thông tin.
- Chọn một cách chia tập kiểm định phù hợp với mục tiêu sử dụng. Nếu dùng TimeSeriesSplit, GroupKFold hoặc cách kết hợp, chỉ kích hoạt lời gọi tương ứng rồi chạy lại các ô mô hình phía sau.
- Ví dụ tạo biến dùng tên cột giả như `application_date`, `income`, `numerator_col`. Kiểm tra và thay tên cột, miền giá trị trước khi chạy.
- Nếu thêm biến sau khi tạo `X`, hãy tạo lại `X`, `X_test`, `numerical_cols`, `categorical_cols` rồi chạy lại các bước phía sau.

## Trình tự chạy gợi ý

1. Nhập thư viện, nạp dữ liệu và chạy kiểm tra nhanh để xem cấu trúc tệp và các cột.
2. Chọn cột nhãn, cột ID và các cột cần loại; gộp các tệp có nhãn nếu cần.
3. Kiểm tra nguy cơ rò rỉ, chất lượng dữ liệu và tỷ lệ các lớp. Không xóa dòng hoặc cột đáng ngờ nếu chưa có căn cứ.
4. Chọn một cách kiểm định phù hợp với mục tiêu bài toán, rồi kích hoạt helper và lời gọi tương ứng.
5. Chọn pipeline tiền xử lý; chạy Dummy làm mốc trước, sau đó chạy một số mô hình phù hợp.
6. So sánh kết quả trên cùng tập kiểm định; nếu cần, tinh chỉnh mô hình và chọn ngưỡng.
7. Đồng bộ `selected_model`, `selected_prob`, `selected_pred`, `final_model`, `final_threshold` với lựa chọn thực tế.
8. Huấn luyện lại mô hình cuối trên toàn bộ dữ liệu có nhãn và dự đoán tệp chưa có nhãn.
9. Kiểm tra số dòng, ID/thứ tự, tên cột và các giá trị dự đoán bị thiếu trong tệp nộp.

## Những điểm cần lưu ý

- Khi các lớp mất cân bằng, chỉ số accuracy là chưa đủ. Hãy diễn giải precision, recall, F1 và Average Precision của lớp dương theo mục tiêu bài toán.
- Dù chọn cách chia ngẫu nhiên, pipeline tiền xử lý vẫn chỉ được khớp trên dữ liệu huấn luyện.
- Nếu chia theo thời gian, hãy thay `cv=3` trong phần tinh chỉnh bằng các fold giữ đúng thứ tự thời gian. Mục tiêu chia theo nhóm hoặc kết hợp thời gian và nhóm cũng cần fold tương ứng.
- Tinh chỉnh mô hình, XGBoost và SHAP có thể cần thêm thời gian hoặc thư viện. Trước hết hãy chạy pipeline chính, mô hình cơ sở và phần đánh giá.
- SHAP và mức độ quan trọng của biến mô tả cách mô hình hoạt động, không xác lập quan hệ nhân quả.
- Trong báo cáo, chỉ nêu mô hình đã chạy, chỉ số đã đo, biểu đồ đã tạo và giới hạn đã quan sát được.

## Các tệp liên quan

- [Hướng dẫn chia dữ liệu](split-data-guideline.md) — khi nào nên chia ngẫu nhiên, theo thời gian, theo nhóm hoặc kết hợp; kèm CSV mẫu và lưu ý về khoảng cách/rò rỉ.
- [Tệp tạo notebook](../generator/notebook-generator.py) — mã Python tạo và cập nhật cấu trúc notebook.
