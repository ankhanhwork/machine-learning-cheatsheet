# Hướng dẫn notebook Machine Learning Pipeline for Binary Target

Tài liệu này mô tả notebook [Machine Learning Pipeline for Binary Target](Machine%20Learning%20Pipeline%20for%20Binary%20Target.ipynb): notebook gồm những phần nào, mỗi phần dùng để làm gì và nên chạy theo thứ tự nào. Hướng dẫn lựa chọn cách chia dữ liệu theo thời gian/nhóm, kèm các tình huống và dataset mẫu, nằm riêng tại [split-data-guideline.md](split-data-guideline.md).

## Mục đích của notebook

Notebook là khung làm việc cho bài toán phân loại nhị phân. Luồng tổng quát là đọc dữ liệu → kiểm tra và chuẩn bị cột → chia dữ liệu để đánh giá → tiền xử lý → huấn luyện và so sánh mô hình → chọn mô hình/ngưỡng → diễn giải → tạo dự đoán cho dữ liệu chưa có nhãn.

Notebook có 168 ô: 69 ô Markdown giải thích bằng tiếng Anh và 99 ô code. Một số ô code là lựa chọn thay thế, ví dụ hoặc đòi hỏi tên cột/package cụ thể; không nên chạy toàn bộ notebook một cách máy móc.

## Nội dung theo phần

| Phần | Nội dung và mục đích |
|---|---|
| 1–3. Import, Load Data, Quick Data Check | Nạp thư viện và CSV; xem kích thước, kiểu dữ liệu, giá trị thiếu và một số thống kê cơ bản. |
| 4. Target Definition | Xác định ý nghĩa nhãn 0/1. Ô tạo nhãn từ số tháng quá hạn chỉ dùng nếu dữ liệu chưa có target. |
| 5–6. Select Target and Features; Select Numerical and Categorical Columns | Chọn cột nhãn, ID, cột cần loại và phân loại biến đầu vào theo kiểu dữ liệu. |
| 7–10. Data Quality, Outlier, PSI, Leakage | Kiểm tra dữ liệu thiếu, cardinality, cột hằng, vô cực, mất cân bằng, giá trị ngoại lệ, dịch chuyển train/test và nguy cơ rò rỉ. Đây là các chẩn đoán; không tự động quyết định xóa dữ liệu. |
| 11. Optional Feature Engineering | Ví dụ tạo biến ngày, log, tỷ lệ, tương tác và nhóm. Chỉ dùng sau khi thay tên cột và xác nhận ý nghĩa phù hợp. |
| 12. Train and Validation Split | Mặc định là chia ngẫu nhiên phân tầng. Có các lựa chọn bổ sung TimeSeriesSplit, GroupKFold và kết hợp thời gian/nhóm; hướng dẫn chọn nằm trong tài liệu riêng. |
| 13–16. Missing Values, Encoding, Scaling, Preprocessing Pipelines | Tham khảo cách điền khuyết, mã hóa biến phân loại, chuẩn hóa số và ghép bước vào pipeline. Chọn pipeline phù hợp với mô hình thay vì áp dụng mọi phương án liên tiếp. |
| 17–26. Models | Ghi kết quả đánh giá và thử Dummy baseline, Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting và XGBoost tùy điều kiện. |
| 27–29. Compare, Detailed Result, Threshold Tuning | So sánh các mô hình, xem metric/lỗi của mô hình được chọn và đánh giá tác động của các ngưỡng phân loại. |
| 30–32. Tuning and Cross-Validation | Tinh chỉnh Random Forest/XGBoost và đánh giá qua nhiều fold. Tuning tốn thời gian; loại fold phải khớp chiến lược chia đã chọn. |
| 33–35. Interpretation | Xem feature importance, hệ số Logistic Regression và permutation importance. |
| 36–40. Final Model and Submission | Khai báo mô hình/ngưỡng cuối, fit lại trên dữ liệu có nhãn, dự đoán dữ liệu chưa có nhãn và tạo tệp submission. |
| 41. SHAP Analysis | Các ví dụ SHAP cho XGBoost và Random Forest. Có thể dùng biểu đồ global để mô tả hành vi mô hình; SHAP không chứng minh quan hệ nhân quả. |
| 42. Report Outline | Danh sách gợi ý nội dung báo cáo: quy trình, validation, kết quả, trade-off, giới hạn và diễn giải. |

## Cách nạp dữ liệu

Mặc định, phần 2 đọc `train.csv`, `test.csv` và `submission_template.csv`. `train.csv` cần target, `test.csv` dùng để tạo dự đoán cuối. Nếu labelled rows nằm ở cả `train.csv` và `test.csv`, còn file dự đoán là `data_without_label.csv`, thay các dòng đọc mặc định bằng đoạn `pd.concat` đã comment sẵn trong code. Hai file labelled phải có cùng cột và đều có target. Tập chưa có nhãn không được dùng làm validation.

## Cấu hình cần làm trước khi chạy

Ở các cell cấu hình, thay các giá trị ví dụ cho phù hợp:

- `target_col`: tên cột nhãn; các metric mặc định xem `1` là positive class.
- `id_cols`: mã định danh không đưa vào mô hình nhưng có thể cần giữ lại cho tệp nộp.
- `drop_cols`: các cột không sử dụng được hoặc có nguy cơ rò rỉ thông tin.
- Chọn một cách chia tập kiểm định phù hợp với mục tiêu sử dụng. Nếu dùng TimeSeriesSplit, GroupKFold hoặc cách kết hợp, chỉ kích hoạt lời gọi tương ứng rồi chạy lại các ô mô hình phía sau.
- Ví dụ tạo biến dùng tên cột giả như `application_date`, `income`, `numerator_col`. Kiểm tra và thay tên cột, miền giá trị trước khi chạy.
- Nếu thêm biến sau khi tạo `X`, hãy tạo lại `X`, `X_test`, `numerical_cols`, `categorical_cols` rồi chạy lại các bước phía sau.

## Санал болгох ажиллуулах дараалал

1. Import, Load Data, Quick Data Check-ийг ажиллуулж, файлууд болон баганын бүтцийг шалга.
2. Target, ID болон хасах багануудыг тохируул; шаардлагатай бол шошготой файлуудыг нэгтгэ.
3. Leakage, өгөгдлийн чанар болон классын харьцааг хяна. Шалтгаангүйгээр сэжигтэй мөр/багана устгаж болохгүй.
4. Асуултад тохирох нэг validation стратеги сонгож, split helper болон дуудлагыг идэвхжүүл.
5. Preprocessing pipeline сонгож, эхлээд Dummy baseline, дараа нь боломжийн хэдэн загвар ажиллуул.
6. Ижил validation дээрх үр дүнг харьцуул; шаардлагатай бол tuning болон threshold сонголтыг хий.
7. `selected_model`, `selected_prob`, `selected_pred`, `final_model`, `final_threshold` утгуудыг бодит сонголттой тааруул.
8. Эцсийн загварыг бүх labelled өгөгдөл дээр fit хийж, шошгогүй файлд таамаглал гарга.
9. Submission-ийн мөрийн тоо, ID/дараалал, баганын нэр болон хоосон таамаглал байгаа эсэхийг шалга.

## Анхаарах гол зүйл

- Түүврийн class imbalance их үед accuracy дангаараа хангалтгүй. Positive class-ийн precision, recall, F1 болон Average Precision-ийг зорилгодоо нийцүүлэн тайлбарла.
- Random split сонгосон ч preprocessing pipeline-ийг зөвхөн training data дээр fit хийнэ.
- Хугацааны split сонговол tuning доторх `cv=3`-ыг тохирох хугацааны fold-оор солино. Group эсвэл time+group зорилгод мөн тохирсон group fold хэрэгтэй.
- Tuning, XGBoost, SHAP нь хугацаа эсвэл нэмэлт package шаардаж болно. Эхлээд үндсэн pipeline, baseline болон үнэлгээгээ гарга.
- SHAP болон feature importance нь загварын хамаарлыг тайлбарлана; шалтгаан-үр дагаврыг тогтоохгүй.
- Тайланд зөвхөн үнэхээр ажиллуулсан загвар, хэмжсэн metric, үүсгэсэн график болон бодитоор илэрсэн хязгаарлалтыг бич.

## Холбогдох файлууд

- [Split data guideline](split-data-guideline.md) — ямар нөхцөлд random, time, group эсвэл хосолсон split хэрэглэх, жишээ CSV болон gap/leakage-ийн тайлбар.
- [Notebook generator](../generator/notebook-generator.py) — notebook-ийн бүтцийг үүсгэж/шинэчилдэг Python файл.
