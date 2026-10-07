# 09 · Hướng dẫn notebook Time Series Split

Notebook gốc: [Machine Learning Pipeline for Binary Target - Time Series Split.ipynb](../ML%20Pipeline/Machine%20Learning%20Pipeline%20for%20Binary%20Target%20-%20Time%20Series%20Split.ipynb)

## Mục tiêu và khi nào dùng

Notebook huấn luyện bộ phân loại nhị phân khi dữ liệu có mốc thời gian và mục tiêu là dự đoán các quan sát ở giai đoạn sau. Train đứng trước validation, validation đứng trước test theo thời gian. Cách này mô phỏng tốt hơn triển khai tương lai và tránh dùng thông tin tương lai để dự đoán quá khứ.

Cần có cột thời gian hợp lệ trong dữ liệu labelled. Nếu các dòng cùng timestamp có liên quan, notebook giữ toàn bộ timestamp trong cùng partition. Nếu dữ liệu không có thứ tự thời gian có ý nghĩa, dùng [Random Split](08_Notebook_Random_Split_Guide.md).

## Chuẩn bị và chạy

1. Mở notebook gốc trong `ML Pipeline` bằng JupyterLab, VS Code hoặc môi trường notebook tương thích; chọn kernel Python của dự án.
2. Đặt `train.csv` và `test.csv` trong working directory. Một file có target, file kia là dữ liệu chưa nhãn. Cột target phải nhị phân 0/1.
3. Sửa cell 4 `target_col`; cell 13 `id_cols`, `drop_cols`, và đặc biệt `TIME_COL` thành cột thời gian thật. Notebook cần timestamp parse được cho mọi labelled row; giá trị rỗng hoặc sai định dạng sẽ làm split dừng với lỗi.
4. Kiểm tra cell feature engineering ngày (38) và thay/tắt các ví dụ placeholder như `income`, numerator/denominator, `col_a`/`col_b` nếu không phù hợp. Thời gian dùng để chia được loại khỏi feature ở cell 48.
5. Chạy từ trên xuống, không chạy riêng một cell giữa notebook sau khi kernel mới khởi động vì nhiều cell tạo helper/biến phụ thuộc. Có thể dùng **Restart Kernel and Run All**.
6. Cell 54 in khoảng ngày và số hàng của train/validation/test. Xác nhận các khoảng ngày đúng thứ tự, không trùng timestamp giữa các partition.
7. Thử nhanh với `SEARCH_ITERATIONS = 5` ở cell 97; tăng lên khi chạy chính thức. SHAP ở cell 132 là tùy chọn, cần cài riêng nếu muốn dùng.

Cài dependency khi cần bằng một cell riêng `%pip install pandas numpy matplotlib scikit-learn xgboost`, hoặc terminal trong đúng môi trường Python. Dòng nhắc cài package ở cell 2 chỉ là comment. Sau cài đặt, khởi động lại kernel và chạy lại từ đầu.

## Thuật toán chia theo thời gian

### Tạo train / validation / test

Cell 51 định nghĩa `make_time_splits` và `make_time_three_way_split`:

1. Parse `time_part` thành datetime; dừng nếu timestamp thiếu hoặc không hợp lệ.
2. Lấy danh sách timestamp duy nhất và sắp xếp tăng dần.
3. Chạy `TimeSeriesSplit(n_splits=4)` trên danh sách timestamp, không phải trực tiếp trên từng row. Như vậy mọi hàng của cùng một thời điểm cùng nằm trong train hoặc validation.
4. Chọn train indices và validation indices từ fold áp chót; lấy validation portion của fold cuối làm test. Các fold mở rộng theo hướng về tương lai.

Vì `TimeSeriesSplit` để phần đầu làm lịch sử train ở từng fold, hai fold cuối tạo ba giai đoạn liên tiếp: train sớm hơn, validation ở sau, test mới nhất. Tỷ lệ row không nhất thiết đúng 70/15/15; độ dài giai đoạn phụ thuộc số timestamp và khoảng cách giữa chúng.

### Forward cross-validation và gap

Cell 97 chạy lại `make_time_splits` chỉ trên `X_train`, với 3 folds, để tuning. Trong mỗi fold, training timestamps luôn đứng trước validation timestamps. `CV_GAP` và `gap` đếm số **timestamp buckets** bị bỏ quanh ranh giới, không phải số hàng. Tăng gap nếu nhãn có cửa sổ tương lai chồng lấn hoặc feature/lookahead có thể truyền thông tin qua ranh giới.

## Bản đồ từng cell

Số trong ngoặc vuông là **index cell** trong notebook, tính từ 0. Markdown giải thích mục đích; Code cell thực thi. Các cell khác với random split được nêu rõ; phần modeling còn lại tương tự nhưng CV và ý nghĩa đánh giá là theo thời gian.

| Cell | Nội dung / kết quả cần xem |
|---|---|
| 0-2 | Tiêu đề, giả định temporal split, yêu cầu file; import thư viện và seed. Dòng `%pip` trong comment cell 2 chỉ nhắc package cần có, không tự cài. |
| 3-4 | Đọc hai CSV, xác định file có target, gắn nhãn nguồn `labeled`/`predict`, ghép để tạo feature đồng nhất. |
| 5-8 | Kiểm tra cấu trúc, dtype, missing, unique, duplicate và thống kê. Dùng kết quả để sửa dữ liệu trước khi chạy model. |
| 9-11 | Giải thích nhãn; cell 11 là tùy chọn tạo target từ overdue months, mặc định comment. Xác nhận nhãn nghiệp vụ trước khi bật. |
| 12-15 | Cấu hình target/ID/drop/time; kiểm tra tỷ lệ target và tạo X/y sơ bộ. Cell 13 phải trỏ `TIME_COL` đến timestamp thật. |
| 16-17 | Tách cột số và categorical theo dtype; xem danh sách và xử lý cột parse sai. |
| 18-21 | Kiểm tra missingness, cardinality, feature constant/ID-like. Chẩn đoán, không tự động drop. |
| 22-23 | Tìm giá trị vô cực trong feature số; xử lý trước model fitting. |
| 24-25 | Xem phân phối lớp của target; đánh giá mức mất cân bằng. |
| 26-27 | IQR-based outlier flags; chỉ điều tra giá trị cực đoan, không xóa tự động. |
| 28-31 | PSI numeric/categorical giữa labelled và prediction, báo cáo drift heuristic. Không phải metric chất lượng phân loại. |
| 32-35 | Tìm cột có tên đáng ngờ, tương quan số-target cao và feature nhị phân có vẻ copy label. Xác minh khả dụng tại thời điểm dự đoán. |
| 36-46 | Ví dụ feature engineering: calendar (38), log1p (40), ratio (42), interaction (44), binning (46). Thay placeholder bằng công thức có ý nghĩa; tránh feature dùng thông tin tương lai. |
| 47-48 | Sau khi tạo feature, tách labelled và predict; parse `time_values`, loại target/time/ID/drop khỏi X và đồng bộ cột predict. Thời gian không được dùng trực tiếp như cột feature ở cấu hình mặc định. |
| 49-51 | Markdown giải thích chronological split. Cell 51 dựng X_predict, kiểm tra timestamp và định nghĩa split helper dựa trên timestamp duy nhất. |
| 52-54 | Chạy split với `n_splits=4`, `gap=0`; cell 54 in min/max ngày và số hàng. Kiểm tra train dates < validation dates < test dates. |
| 55-63 | Minh họa imputer median/mean/mode/Unknown. Đây là các transformer mẫu; pipeline ở cell 76-80 dùng cấu hình thật bên trong CV. |
| 64-68 | Minh họa OneHotEncoder và OrdinalEncoder, bao gồm unknown category. Không tự fit riêng trên test/predict. |
| 69-73 | Minh họa StandardScaler và RobustScaler; lựa chọn theo thuật toán, fit thực tế được quản lý bởi pipeline. |
| 74-80 | Pipeline số + categorical: scale và one-hot cho linear; no-scale + one-hot cho cây; ordinal là phương án có điều kiện. |
| 81-82 | Chọn F1 averaging và positive label. Mặc định binary F1 của nhãn 1; helper metric báo thêm ROC AUC/PR AUC/accuracy/precision/recall. |
| 83-94 | Tạo helper baseline; fit Logistic, Decision Tree, RF và XGBoost trên cửa sổ train sớm nhất; báo cáo baseline trên train/validation/test với cutoff 0.50. Cell 94 gom bảng. |
| 95-97 | Thiết lập forward CV. Cell 97 tạo folds từ riêng X_train và `time_values` tương ứng; cấu hình iterations, jobs, số folds và gap. |
| 98-99 | RandomizedSearchCV cho Random Forest với forward folds; báo best CV F1 và tham số. |
| 100-101 | Search XGBoost; `scale_pos_weight` chỉ tính từ target của train. Cấu hình tham số/đầu ra tương tự cell 100 trong notebook random nhưng dùng time-ordered folds. |
| 102-103 | Search Logistic Regression với solver `liblinear`; chọn regularization/penalty/class weight bằng cùng forward CV. |
| 104-105 | Tùy chọn Optuna đang comment; chỉ bật thay cho các cell search mặc định nếu đã cài và hiểu thời gian chạy. |
| 106-107 | Dựng ba model với best params và fit trên train; tính xác suất trên các partition để dùng cho chọn ngưỡng. |
| 108-113 | Tìm threshold tối đa F1 riêng cho RF, XGBoost, Logistic trên validation ở giai đoạn sau train; áp dụng threshold giữ nguyên lên test giai đoạn muộn hơn. |
| 114-120 | Trung bình xác suất để thử ensemble RF + XGBoost và ensemble thêm Logistic; threshold mỗi ensemble được tune trên validation riêng. |
| 121-122 | Xếp hạng ứng viên theo validation F1; chọn model theo validation và ràng buộc triển khai. Không chọn theo test. |
| 123-124 | Refit trên train + validation rồi so với test holdout mới nhất; test vẫn nằm ngoài mọi lần fit/search/ngưỡng. Validation sau refit không còn là tập đánh giá. |
| 125-126 | Chọn ensemble cuối và vẽ ROC/PR trên test holdout theo xác suất. Không dùng đồ thị test để quay lại tuning. |
| 127-128 | Chọn model cây và xem impurity feature importance. Đây là gợi ý, có thể thiên lệch bởi scale/cardinality/tương quan. |
| 129-130 | Permutation importance trên validation, đo mức giảm F1 khi tráo feature. Diễn giải như kiểm tra dự báo trên giai đoạn validation, không phải quan hệ nhân quả. |
| 131-132 | SHAP tùy chọn cho mẫu validation; xử lý lỗi nếu model/phiên bản SHAP không tương thích. |
| 133-134 | Refit model đã chọn trên mọi hàng labelled để dự đoán file unlabelled; ngưỡng vẫn là ngưỡng chọn bằng validation. |
| 135-136 | Chọn một model hoặc final ensemble, tính xác suất trên X_predict, áp threshold, gắn prediction/ID và tạo output. |
| 137-138 | Cell cuối hiển thị tổng quan F1, positive label, số hàng/feature và threshold đã khóa; lưu/kiểm tra file prediction được tạo. |

## Cách đọc kết quả

- Thứ tự ngày của ba partition quan trọng hơn phần trăm chia: test phải đại diện cho tương lai được mô phỏng.
- Điểm CV đo khả năng tổng quát hóa qua nhiều lần huấn luyện quá khứ sang khoảng thời gian kế tiếp; dao động giữa folds phản ánh mức bất ổn theo thời gian.
- Validation là giai đoạn gần hơn dùng chọn threshold/model; test là giai đoạn mới nhất, chỉ dùng báo cáo cuối.
- Tụt điểm ở test so với validation có thể là drift theo thời gian, thay đổi chính sách/dữ liệu hoặc overfit vào validation; kiểm tra PSI và metric theo từng khoảng.
- Nếu timestamp lặp nhiều, giữ nhóm timestamp cùng nhau tránh cùng một thời điểm lọt cả hai phía ranh giới.

## Những giả định cần kiểm tra

- Dự đoán tương lai từ feature có sẵn tại thời điểm dự đoán; không dùng trạng thái được cập nhật sau đó.
- `TIME_COL` có thể parse thành datetime và đúng múi giờ/quy ước ngày.
- `gap=0` chỉ phù hợp khi không có overlap/lookahead ở ranh giới. Với nhãn dựa trên một cửa sổ tương lai, cân nhắc gap đủ lớn theo số timestamp buckets.
- `TimeSeriesSplit` phân chia theo thứ tự timestamp, không bảo đảm các fold có cùng số hàng nếu số bản ghi mỗi thời điểm khác nhau.
