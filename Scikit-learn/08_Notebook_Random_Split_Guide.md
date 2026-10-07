# 08 · Hướng dẫn notebook Random Split

Notebook gốc: [Machine Learning Pipeline for Binary Target - Random Split.ipynb](../ML%20Pipeline/Machine%20Learning%20Pipeline%20for%20Binary%20Target%20-%20Random%20Split.ipynb)

## Mục tiêu và khi nào dùng

Notebook xây dựng bộ phân loại nhị phân từ dữ liệu có nhãn, so sánh baseline, tìm tham số, chọn ngưỡng dự đoán, kiểm tra holdout và tạo dự đoán cho file chưa có nhãn. Cách chia ngẫu nhiên có phân tầng phù hợp khi các quan sát gần như độc lập và dữ liệu tương lai có cùng phân phối với dữ liệu hiện tại.

Không dùng random split nếu nhiều dòng của cùng một khách hàng/đơn vị có thể lọt vào cả train và test, hoặc mục tiêu là dự đoán một giai đoạn tương lai. Khi có ràng buộc thời gian, dùng hướng dẫn [Time Split](09_Notebook_Time_Split_Guide.md).

## Chuẩn bị và chạy

1. Mở notebook bằng JupyterLab, VS Code hoặc môi trường notebook tương thích. Chọn kernel Python của dự án.
2. Đặt `train.csv` và `test.csv` trong working directory của kernel. `train.csv` cần có target; file còn lại là dữ liệu chưa nhãn để dự đoán. Notebook phát hiện trường hợp hai tên file bị đảo nếu target chỉ có trong `test.csv`.
3. Sửa cell 4: `target_col`. Sửa cell 13: `id_cols`, `drop_cols`, và `TIME_COL` nếu cần. ID được giữ để xuất kết quả nhưng không dùng làm feature; `drop_cols` dùng cho biến rò rỉ hoặc không hợp lệ.
4. Nếu target phải được suy ra từ `overdue_months`, xem cell 11: bỏ dấu comment và xác nhận quy tắc nhãn trước khi chạy. Mặc định target phải là 0/1.
5. Chạy các cell theo thứ tự từ trên xuống. Có thể dùng **Restart Kernel and Run All** sau khi cấu hình. Cell Optuna (104) là lựa chọn thay thế, không chạy cùng lúc với ba search cell mặc định.
6. Bắt đầu thử nhanh với `SEARCH_ITERATIONS = 5` ở cell 96; chạy đầy đủ hơn bằng cách tăng giá trị. Các cell fit XGBoost và tìm kiếm song song có thể dùng nhiều thời gian/CPU.
7. Cuối notebook, chọn model/ensemble ở cell 135 rồi chạy cell 137 để xem tóm tắt. SHAP là tùy chọn; thiếu package thì phần giải thích SHAP được bỏ qua.

Cài dependency nếu kernel báo thiếu package: chạy `%pip install pandas numpy matplotlib scikit-learn xgboost` trong một cell riêng, hoặc cài trong terminal của đúng môi trường Python. Dòng nhắc cài package ở cell 2 chỉ là comment. Sau khi cài, khởi động lại kernel và chạy lại từ đầu. Muốn dùng SHAP thì cài riêng `%pip install shap`.

## Các thuật toán chính

### Chia train, validation, test

Cell 50 định nghĩa `make_random_train_valid_test`. Lần chia đầu lấy 70% làm train và 30% holdout; lần hai tách holdout thành validation và test theo tỷ lệ còn lại. `stratify=y` được dùng ở cả hai lần để bảo toàn tỷ lệ lớp 0/1. Mặc định kết quả là 70% / 15% / 15%; `random_state=42` giúp lặp lại cùng cách chia.

- **Train**: fit baseline và search tham số.
- **Validation**: chọn ngưỡng xác suất và chọn ứng viên cuối.
- **Test holdout**: so sánh cuối; không dùng để chọn tham số/ngưỡng.
- **Predict**: hàng chưa nhãn trong file còn lại; chỉ dùng sau khi đã chọn quy trình.

### Tiền xử lý và model

Cell 75 tạo pipeline cho model nhạy với scale: median imputation + `StandardScaler` cho số; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` cho categorical. Cell 77 bỏ scaling cho model cây. Cell 79 là lựa chọn ordinal encoding, cần thận trọng vì mã số có thể ngụ ý thứ tự không tồn tại.

Các baseline: Logistic Regression (mô hình tuyến tính, `class_weight="balanced"`), Decision Tree, Random Forest và XGBoost. Pipeline đảm bảo imputer/encoder/scaler được fit bên trong từng fold, tránh học thống kê từ validation/test.

### Tuning, ngưỡng và ensemble

Cell 96 tạo ba fold CV từ train. Cell 98/100/102 dùng `RandomizedSearchCV` để thử một số tổ hợp tham số của Random Forest, XGBoost và Logistic Regression. Scoring mặc định là F1 cho lớp dương 1, được cấu hình tại cell 81. `refit=True` fit lại estimator tốt nhất trên toàn bộ train.

Model thường trả xác suất; ngưỡng 0.5 không nhất thiết tối ưu F1. Các cell 108/110/112 quét các ngưỡng ứng viên trên validation riêng từng model, sau đó giữ ngưỡng đó cố định để báo cáo train/validation/test. Cell 115-120 thử ensemble bằng trung bình xác suất của RF + XGBoost hoặc Logistic + RF + XGBoost; ngưỡng ensemble được chọn riêng trên validation.

Cell 123 fit lại model trên train + validation và đánh giá trên cùng test holdout, giữ nguyên ngưỡng validation. Sau khi báo cáo, validation đã trở thành dữ liệu train và không còn là tập đánh giá độc lập.

## Bản đồ từng cell

Số trong ngoặc vuông là **index cell** của notebook, bắt đầu từ 0. Các cell Markdown là lời giải thích/tiêu đề, không cần bấm chạy; cell Code cần chạy theo thứ tự. Mục dưới đây mô tả các code cell và vai trò của Markdown đi kèm.

| Cell | Nội dung / kết quả cần xem |
|---|---|
| 0-2 | Tiêu đề, điều kiện đầu vào, import thư viện và đặt seed 42. Dòng `%pip` trong comment cell 2 chỉ nhắc package cần có, không tự cài. |
| 3-4 | Đọc hai CSV, xác định file có nhãn, gắn cờ nguồn `labeled`/`predict`, ghép để tạo feature đồng nhất. Kiểm tra số hàng và chế độ labelled/predict được in. |
| 5-8 | EDA ban đầu: `info`, dtype, missing %, cardinality, duplicate rows và thống kê mô tả. Dùng output để sửa lỗi dữ liệu, không phải để tự động xóa hàng. |
| 9-11 | Markdown định nghĩa nhãn bad debt. Cell 11 là ví dụ tạo target từ tháng quá hạn, mặc định đang comment; chỉ bật khi đúng định nghĩa bài toán. |
| 12-15 | Chọn `target_col`, ID, cột bỏ; xem phân phối target; tạo X/y sơ bộ để kiểm tra. Cell 13 là nơi sửa cấu hình cột. |
| 16-17 | Phân nhóm cột số và cột categorical bằng dtype. Xem danh sách và sửa dtype sai trong dữ liệu nguồn nếu cần. |
| 18-21 | Kiểm tra missing, số giá trị duy nhất, tỷ lệ cardinality, cột hằng và cột gần như ID. Các phát hiện là gợi ý để review, không tự xóa. |
| 22-23 | Tìm `+inf`/`-inf` trong cột số; xử lý nguồn sinh ra infinity trước khi fit. |
| 24-25 | Đếm và tính tỷ lệ từng lớp target để hiểu mất cân bằng. Đây là lý do cần metric như F1/balanced accuracy, không chỉ accuracy. |
| 26-27 | IQR flag outlier (nhỏ hơn Q1 - 1.5×IQR hoặc lớn hơn Q3 + 1.5×IQR). Outlier cần điều tra, không mặc định loại bỏ. |
| 28-31 | Tính PSI giữa file labelled và predict. Cell 29 dùng quantile bins cho số; 30 so phân bố categories và missing; 31 áp dụng, phân loại shift theo ngưỡng heuristic 0.10/0.25. |
| 32-35 | Rà soát leakage: tên cột khả nghi, tương quan số với target, feature nhị phân gần như copy target. Đây chỉ là detector sơ bộ; cần quyết định theo thời điểm dự đoán. |
| 36-46 | Feature engineering tùy chọn trên cả hai nguồn: ngày thành year/month/quarter/day-of-week (38), log1p (40), ratio (42), interaction (44), age bins (46). Thay các placeholder bằng cột thật; cell nào không áp dụng thì giữ nguyên/tắt. |
| 47 | Tách labelled/predict sau feature engineering; loại target, thời gian, ID, cột drop khỏi X; kiểm tra target không thiếu và chỉ nhận 0/1; căn chỉnh schema `X_predict` theo X. |
| 48-50 | Markdown giải thích stratified split. Cell 50 tạo X_predict và hàm chia ba tập ngẫu nhiên có kiểm tra tổng tỷ lệ bằng 1. |
| 51-53 | Giải thích và chạy split 70/15/15. Xem số hàng mỗi partition; nếu lớp quá ít, stratification có thể không chia được. |
| 54-62 | So sánh cách impute median, mean, mode và constant `Unknown`. Các cell này khởi tạo transformer minh họa; pipeline thực tế phía dưới quyết định transformer nào được dùng. |
| 63-67 | Minh họa One-Hot Encoding và Ordinal Encoding; category mới được xử lý bằng `handle_unknown`. Không fit encoder riêng trên train/test. |
| 68-72 | Minh họa StandardScaler và RobustScaler. Chọn theo model và phân phối; fit thực tế diễn ra trong pipeline. |
| 73-79 | Tạo `ColumnTransformer`/pipeline cho Logistic Regression (scale + one-hot), cây (impute + one-hot, không scale), hoặc ordinal. Cell 75, 77, 79 là phần cấu hình preprocessing. |
| 80-81 | Đặt F1 averaging mặc định `binary` và positive label 1; tạo helper metrics gồm F1, accuracy, precision, recall, ROC AUC và PR AUC. Đổi averaging chỉ theo yêu cầu bài toán. |
| 82-93 | Tạo helper đo baseline (83); fit Logistic (85), Decision Tree (87), RF (89), XGBoost (91) trên train; đánh giá các partition bằng threshold 0.50; cell 93 gom bảng kết quả. |
| 94-96 | Cấu hình randomized CV. Cell 96 tạo 3 fold `StratifiedKFold` có shuffle, chỉ từ `X_train/y_train`. `SEARCH_ITERATIONS` điều khiển số tổ hợp thử; `SEARCH_N_JOBS=-1` dùng CPU song song. |
| 97-98 | Search tham số Random Forest bằng F1 CV; output best score và best parameters. |
| 99-100 | Search XGBoost, bao gồm `scale_pos_weight` tính từ tỷ lệ lớp train; output tham số và F1 tốt nhất. |
| 101-102 | Search Logistic Regression với `C`, penalty và class weight; solver `liblinear` tương thích penalty L1/L2 trong cấu hình này. |
| 103-104 | Markdown hướng dẫn Optuna; cell 104 là code thay thế đang comment. Chỉ dùng khi chủ động bật cả block và đã cài Optuna. |
| 105-106 | Lấy tham số tốt nhất, dựng lại ba pipeline tuned và fit trên train; đồng thời tính xác suất cho train/validation/test holdout. |
| 107-112 | Tìm ngưỡng F1 riêng cho RF (108), XGBoost (110), Logistic (112) trên validation; cùng ngưỡng được báo cáo trên mọi split. Không chọn ngưỡng dựa vào test. |
| 113-119 | Tạo helper trung bình xác suất (115); chạy thử blend RF + XGBoost (117) và blend cả ba model (119). So sánh F1 validation và threshold riêng của từng blend. |
| 120-121 | Bảng so sánh tuned model và ensemble theo validation F1. Chọn ứng viên dựa trên validation và ràng buộc vận hành. |
| 122-123 | Fit lại từng tuned model trên train + validation, so kết quả với test holdout trước/sau refit; ngưỡng giữ nguyên. Cell 123 không dùng để tune lại. |
| 124-125 | Chọn tên ensemble để vẽ ROC và Precision-Recall trên test holdout. Đồ thị đánh giá thứ hạng xác suất, không chọn threshold mới. |
| 126-127 | Chọn RF hoặc XGBoost và xem `feature_importances_`; đây là chỉ báo, có thể lệch khi feature tương quan. |
| 128-129 | Permutation importance trên validation bằng F1: tráo từng cột rồi đo suy giảm điểm. Dùng để xếp hạng feature, không phải bằng chứng nhân quả. |
| 130-131 | SHAP tùy chọn trên mẫu validation, giải thích model cây đã chọn. Nếu chưa cài `shap`, cell thông báo và bỏ qua. |
| 132-133 | Refit model tuned trên toàn bộ hàng labelled để phục vụ inference; ghi lại ensemble và ngưỡng đã chọn từ validation. |
| 134-135 | Chọn `PREDICTION_MODEL_NAME`; tạo xác suất cho `X_predict`, trung bình xác suất nếu chọn ensemble, áp threshold đã chọn, ghép với ID và chuẩn bị file output. |
| 136-137 | Tóm tắt cấu hình chạy: F1 rule, positive label, tỷ lệ row, số feature, ensemble và threshold. Lưu output prediction theo tên file được notebook in ra. |

## Cách đọc kết quả

- Baseline cho biết model đơn giản đạt mức nào; so sánh cùng metric, đặc biệt F1 của lớp 1.
- `best_score_` là trung bình CV trên train, không phải điểm test.
- Validation chọn ngưỡng và ứng viên; test holdout là ước lượng độc lập hơn cho dữ liệu cùng phân phối.
- ROC AUC/PR AUC dùng xác suất, không phụ thuộc threshold; F1/precision/recall phụ thuộc threshold.
- Kết quả tốt bất thường cần kiểm tra leakage, entity overlap và việc feature có tồn tại tại thời điểm dự đoán.

## Giới hạn cần nhớ

Notebook tính một số kiểm tra mô tả và feature engineering xác định trên dữ liệu gộp labelled + predict trước khi chia. Các phép biến đổi này không học thống kê từ target, nhưng PSI giữa hai file dùng để chẩn đoán drift chứ không phải đánh giá model. Không dùng test labelled/holdout để quyết định feature hoặc thay đổi lặp đi lặp lại sau khi xem điểm.
