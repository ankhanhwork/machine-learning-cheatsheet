# Hướng dẫn lý thuyết: Machine Learning Pipeline cho phân loại nhị phân

> Tài liệu này giải thích notebook `Machine Learning Pipeline for Binary Target.ipynb` theo thứ tự quy trình. Notebook là khung thực hành: một số ô là mặc định, một số là ví dụ tùy chọn cần đổi tên cột hoặc bỏ comment trước khi chạy. Các giá trị tham số nêu dưới đây là cấu hình được viết trong notebook, không phải “đáp án tốt nhất” cho mọi dữ liệu.

ankhanhnguyen2305@gmail.com - Ankhanh2305$ - HuggingFace

## Mục lục

1. [Bức tranh toàn cảnh](#1-bức-tranh-toàn-cảnh)
2. [Thư viện và tính tái lập](#2-thư-viện-và-tính-tái-lập)
3. [Nạp dữ liệu và khám phá ban đầu](#3-nạp-dữ-liệu-và-khám-phá-ban-đầu)
4. [Định nghĩa nhãn, X và y](#4-định-nghĩa-nhãn-x-và-y)
5. [Kiểm tra chất lượng, mất cân bằng, ngoại lệ và PSI](#5-kiểm-tra-chất-lượng-mất-cân-bằng-ngoại-lệ-và-psi)
6. [Rò rỉ dữ liệu](#6-rò-rỉ-dữ-liệu)
7. [Feature engineering tùy chọn](#7-feature-engineering-tùy-chọn)
8. [Chia tập và validation](#8-chia-tập-và-validation)
9. [Tiền xử lý và Pipeline](#9-tiền-xử-lý-và-pipeline)
10. [Các thuật toán phân loại](#10-các-thuật-toán-phân-loại)
11. [Chỉ số đánh giá và chọn ngưỡng](#11-chỉ-số-đánh-giá-và-chọn-ngưỡng)
12. [Tuning và cross-validation](#12-tuning-và-cross-validation)
13. [Giải thích mô hình](#13-giải-thích-mô-hình)
14. [Huấn luyện cuối, dự đoán, submission](#14-huấn-luyện-cuối-dự-đoán-submission)
15. [Khung viết report](#15-khung-viết-report)
16. [Lưu ý trước khi dùng notebook với dữ liệu thật](#16-lưu-ý-trước-khi-dùng-notebook-với-dữ-liệu-thật)

## 1. Bức tranh toàn cảnh

Notebook giải bài toán **phân loại nhị phân**: từ thông tin khách hàng/khoản vay, dự đoán một quan sát thuộc lớp 0 hay lớp 1. Ngữ cảnh ví dụ là nợ xấu: `1 = bad debt`, `0 = không bad debt`. Mục tiêu không chỉ là huấn luyện thuật toán, mà là xây dựng **pipeline** gồm tiền xử lý, mô hình, đánh giá, giải thích và xuất dự đoán.

```text
Dữ liệu có nhãn + dữ liệu cần chấm điểm
             ↓
Kiểm tra cấu trúc, chất lượng, nhãn, leakage, dịch chuyển phân phối
             ↓
Tạo/chọn đặc trưng và chia train-validation phù hợp
             ↓
Pipeline tiền xử lý → mô hình
             ↓
So sánh bằng AP, F1 và ma trận nhầm lẫn; chọn ngưỡng
             ↓
Tuning/cross-validation/giải thích
             ↓
Fit lại trên toàn bộ dữ liệu có nhãn → dự đoán test → submission/report
```

Pipeline áp dụng cùng phép biến đổi khi fit và predict; đồng thời học giá trị điền thiếu, scaling, danh mục mã hóa chỉ từ phần train của mỗi fold. Đây là cách hạn chế **data leakage** trong tiền xử lý.

## 2. Thư viện và tính tái lập

- `pandas`, `numpy`: bảng dữ liệu, phép tính số, missing/infinity.
- `matplotlib`: đồ thị PR và confusion matrix.
- `scikit-learn`: chia dữ liệu, tiền xử lý, mô hình, metric, tuning, permutation importance.
- `xgboost`, `shap`: tùy chọn; cần cài và có phiên bản tương thích.
- `RANDOM_STATE = 42`: cố định một số thao tác ngẫu nhiên để dễ tái lập trong cùng môi trường. Seed không làm dữ liệu đại diện hơn; khác phiên bản/thư viện/phần cứng vẫn có thể cho sai khác nhỏ.

Notebook import một số công cụ không dùng ở mọi nhánh (chẳng hạn `GridSearchCV`, `RobustScaler`, `accuracy_score`). Import không có nghĩa là công cụ đó đã được áp dụng.

## 3. Nạp dữ liệu và khám phá ban đầu

### Các tệp

Mặc định đọc `train.csv` (có nhãn), `test.csv` (cần dự đoán), `submission_template.csv` (khung nộp). Nhánh chú thích minh họa trường hợp nhãn nằm ở cả train và test, còn dữ liệu cần chấm điểm là `data_without_label.csv`; chỉ dùng nếu cả hai file ghép đều có target.

`shape` cho số dòng/cột; `head()` xem vài bản ghi. `info()` xem dtype và số giá trị không-null. Bảng kiểm tra tính dtype, số/tỷ lệ thiếu, số giá trị phân biệt. `describe(include="all").T` cho thống kê mô tả; `duplicated()` đếm dòng trùng hoàn toàn.

**Ý nghĩa:** phát hiện sớm kiểu sai (ngày bị đọc thành chuỗi), cột gần như rỗng, ID lẫn trong predictors, dòng trùng và phạm vi bất thường.

**Điểm mạnh:** nhanh, dễ hiểu, giúp đặt câu hỏi về chất lượng trước modeling.

**Giới hạn:** thống kê không xác định nguyên nhân hay ý nghĩa nghiệp vụ. Dòng trùng có thể hợp lệ, không tự động xóa. `nunique(dropna=False)` xem missing là một mức riêng.

## 4. Định nghĩa nhãn, X và y

Nhãn 1 được mô tả là nợ xấu, ví dụ định nghĩa “quá hạn hơn 3 tháng”. Ô tạo nhãn dùng `overdue_months > 3`; đúng 3 tháng vẫn thành 0. Chỉ chạy nếu dữ liệu **chưa có target** và có cột quá hạn đúng nghĩa. Nếu target đã có, không chạy vì có thể ghi đè hoặc tạo nhãn khác định nghĩa.

Các placeholder: `target_col = "target"`, `id_cols = ["ID"]`, `drop_cols = []`. Thay bằng tên thực tế. `X` là predictor sau khi bỏ target, ID và các cột chủ động loại; `y` là target. `X_test` bỏ ID/drop columns nhưng không có target. ID thường phục vụ ghép kết quả, không nên làm predictor nếu chỉ là mã định danh.

`value_counts()` và tỷ lệ target giúp hiểu lớp 1 hiếm đến mức nào. Kiểm tra target là nhị phân, mã hóa đúng và cùng ý nghĩa giữa các tệp. Định nghĩa nhãn là quyết định nghiệp vụ: thời điểm quan sát, cửa sổ dự báo, khoảng thời gian xác nhận nợ xấu.

## 5. Kiểm tra chất lượng, mất cân bằng, ngoại lệ và PSI

### Kiểu cột, missing, hằng số, cardinality

Notebook chia cột tự động: dtype số → `numerical_cols`, còn lại → `categorical_cols`. Tiện nhưng có thể sai: mã danh mục số bị xem như liên tục; ngày dạng chuỗi thành category. Cần rà soát theo nghiệp vụ.

- **Missingness:** tỷ lệ ô thiếu. Missing có thể ngẫu nhiên, do quy trình, hoặc tự nó mang ý nghĩa.
- **Constant feature:** `unique <= 1`; không phân biệt quan sát, thường ít ích. Xác nhận đó không phải cờ nghiệp vụ đặc biệt rồi mới bỏ.
- **High cardinality:** notebook gắn cờ `unique_ratio >= 0.80` (unique chia số dòng). Đây là heuristic; nhiều mức có thể là ID hoặc biến liên tục có ích.
- **Infinity:** `+inf`/`-inf` thường do chia 0/biến đổi. Imputer thông thường không đảm bảo xử lý; hãy chuyển có chủ đích thành missing hay sửa công thức.

### Mất cân bằng lớp

Nếu bad debt ít hơn nhiều, mô hình đoán toàn 0 có thể có accuracy cao. Notebook vì thế xem phân bố target và nhấn mạnh F1 lớp 1, Average Precision, class weight và threshold. Không có kỹ thuật cân bằng nào luôn đúng; metric phải phản ánh hậu quả của FN và FP.

### Ngoại lệ theo IQR

Cho từng biến số: `IQR = Q3-Q1`, cận dưới `Q1-1.5×IQR`, cận trên `Q3+1.5×IQR`; đếm số/tỷ lệ ngoài cận và tính skewness (độ lệch phân phối). Đây là cờ chẩn đoán, không chứng minh dữ liệu sai. Thu nhập/khoản vay có thể lệch phải tự nhiên; xóa/winsorize tùy tiện sẽ mất tín hiệu. IQR cũng kém ổn định với phân phối rất lệch hoặc nhiều giá trị trùng.

### Population Stability Index (PSI)

PSI so sánh tỷ lệ phân bố train và test trên các bin/mức:

`PSI = Σ (p_test - p_train) × ln(p_test / p_train)`.

Biến số: biên bin lấy từ quantile train; categorical: hợp các mức train/test. Tỷ lệ bị clip tối thiểu `1e-6` để log không gặp 0. Heuristic notebook: `<0.10` nhỏ, `0.10–0.25` vừa, `>0.25` đáng kể.

**Dùng khi:** kiểm tra train/test có thể đại diện cho cùng quần thể không, hoặc theo dõi drift.

**Mạnh:** tóm tắt thay đổi phân phối dễ báo cáo.

**Yếu:** ngưỡng phụ thuộc bối cảnh; bin/cỡ mẫu/missing/category mới ảnh hưởng; không kiểm tra phân phối target test, không chứng minh hiệu năng giảm hay shift có hại. Cách tính category có thể nhạy với cardinality cao.

## 6. Rò rỉ dữ liệu

**Leakage** là predictor chứa thông tin chỉ xuất hiện sau thời điểm cần dự đoán hoặc trực tiếp tiết lộ target. Ví dụ: quá hạn, thu hồi nợ, xóa nợ, trạng thái sau vỡ nợ. Điểm validation có thể cao nhưng không triển khai được.

Notebook sàng lọc qua: (1) từ khóa trong tên cột như `target`, `default`, `overdue`, `collection`, `writeoff`, `recovery`, `dpd`; (2) tương quan tuyệt đối của cột số với target; (3) tỷ lệ khớp target của các cột tối đa hai giá trị.

Đây chỉ là **cờ điều tra**. Tương quan cao không chứng minh leakage, thấp cũng không chứng minh an toàn. Cần hỏi feature có sẵn ở thời điểm score không, được tạo lúc nào, định nghĩa có dùng kết quả tương lai không. Với dữ liệu thời gian, random split có thể làm rò rỉ cấu trúc giữa các kỳ.

## 7. Feature engineering tùy chọn

Các ô là ví dụ với tên cột giả. Chỉ bật nếu cột có thật, phép biến đổi hợp lý; làm nhất quán cho train/test. Trong notebook `X` được tạo trước phần engineering. Sau khi thêm biến vào train/test phải tạo lại `X`, `y`, `X_test`, danh sách numeric/category và split; nếu không, biến mới không vào model.

### Thành phần ngày

`pd.to_datetime(..., errors="coerce")` biến giá trị lỗi thành missing; tạo năm, tháng, quý, thứ trong tuần. Hữu ích khi có mùa vụ. Có thể bỏ qua nếu ngày không có tín hiệu, không sẵn ở thời điểm dự báo, hoặc cách biểu diễn thành phần chu kỳ không thích hợp. Dữ liệu thời gian thường cần validation thời gian.

### Logarithm `log1p`

`log1p(x)=ln(1+x)` dùng cho biến không âm, lệch phải; giảm ảnh hưởng khoảng cách của giá trị rất lớn. Không dùng thẳng nếu `x < -1`; dữ liệu âm cần phép biến đổi phù hợp. Giữ cả bản gốc và bản log có thể tạo đa cộng tuyến trong hồi quy.

### Tỷ lệ

`numerator / denominator` tạo đại lượng tương đối như nợ/thu nhập. Ví dụ notebook chưa xử lý mẫu số 0/missing, có thể sinh infinity/NaN; cần quy tắc rõ. Không tạo ratio giữa đại lượng không có quan hệ đơn vị/nghiệp vụ.

### Tương tác

Tích `col_a * col_b` biểu diễn tác động kết hợp, đặc biệt hữu ích khi mô hình tuyến tính không tự học tương tác. Cây có thể học tương tác qua nhánh. Tạo quá nhiều làm tăng chiều, giảm khả năng giải thích và tăng overfit.

### Binning

Ví dụ chia tuổi tại `[0,25,40,60,∞]`. Binning làm mất chi tiết và phụ thuộc ngưỡng. Nhãn `<=25` cần khớp quy tắc biên của `pd.cut`; mặc định khoảng đầu có thể không nhận giá trị 0 nếu không dùng `include_lowest=True`. Kiểm tra missing, biên và giá trị ngoài miền. Dùng khi mức có ý nghĩa nghiệp vụ/quan hệ bậc; tránh chia tùy tiện.

## 8. Chia tập và validation

### Holdout ngẫu nhiên phân tầng (mặc định)

`train_test_split(test_size=0.20, stratify=y, random_state=42)`: dành 20% dữ liệu có nhãn để validation; `stratify` giữ gần nguyên tỷ lệ lớp; seed làm split tái lập. Dùng khi dòng gần độc lập và triển khai giống lấy mẫu ngẫu nhiên.

Không phù hợp khi cùng khách hàng có nhiều dòng, mục tiêu là dự báo tương lai, hoặc các nhóm phụ thuộc. Random split có thể đưa bản ghi cùng khách hàng/thời kỳ vào cả hai phía và làm điểm quá lạc quan.

### Chia theo thời gian

`make_time_splits` sắp timestamp duy nhất rồi dùng `TimeSeriesSplit`; dòng có cùng timestamp ở cùng fold. Ví dụ `n_splits=5`, `gap=0`. Validation sau thời gian train. Dùng để mô phỏng dự báo các kỳ tương lai. `gap` tạo khoảng cách train-validation, hữu ích khi nhãn hoàn tất trễ hoặc cửa sổ quan sát chồng lấn.

Điểm yếu: kỳ gần nhất có thể khác cơ cấu; folds không phân tầng; cần đủ mốc thời gian. Nếu khách hàng lặp qua kỳ, cách này đo hiệu năng tương lai trên danh mục khách hàng tiếp tục, không đo khả năng với khách hàng chưa từng thấy.

### Chia theo nhóm: GroupKFold

`make_group_splits`/`make_group_holdout` dùng `GroupKFold` để mọi dòng của một group (ví dụ khách hàng) nằm cùng fold; kiểm tra group không missing và có ít nhất `n_splits` nhóm. Dùng khi triển khai cần score khách hàng mới. Không giữ thứ tự thời gian và không stratify target; kiểm tra số positive mỗi fold.

### Kết hợp thời gian và nhóm

`make_time_group_holdout` chọn cửa sổ thời gian cuối, lấy mẫu nhóm khách hàng trong cửa sổ đó cho validation, rồi loại mọi dòng cũ của các nhóm ấy khỏi train. Mặc định helper: `n_splits=5`, `gap=0`, `group_fraction=0.25`, `random_state=42`. Đây trả lời câu hỏi khắt khe hơn: “kỳ tương lai, khách hàng chưa thấy”. Có thể loại nhiều dữ liệu và tạo validation nhỏ/mất cân bằng. Kiểm tra thời gian, số dòng, số nhóm và số positive. Chọn phương thức split theo câu hỏi triển khai; không chạy chồng các lựa chọn một cách vô thức.

## 9. Tiền xử lý và Pipeline

### Imputation

- `SimpleImputer(strategy="median")`: trung vị, ít bị ngoại lệ kéo lệch; không khôi phục giá trị thật và giảm biến thiên.
- `mean`: trung bình số học; phù hợp hơn với phân phối gần đối xứng, nhạy với ngoại lệ.
- `most_frequent`: mode; tiện cho category nhưng missing bị nhập vào mức phổ biến.
- `constant`, `fill_value="Unknown"`: biến missing thành mức riêng; cần tương thích dtype/category.

Notebook tạo các imputer riêng để minh họa. Pipeline chính thực sự dùng median cho số và most-frequent cho category.

### Encoding category

- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`: tạo cột nhị phân từng mức; mức mới ở test không gây lỗi và thành vector 0 cho nhóm đó. Không áp đặt thứ tự. Có thể tạo rất nhiều cột; dense matrix tốn RAM. `sparse_output` phụ thuộc phiên bản sklearn.
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`: mức thành số nguyên, giá trị mới thành -1. Gọn nhưng gán thứ tự tùy ý. Với category danh nghĩa, thứ tự giả gây vấn đề, nhất là mô hình tuyến tính và cây chia theo ngưỡng. Dùng khi mức thực sự có thứ bậc đã được khai báo đúng.

### Scaling

- `StandardScaler`: trừ mean, chia standard deviation, thường về mean 0/std 1. Hữu ích cho Logistic Regression/thuật toán dựa khoảng cách/gradient; nhạy ngoại lệ.
- `RobustScaler`: dùng median/IQR, bền hơn trước ngoại lệ; notebook chỉ khai báo, không dùng ở pipeline chính.
- Cây thường không cần scale vì split dựa thứ tự/điểm chia.

### Ba bộ tiền xử lý

1. `preprocess_standard`: numeric median + StandardScaler; categorical most-frequent + one-hot. Dùng cho Logistic Regression.
2. `preprocess_tree`: numeric median, không scale; categorical most-frequent + one-hot. Dùng cho Dummy, Decision Tree, Random Forest, Extra Trees, XGBoost.
3. `preprocess_ordinal`: numeric median; categorical most-frequent + OrdinalEncoder. Dùng cho HistGradientBoosting.

`ColumnTransformer` chuyển mỗi nhóm cột qua transformer phù hợp rồi ghép kết quả. `Pipeline` ghép preprocessing và estimator, giúp mỗi fold chỉ học biến đổi từ tập train của fold.

## 10. Các thuật toán phân loại

Tham số sau đây gồm cấu hình cụ thể trong notebook và tham số quan trọng có mặt trong không gian tuning. Tham số không nêu vẫn dùng mặc định thư viện; mặc định có thể đổi theo phiên bản.

### DummyClassifier — baseline

Notebook: `strategy="prior"`; dự đoán lớp phổ biến nhất theo train và trả xác suất dựa trên prior. Không học quan hệ giữa X và y.

**Mạnh:** mốc đơn giản để biết mô hình thực có thêm giá trị không. **Yếu:** không dùng tín hiệu cá thể. **Dùng:** nên có baseline. **Không dùng:** làm mô hình quyết định khi dự báo có giá trị; accuracy cao không đủ nếu lớp lệch.

### Logistic Regression

Mô hình tuyến tính dự đoán xác suất qua sigmoid của tổng có trọng số. Hệ số cho chiều liên hệ trên thang biến đổi; không phải nhân quả.

Notebook thường đặt `max_iter=2000`, `random_state=42`; bản cân bằng thêm `class_weight="balanced"`.

- `max_iter`: số vòng tối đa để solver hội tụ. Tăng nếu chưa hội tụ; nếu scale/regularization sai, chỉ tăng vòng không giải quyết căn nguyên.
- `class_weight="balanced"`: tăng trọng số lỗi lớp hiếm theo tần suất. Có thể tăng recall positive, giảm precision và làm xác suất kém hiệu chuẩn.
- `random_state`: seed ở solver/ngữ cảnh có ngẫu nhiên.
- `C` (quan trọng nhưng notebook không tune): nghịch đảo độ mạnh regularization; C nhỏ → regularization mạnh, giảm overfit nhưng có thể underfit.
- `penalty`, `solver`: loại regularization và cách tối ưu; các lựa chọn phải tương thích.

**Mạnh:** nhanh, baseline tốt, khá dễ giải thích, xác suất hữu ích; cần scaling. **Yếu:** quan hệ tuyến tính trong không gian biến đổi; cần feature engineering cho phi tuyến/tương tác; đa cộng tuyến làm hệ số thiếu ổn định.

**Dùng:** cần mô hình gọn/giải thích, quan hệ gần tuyến tính. **Không ưu tiên:** dữ liệu có quan hệ phi tuyến phức tạp mà không thể tạo feature phù hợp.

### Decision Tree

Cây đặt câu hỏi tuần tự như `income <= ngưỡng?`, phân vùng dữ liệu và dự đoán ở lá. Notebook chỉ đặt `random_state=42`; tham số độ phức tạp còn lại mặc định.

- `max_depth`: độ sâu tối đa; nhỏ giảm overfit nhưng quá nhỏ underfit.
- `min_samples_split`: số mẫu tối thiểu để tách nút; tăng làm cây bảo thủ hơn.
- `min_samples_leaf`: số mẫu tối thiểu trong lá; tăng làm kết quả ổn định hơn.
- `criterion`: độ đo split như Gini/entropy/log-loss tùy estimator/phiên bản.
- `class_weight`: trọng số lớp.
- `max_features`: số feature xét mỗi lần split.

**Mạnh:** phi tuyến/tương tác, không cần scaling, cây nông dễ diễn giải. **Yếu:** cây đơn bất ổn, phương sai và overfit cao; xác suất lá thô. **Dùng:** muốn quy tắc đơn giản. **Không dùng cây sâu không giới hạn** khi ưu tiên ổn định/tổng quát hóa.

### Random Forest

Huấn luyện nhiều cây trên bootstrap samples, mỗi lần split xét tập con feature, rồi tổng hợp phiếu/xác suất để giảm phương sai cây đơn. Notebook mặc định `n_estimators=300`, `n_jobs=-1`, `random_state=42`; có biến thể `class_weight="balanced"`.

- `n_estimators`: số cây; tăng thường ổn định hơn nhưng tốn RAM/thời gian, lợi ích giảm dần.
- `n_jobs=-1`: dùng các lõi CPU khả dụng; nhanh hơn nhưng chiếm tài nguyên.
- `random_state`: tái lập phần ngẫu nhiên.
- `max_depth`: giới hạn độ sâu; `None` cho cây phát triển đến điều kiện dừng, có thể overfit thành phần.
- `min_samples_split`: tối thiểu mẫu để chia.
- `min_samples_leaf`: tối thiểu mẫu mỗi lá; tăng giúp dự đoán bớt nhiễu.
- `max_features`: số feature thử tại split (`sqrt`, `log2`, `None`); sampling giảm tương quan giữa cây nhưng quá ít làm cây yếu.
- `class_weight="balanced"`: bù tần suất lớp; phải kiểm chứng chứ không tự động tốt hơn.

**Mạnh:** mô hình phi tuyến vững cho bảng, ít cần scale, thường tổng quát hóa hơn cây đơn. **Yếu:** lớn, khó giải thích trực tiếp, one-hot cardinality cao tốn RAM, ngoại suy kém; impurity importance thiên lệch với biến nhiều mức.

**Dùng:** ứng viên mạnh cho dữ liệu bảng. **Không nhất thiết dùng:** cần latency/mô hình nhỏ, ngoại suy, hoặc quy tắc giải thích chính xác.

### Extra Trees (Extremely Randomized Trees)

Ensemble tương tự forest nhưng ngẫu nhiên hóa ngưỡng chia mạnh hơn. Notebook: `n_estimators=300`, `n_jobs=-1`, `random_state=42`. `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` có vai trò điều chỉnh độ phức tạp/cân bằng như RF.

**Mạnh:** có thể giảm phương sai, đôi khi nhanh; benchmark bổ sung hữu ích. **Yếu:** ngẫu nhiên hơn có thể tăng bias; không đảm bảo tốt hơn RF; khó giải thích. **Dùng:** đối chứng ensemble cho dữ liệu bảng. **Không cần:** nếu tài nguyên hạn chế và các mô hình khác đã đáp ứng.

### HistGradientBoostingClassifier

Boosting cộng dồn cây tuần tự: cây sau sửa sai số ensemble trước; histogram binning giúp tìm split hiệu quả. Notebook: `learning_rate=0.08`, `max_iter=250`, `random_state=42`, đi với ordinal encoding.

- `learning_rate`: độ đóng góp mỗi cây; thấp thường cần nhiều vòng hơn.
- `max_iter`: số vòng/cây; tăng sức học và rủi ro overfit nếu thiếu regularization/early stop.
- `max_leaf_nodes` (quan trọng, không đặt): giới hạn lá mỗi cây.
- `max_depth`, `min_samples_leaf`, `l2_regularization` (tùy phiên bản): điều khiển cấu trúc và regularization.
- `random_state`: seed.

**Mạnh:** phi tuyến tốt trên dữ liệu bảng, không cần scale. **Yếu:** nhạy tuning; ordinal encoding category danh nghĩa áp thứ bậc giả; hỗ trợ category/NaN phụ thuộc phiên bản. **Dùng:** muốn thử boosting sklearn. **Không ưu tiên:** cần giải thích đơn giản hoặc ordinal encoding không hợp.

### XGBoost (tùy chọn)

Gradient boosting tối ưu loss, có regularization và sampling. Phải cài package. Notebook: `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `eval_metric="logloss"`, `n_jobs=-1`, `random_state=42`.

- `n_estimators`: số vòng/cây; tăng khả năng học và chi phí/overfit.
- `max_depth`: độ sâu; sâu hơn học tương tác phức tạp nhưng overfit hơn.
- `learning_rate` (`eta`): co đóng góp mỗi cây; thấp thường cần nhiều cây.
- `subsample`: tỷ lệ hàng lấy mẫu mỗi vòng; dưới 1 có thể regularize.
- `colsample_bytree`: tỷ lệ feature mỗi cây; giảm tương quan/overfit và thời gian.
- `min_child_weight` (trong tuning): ngưỡng tổng Hessian/trọng số tối thiểu nút con; cao thường bảo thủ.
- `reg_alpha`: L1, khuyến khích trọng số thưa.
- `reg_lambda`: L2, co trọng số.
- `scale_pos_weight`: trọng số positive; tỷ số âm/dương là điểm khởi đầu heuristic, không đảm bảo xác suất hiệu chuẩn hay metric tốt hơn.
- `eval_metric="logloss"`: metric loss log trong estimator; bảng notebook xếp hạng bằng AP.
- `n_jobs`, `random_state`: song song và tái lập.

**Mạnh:** thường rất tốt cho bảng, nhiều cơ chế regularization, bắt tương tác. **Yếu:** nhiều tham số, tuning tốn, khó giải thích trực tiếp; phụ thuộc package/version. **Dùng:** có tài nguyên tuning và cần benchmark. **Không cần:** không được cài thêm package, runtime hạn chế, hoặc mô hình đơn giản đã đủ.

Notebook chỉ tune RF và XGBoost. Logistic, Decision Tree, Extra Trees, HistGradientBoosting không được tune tự động. Giá trị khởi đầu không phải giá trị tối ưu; chỉ validation/CV phù hợp dùng cho lựa chọn.

## 11. Chỉ số đánh giá và chọn ngưỡng

### Confusion matrix

| Thực tế / dự đoán | Dự đoán 0 | Dự đoán 1 |
|---|---:|---:|
| Thực tế 0 | TN | FP |
| Thực tế 1 | FN | TP |

- **FN:** nợ xấu bị bỏ sót, có thể gây tổn thất tín dụng.
- **FP:** khách tốt bị gắn cờ, có thể mất cơ hội/đánh giá sai.

Ma trận phụ thuộc threshold, cần đọc với chi phí nghiệp vụ.

### Precision, recall, F1

- Precision lớp 1 = `TP/(TP+FP)`: trong số dự đoán xấu, bao nhiêu đúng.
- Recall lớp 1 = `TP/(TP+FN)`: trong số thực sự xấu, phát hiện được bao nhiêu.
- F1 là trung bình điều hòa precision/recall.
- `F1_0`, `F1_1`: F1 từng lớp one-vs-rest. `Weighted_F1`: trung bình có trọng số support, nên lớp đông ảnh hưởng nhiều.

F1 hữu ích cho cân bằng precision/recall nhưng không mã hóa chi phí riêng và bỏ qua true negatives.

### Average Precision và PR curve

`average_precision_score(y, probability)` tổng hợp precision theo recall qua các ngưỡng, đánh giá xếp hạng positive. Hữu ích khi lớp dương hiếm; baseline gần tỷ lệ positive, cần báo cùng prevalence. Model table xếp theo AP giảm dần rồi Weighted F1. AP dùng probability/score; F1 trong bảng dùng nhãn dự đoán ở threshold mặc định 0.5.

### Classification report

Báo precision, recall, F1, support cho mỗi lớp, macro average (trung bình đều) và weighted average. `digits=4` chỉ số chữ số; `zero_division=0` trả 0 nếu mẫu số bằng 0.

### Threshold tuning

Notebook thử 0.10–0.90 bước 0.05, tính precision/recall/F1 lớp 1 và weighted F1, rồi chọn dòng F1 lớp 1 lớn nhất.

Ngưỡng thấp thường dự đoán nhiều positive hơn: recall tăng, precision có thể giảm. Ngưỡng cao thường ít positive: precision có thể tăng, recall có thể giảm. Chọn theo chi phí và năng lực xử lý, không chỉ tối đa F1. Ngưỡng phải được chọn trên dữ liệu validation/CV; nếu dùng cùng validation để chọn mô hình, ngưỡng và báo kết quả cuối thì kết quả lạc quan do nhiều lần chọn. Cân nhắc nested CV hoặc holdout cuối.

## 12. Tuning và cross-validation

### RandomizedSearchCV

Lấy ngẫu nhiên một phần cấu hình trong không gian tìm kiếm, ít tốn hơn grid search lớn. Pipeline đảm bảo preprocessing được fit riêng từng fold.

- `n_iter=20`: thử 20 cấu hình; ít thì rẻ hơn nhưng dễ bỏ cấu hình tốt.
- `scoring="average_precision"`: tối ưu AP, hợp ưu tiên xếp hạng positive hiếm.
- `cv=3`: ba fold theo mặc định sklearn; temporal/group task cần truyền split phù hợp.
- `random_state=42`: tái lập việc lấy cấu hình.
- `n_jobs=-1`: chạy song song, cần CPU/RAM.
- `verbose=1`: in tiến độ.

Không gian Random Forest:

| Tham số | Giá trị thử | Ý nghĩa |
|---|---|---|
| `n_estimators` | 200, 300, 400, 500, 700 | Số cây/độ ổn định/chi phí |
| `max_depth` | 6, 8, 10, 12, 16, `None` | Độ sâu cây |
| `min_samples_split` | 2, 5, 10 | Mẫu tối thiểu để chia |
| `min_samples_leaf` | 1, 2, 3, 5 | Mẫu tối thiểu trong lá |
| `max_features` | `sqrt`, `log2`, `None` | Số biến xem ở split |
| `class_weight` | `None`, `balanced` | Điều chỉnh trọng số lớp |

Không gian XGBoost bổ sung `learning_rate [0.01,0.03,0.05,0.08,0.10]`, `subsample [0.7,0.8,0.9,1.0]`, `colsample_bytree` cùng dãy, `min_child_weight [1,3,5,8]`, `reg_alpha [0,0.01,0.1,0.5]`, `reg_lambda [1,2,5,10]`, `scale_pos_weight [1, class_ratio]` và số cây/độ sâu đã mô tả ở mục XGBoost.

`best_score_` là CV score trong tập train; `best_estimator_` sau đó được chấm trên holdout validation. Nếu dùng validation để tune rồi báo chính nó như kết quả độc lập thì bị optimistic bias.

### Cross-validation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` tạo năm fold với tỷ lệ target gần nhau. `cross_val_score(..., scoring="average_precision")` fit lại pipeline từng fold. Mean là mức trung bình, std cao cho thấy nhạy với cách chia.

Notebook chạy ví dụ trên toàn bộ `X,y`; nếu đã chọn mô hình/feature/ngưỡng bằng cùng dữ liệu thì CV không còn hoàn toàn độc lập. Temporal task dùng `make_time_splits` và thay `cv=3`; group task dùng split group. CV không sửa leakage trong feature hay định nghĩa nhãn.

## 13. Giải thích mô hình

### Tree feature importance

`feature_importances_` tổng hợp giảm impurity do biến tạo ra. Tên hiển thị là feature sau transform nên one-hot có thể tách thành nhiều hàng.

**Mạnh:** nhanh, sẵn có, xem biến mô hình đã dùng. **Yếu:** impurity importance thiên lệch về cardinality/continuous, tương quan chia sẻ importance, không cho chiều tác động và không chứng minh nhân quả.

### Hệ số Logistic Regression

Hệ số dương làm log-odds lớp dương tăng khi feature tăng, giữ biến khác cố định trong mô hình; hệ số âm làm giảm. Với hệ số β, odds ratio mỗi đơn vị là `exp(β)`; đơn vị là thang sau preprocessing. Feature đã scale/one-hot nên so sánh thận trọng; tương quan cao làm hệ số thiếu ổn định.

### Permutation importance

Xáo trộn một cột raw trong validation rồi đo đổi AP. Notebook dùng `n_repeats=5`, `random_state=42`. AP giảm nhiều khi xáo trộn → mô hình dựa vào feature đó.

**Mạnh:** gắn với metric ngoài mẫu, áp dụng nhiều mô hình. **Yếu:** tốn thời gian; biến tương quan thay thế nhau khiến từng cột có vẻ ít quan trọng; 5 lần có thể nhiễu.

### SHAP (tùy chọn)

SHAP phân rã dự đoán thành baseline cộng đóng góp biến theo cách phân bổ game-theoretic. Notebook lấy tối đa 500 dòng validation, preprocess, rồi `TreeExplainer` cho XGBoost và RF.

- **Bar:** importance toàn cục (thường trị tuyệt đối SHAP trung bình), không có hướng.
- **Beeswarm:** phân bố SHAP; màu thể hiện giá trị feature, trục ngang thể hiện hướng/độ lớn đóng góp theo model.
- **Waterfall:** giải thích một quan sát từ baseline đến output.

Random Forest lấy lớp 1 bằng `rf_shap_values[:, :, 1]`. SHAP output cho XGBoost thay đổi theo phiên bản/objective; xác nhận đúng lớp và thang output. SHAP giải thích model, không chứng minh nhân quả hay loại trừ leakage/bias.

## 14. Huấn luyện cuối, dự đoán, submission

Notebook đặt `final_model = random_forest`, `final_threshold = 0.50`, fit trên toàn `X,y`, gọi `predict_proba(X_test)[:,1]`, áp threshold, xem 10 dòng và tỷ lệ positive, ghi `target` vào submission template và lưu `submission_final.csv`.

Sau khi lựa chọn xong bằng train/validation, fit toàn dữ liệu nhãn để dùng nhiều dữ liệu hơn. Test không có nhãn nên không đánh giá được hiệu năng; tỷ lệ positive chỉ sanity check. Template phải khớp số dòng/thứ tự `X_test` và đúng tên/kiểu cột. Nếu dùng feature engineering phải tái tạo trên test. Nếu chọn `rf_best` hoặc threshold khác, cập nhật `final_model`, `final_threshold`, final summary nhất quán.

## 15. Khung viết report

### A. Bài toán và nhãn

- Đối tượng/thời điểm dự đoán, định nghĩa positive label và cửa sổ target.
- Kích thước train/validation/test, tỷ lệ lớp.
- Hậu quả FN/FP.

### B. Dữ liệu và xử lý

- Kiểu cột, missing, duplicate, constant, cardinality, infinity, outlier.
- Quyết định xử lý và lý do; phân biệt phần chỉ kiểm tra với phần thực sự thay đổi.
- Leakage review: feature có sẵn khi score không.
- PSI/shift và giới hạn diễn giải.
- Chỉ mô tả feature engineering thực sự đã chạy.

### C. Phương pháp

- Split theo random/time/group đúng kịch bản triển khai.
- Pipeline imputation, encoding, scaling.
- Baseline, mô hình, tham số, class weighting.
- Tuning space, AP objective, CV, seed.

### D. Kết quả

- AP, F1 lớp 0/1, weighted F1; thêm precision/recall khi cần.
- Tiêu chí chọn model, confusion matrix, PR curve, threshold và trade-off.
- CV mean/std; phân biệt CV, validation holdout, test prediction.

### E. Giải thích và giới hạn

- Feature importance/permutation/SHAP và giới hạn; tránh khẳng định nhân quả.
- Drift, calibration, tính đại diện, chi phí lỗi, fairness nếu thích hợp.
- Cải thiện có căn cứ: temporal/group validation, loại leakage, calibration, chọn threshold theo chi phí, theo dõi drift, bổ sung dữ liệu.

Notebook không lưu output tính toán, vì vậy hãy điền số thật sau khi chạy trên dữ liệu. Không trình bày cấu hình ví dụ như kết quả thực nghiệm.

## 16. Lưu ý trước khi dùng notebook với dữ liệu thật

1. Sửa tên target/ID/cột bỏ, tên file và cột submission theo dataset.
2. Không tạo lại `bad_debt` nếu target đã có; xác nhận quy tắc “hơn 3 tháng”.
3. Chọn random/time/group split đúng câu hỏi triển khai; các nhánh tùy chọn thay thế nhau.
4. Kiểm tra leakage và thời điểm feature có sẵn.
5. Sau feature engineering, tạo lại `X`, `X_test`, danh sách cột và split; xử lý chia 0, missing, miền log, biên bin.
6. Cardinality cao + one-hot dense có thể ngốn RAM; kiểm tra encoder và phiên bản sklearn.
7. Ordinal encoding category không thứ bậc là giả định mạnh.
8. Class weight/threshold cần dựa trên mục tiêu; F1 không thay hàm chi phí.
9. Sau tuning phải chọn đúng `best_estimator_`; final section hiện vẫn dùng RF ban đầu.
10. SHAP RF cần `rf_best` sau khi chạy tuning; XGBoost SHAP cần package/model tuning và kiểm tra format.
11. Chưa có dữ liệu/kết quả thực thì không thể kết luận mô hình nào thắng.
