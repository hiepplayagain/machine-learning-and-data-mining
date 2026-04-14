# Dự án Machine Learning: Dự đoán giá nhà Việt Nam

English version: [README_EN.md](README_EN.md)

Mục tiêu:
- Chuẩn hóa tên cột tiếng Việt bằng stopwordsiso.
- Tiền xử lý dữ liệu nhà đất và tính giá mục tiêu từ Giá/m2.
- Huấn luyện mô hình ExtraTreesRegressor.
- Đánh giá trực quan bằng matplotlib + seaborn.
- Xuất bộ file production để sử dụng lại.

## 1. Cấu trúc dự án

- `main.py`: Pipeline 3 giai đoạn `practice -> train -> test`.
- `data/VN_housing_dataset.csv`: Dữ liệu đầu vào.
- `reports/model_evaluation_dashboard_2020.png`: Biểu đồ scatter Actual vs Predicted để đánh giá mô hình.
- `production/housing_price_model_2020.pkl`: Model đã huấn luyện.
- `production/model_metadata.json`: Metadata (feature, metric, mapping tên cột).
- `production/test_predictions.csv`: Kết quả dự đoán trên tập test.
- `requirements.txt`: Thư viện cần cài.

## 2. Cài đặt môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Tiền xử lý tên cột tiếng Việt

Script dùng `stopwordsiso` (`vi`) để bỏ từ dừng trong tên cột, kết hợp:
- Bỏ dấu tiếng Việt.
- Tách token alphanumeric.
- Giữ lại các token nghiệp vụ quan trọng (`ngay`, `dien`, `tich`, `gia`, `m2`, ...).
- Đổi tên cột về dạng `snake_case`.

Ví dụ:
- `Thông tin Ngày đăng` -> `ngay_dang`
- `Giá/m2 (triệu)` -> `gia_m2_trieu`

## 4. Pipeline 3 giai đoạn

### Stage 1/3 - Practice

- Chuẩn hóa tên cột tiếng Việt.
- Tìm các cột cần thiết theo keyword (`year`, `area`, `bed`, `floor`, `dist`, `type`).
- Tính `price = price_per_m2 * area`.
- Lọc dữ liệu năm 2020 (nếu có cột năm).
- Loại bỏ outlier giá (`5e8` đến `5e11`).
- Mã hóa cột phân loại (`dist`, `type`) bằng `OrdinalEncoder`.

### Stage 2/3 - Train

- Chia `train_test_split(test_size=0.2, random_state=42)`.
- Huấn luyện `ExtraTreesRegressor` trên `log1p(price)`.

### Stage 3/3 - Test

- Suy luận và đổi ngược bằng `expm1`.
- Tính metric: `MAPE`, `MAE`, `R2`.
- Vẽ duy nhất biểu đồ **Actual vs Predicted Scatter**.
- Xuất artifact production.

## 5. Giải thích biểu đồ Actual vs Predicted Scatter

Biểu đồ này so sánh giá thật (`Actual`) và giá dự đoán (`Predicted`) trên tập test.

- Trục X: Giá thực tế (tỷ VND).
- Trục Y: Giá mô hình dự đoán (tỷ VND).
- Đường đỏ nét đứt `y = x`: đường lý tưởng, nghĩa là dự đoán bằng đúng giá thật.

Cách đọc nhanh:
- Điểm càng gần đường `y = x` thì dự đoán càng chính xác.
- Điểm nằm **trên** đường `y = x`: mô hình dự đoán cao hơn thực tế (over-predict).
- Điểm nằm **dưới** đường `y = x`: mô hình dự đoán thấp hơn thực tế (under-predict).
- Nếu ở vùng giá cao các điểm lệch xa đường hơn, mô hình đang khó học tốt với nhóm nhà giá cao.

## 6. Chạy chương trình

```bash
python3 main.py
```

Màn hình sẽ in:
- Trạng thái của 3 stage.
- Số dòng train/test.
- `Accuracy = 100 - MAPE`.
- `MAPE`, `MAE`, `R2`.
- Mẫu 10 dòng dự đoán.

## 7. Artifact production

Sau khi chạy thành công, thư mục `production` gồm:
- `housing_price_model_2020.pkl`
- `model_metadata.json`
- `test_predictions.csv`

## 8. Lưu ý

- `stopwordsiso` hiện vẫn sử dụng `pkg_resources`; đã pin `setuptools<81` trong requirements để tương thích.
- Nếu bộ dữ liệu thay đổi mạnh tên cột, có thể cần cập nhật `column_rules` trong `main.py`.

