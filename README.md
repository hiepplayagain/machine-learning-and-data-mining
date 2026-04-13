
# Dự án Machine Learning: Dự đoán giá nhà Hà Nội

English version: [README_EN.md](README_EN.md)

Mục tiêu dự án:
- Dự đoán giá nhà từ các đặc trưng bất động sản.
- Có khả năng dự đoán cho các năm tương lai mặc định từ 2025 đến 2027.

## 1. Cấu trúc dự án

- `main.py`: Script tiền xử lý dữ liệu, huấn luyện mô hình LightGBM và dự đoán các năm tương lai.
- `data/VN_housing_dataset.csv`: Dữ liệu đầu vào.

## 2. Cài đặt môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2.1 Quy tắc push code

- Không push thư mục môi trường ảo `.venv/` lên GitHub.
- Không push các file phát sinh như `__pycache__/`, `*.pyc`.
- Repository chỉ nên chứa source code, dữ liệu cần thiết, báo cáo và tài liệu.

File `.gitignore` đã được cấu hình để bỏ qua các thành phần trên.

Nếu bạn vừa clone/fork repo, hãy tự cài lại môi trường:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu

Script hỗ trợ trực tiếp format cột tiếng Việt, ví dụ:

- `Ngay`
- `Dia chi`
- `Quan`
- `Huyen`
- `Loai hinh nha o`
- `Giay to phap ly`
- `So tang`
- `So phong ngu`
- `Dien tich` (ví dụ `46 m2`)
- `Gia/m2` (ví dụ `86,96 trieu/m2`)

Script sẽ tự động:
- Tách số từ `Dien tich`, `So phong ngu`, `So tang`
- Lấy `year` từ cột `Ngay`
- Tạo thêm đặc trưng `area_per_bed` (diện tích mỗi phòng ngủ)
- Tạo thêm đặc trưng `rel_area_dist` (diện tích tương đối so với trung bình quận)
- Tính target `price` theo công thức:

$$
price = Dien\ tich \times Gia/m2 \times 1{,}000{,}000
$$

Ví dụ 1 dòng:

```csv
Ngay,Dia chi,Quan,Huyen,Loai hinh nha o,Giay to phap ly,So tang,So phong ngu,Dien tich,Dai,Rong,Gia/m2
2020-08-05,"Duong Hoang Quoc Viet, Phuong Nghia Do, Quan Cau Giay, Ha Noi",Quan Cau Giay,Phuong Nghia Do,"Nha ngo, hem",Da co so,4,5 phong,46 m2,,,"86,96 trieu/m2"
```

Khuyến nghị tối thiểu 30 dòng dữ liệu để mô hình học ổn định.

## 4. Chạy train và dự đoán

```bash
python3 main.py
```

Kết quả in ra màn hình gồm:
- Khoảng năm có trong dữ liệu train.
- Accuracy theo công thức `100 - MAPE` từ 3-fold cross-validation.
- Giá dự đoán cho các năm 2025, 2026, 2027 theo cả VND và tỷ.

Mô hình hiện tại dùng LightGBM hồi quy trên nhãn log-price.

## 4.1 LightGBM trong dự án tính dự đoán như thế nào?

Trong code, mô hình học hàm:

$$
f(\mathbf{x}) \approx \log(1 + price)
$$

với $\mathbf{x}$ là vector đặc trưng đã qua tiền xử lý (impute, scale số, one-hot cho biến phân loại).

Quy trình tính dự đoán gồm 5 bước chính:

1. Biến đổi đầu vào sang không gian đặc trưng
- Biến số (`area_m2`, `bedrooms`, `floors`, `year`, ...): điền thiếu bằng median và chuẩn hóa.
- Biến phân loại (`district`, `property_type`, `legal_status`, ...): điền thiếu bằng giá trị phổ biến nhất rồi one-hot encode.

2. Mô hình cộng dồn nhiều cây quyết định
- LightGBM xây một dãy cây CART.
- Ở vòng lặp thứ $t$, mô hình thêm cây mới $h_t(\mathbf{x})$ để giảm sai số còn lại (gradient) của mô hình hiện tại.
- Điểm dự đoán trên thang log là tổng có trọng số:

$$
\hat{y}_{log}(\mathbf{x}) = \sum_{t=1}^{T} \eta \cdot h_t(\mathbf{x})
$$

Trong đó $\eta$ là `learning_rate`, $T$ là số cây thực dùng (có thể nhỏ hơn `n_estimators` do early stopping).

3. Chọn điểm rẽ nhánh tối ưu bằng gain
- Mỗi cây chọn split sao cho giảm loss tốt nhất.
- LightGBM dùng histogram binning để tăng tốc tìm split trên dữ liệu lớn.

4. Early stopping để tránh overfit
- Code tách validation 15% và dùng `early_stopping(200)`.
- Nếu 200 vòng liên tiếp không cải thiện loss validation, quá trình train dừng sớm.

5. Đưa kết quả về đơn vị giá thực
- Dự đoán ra $\hat{y}_{log}$.
- Chặn dưới bằng log-price nhỏ nhất trong train để tránh giá âm phi thực tế.
- Chuyển ngược về VND:

$$
\hat{price} = \exp(\hat{y}_{log}) - 1
$$

Sau khi có `price_base` tại năm cuối của dữ liệu, script mới áp dụng tăng trưởng lãi kép để ngoại suy các năm tương lai:

$$
price_{future} = price_{base} \cdot (1 + r)^{\Delta year}
$$

với $r = 0.09$ theo cấu hình hiện tại.

Nếu muốn đổi khoảng năm dự đoán, sửa hằng số `FORECAST_YEARS` trong `main.py`.

## 5. Mở rộng tiếp theo

