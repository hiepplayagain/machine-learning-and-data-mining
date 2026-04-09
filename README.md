
# Dự án Machine Learning: Dự đoán giá nhà Hà Nội

English version: [README_EN.md](README_EN.md)

Mục tiêu dự án:
- Dự đoán giá nhà từ các đặc trưng bất động sản.
- Có khả năng dự đoán cho các năm tương lai (ví dụ 2027, 2028).

## 1. Cấu trúc dự án

- `main.py`: Script tiền xử lý dữ liệu, huấn luyện mô hình và dự đoán năm tương lai.
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
python3 main.py --data data/VN_housing_dataset.csv --future-year 2028
```

Kết quả in ra màn hình gồm:
- Số dòng dữ liệu và số feature hợp lệ sau tiền xử lý.
- Giá dự đoán mẫu cho năm tương lai (VND).

Để tăng độ chính xác, mô hình dùng Random Forest Regressor và tự chọn bộ tham số tốt nhất bằng 5-fold cross-validation (GridSearchCV).

## 5. Mở rộng tiếp theo

