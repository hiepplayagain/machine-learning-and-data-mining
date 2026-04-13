
# Du an Machine Learning: Du doan gia nha Viet Nam

English version: [README_EN.md](README_EN.md)

Muc tieu:
- Tien xu ly du lieu bat dong san tu file CSV.
- Huan luyen mo hinh hoi quy de du doan gia nha.
- Danh gia nhanh tren tap test bang chi so MAPE.

## 1. Cau truc du an

- `main.py`: Tien xu ly du lieu, tao dac trung, huan luyen mo hinh va in ket qua.
- `data/VN_housing_dataset.csv`: Du lieu dau vao.
- `requirements.txt`: Danh sach thu vien can cai.
- `train_report.json`, `reports/train_report.json`: Tep bao cao (neu co).

## 2. Cai dat moi truong

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Dinh dang du lieu dau vao

Chuong trinh ho tro anh xa ten cot qua bang alias trong `main.py`, gom:
- `Ngay` hoac `year` -> `year`
- `Dien tich` hoac `area` -> `area_m2`
- `So phong ngu` hoac `bedrooms` -> `bedrooms`
- `So tang` hoac `floors` -> `floors`
- `Quan` hoac `district` -> `district`
- `Loai hinh nha o` hoac `property_type` -> `property_type`

Gia muc tieu `price` duoc tinh tu cot `Gia/m2` (hoac `Gia/m2`) nhan voi `area_m2`.

Ham xu ly so `clean_numeric_string` co the doc cac dang nhu:
- `86,96 trieu/m2`
- `2.5 ty`
- `46 m2`

Va tu dong quy doi:
- `trieu` -> 1,000,000
- `ty` -> 1,000,000,000

## 4. Tien xu ly va tao dac trung

Trong `prepare_advanced`, du lieu duoc xu ly theo cac buoc:
- Rut trich cac cot can thiet tu alias.
- Chuyen cot ngay sang nam (`year`).
- Loai bo dong thieu cac cot bat buoc: `price`, `year`, `area_m2`.
- Tao dac trung `m2_per_bed = area_m2 / bedrooms` (thay 0 bang 1 de tranh chia cho 0).
- Tao dac trung `district_rank` bang trung binh `log1p(price)` theo tung quan.
- Loc outlier cho `price` va `area_m2` theo nguong phan vi 5% va 95%.

## 5. Huan luyen mo hinh

Script chia du lieu theo nam:
- Train: `year < 2020`
- Test: `year == 2020`

Bo dac trung huan luyen:
- `area_m2`
- `bedrooms`
- `floors`
- `m2_per_bed`
- `district_rank`

Pipeline mo hinh:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `ExtraTreesRegressor` voi:
	- `n_estimators=1000`
	- `max_depth=40`
	- `min_samples_split=2`
	- `random_state=42`
	- `n_jobs=-1`

Nhan huan luyen su dung bien doi log:
- Train tren `log1p(price)`
- Du doan xong doi nguoc bang `expm1`

## 6. Chay chuong trinh

```bash
python3 main.py
```

Man hinh se in:
- Trang thai huan luyen.
- Do chinh xac theo cong thuc `100 - MAPE`.
- Thoi gian xu ly.
- Mot so dong so sanh gia thuc te va gia du doan (don vi ty).

## 7. Luu y

- Neu tap test nam 2020 khong co du lieu, can dieu chinh cach chia train/test trong `main.py`.
- Du lieu dau vao nen du lon va da duoc lam sach de giam MAPE.
- Du an hien tai la bai toan danh gia tren nam 2020, khong co phan du doan cac nam tuong lai.

