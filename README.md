
# Du an Machine Learning: Du doan gia nha Ha Noi

Muc tieu du an:
- Du doan gia nha tu cac dac trung bat dong san.
- Co kha nang du doan cho cac nam tuong lai (vi du 2027, 2028).

## 1. Cau truc du an

- `main.py`: Script train model va du doan.
- `data/hanoi_house_prices.csv`: Du lieu train (ban tu tao).

## 2. Cai dat moi truong

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2.1 Quy tac push code

- Khong push thu muc moi truong ao `.venv/` len GitHub.
- Khong push cac file thu vien/he thong phat sinh (`__pycache__/`, `*.pyc`).
- Repo chi push source code, data can thiet, report va tai lieu.

File `.gitignore` da duoc cau hinh san de bo qua cac thanh phan tren.

Neu ban vua clone/fork repo, hay tu cai lai moi truong:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Chuan bi du lieu

Ban co the dung truc tiep format du lieu tieng Viet nhu ban da gui, vi du cac cot:

- `Ngay`
- `Dia chi`
- `Quan`
- `Huyen`
- `Loai hinh nha o`
- `Giay to phap ly`
- `So tang`
- `So phong ngu`
- `Dien tich` (vi du `46 m2`)
- `Gia/m2` (vi du `86,96 trieu/m2`)

Script se tu dong:
- Tach so tu `Dien tich`, `So phong ngu`, `So tang`
- Lay `year` tu cot `Ngay`
- Tinh target `price` theo cong thuc:

$$
price = Dien\ tich \times Gia/m2 \times 1{,}000{,}000
$$

Vi du 1 dong:

```csv
Ngay,Dia chi,Quan,Huyen,Loai hinh nha o,Giay to phap ly,So tang,So phong ngu,Dien tich,Dai,Rong,Gia/m2
2020-08-05,"Duong Hoang Quoc Viet, Phuong Nghia Do, Quan Cau Giay, Ha Noi",Quan Cau Giay,Phuong Nghia Do,"Nha ngo, hem",Da co so,4,5 phong,46 m2,,,"86,96 trieu/m2"
```

Khuyen nghi toi thieu 30 dong du lieu de mo hinh hoc on dinh.

## 4. Chay train va du doan

```bash
python3 main.py --data data/VN_housing_dataset.csv --future-year 2028 --report-out reports/train_report.json --chart-out reports/train_report.png
```

Ket qua se in:
- MAE (VND)
- RMSE (VND)
- R2
- MAPE (%)
- Gia du doan mau cho nam tuong lai (VND)

Ngoai ra script se luu report JSON tai `reports/train_report.json`, gom:
- thong tin dataset (so dong, so features, train/test size)
- cac metric danh gia
- thong tin du doan nam tuong lai

Script cung ve bieu do truc quan va:
- hien cua so chart tren man hinh (mac dinh)
- luu anh chart tai `reports/train_report.png`

Neu chi muon luu file ma khong mo cua so chart:

```bash
python3 main.py --data data/VN_housing_dataset.csv --future-year 2028 --no-show-chart
```

## 5. Mo rong tiep theo

- Them dac trung: huong nha, mat tien, khoang cach den trung tam, phuong, loai nha.
- Thu mo hinh manh hon: RandomForest, XGBoost, LightGBM.
- Luu model ra file (`joblib`) va tao API voi FastAPI.
