import re, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

DATA_PATH = "data/VN_housing_dataset.csv"
SEED      = 42

def clean_num(s):
    if pd.isna(s): return np.nan
    s = str(s).lower().replace(',', '.')
    m = 1
    if 'triệu' in s or 'trieu' in s: m = 1e6
    elif 'tỷ' in s or 'ty' in s: m = 1e9
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(nums[0]) * m if nums else np.nan

def fast_prepare(raw):
    df = pd.DataFrame()
    # Map các cột cần thiết, vẫn lấy 'Ngày' để lọc năm nhưng sẽ bỏ sau đó
    col_map = {"year": "Ngày", "area": "Diện tích", "bed": "Số phòng ngủ",
               "floor": "Số tầng", "dist": "Quận", "type": "Loại hình nhà ở"}

    for k, v in col_map.items():
        found = next((c for c in raw.columns if v in c), None)
        if found:
            if k == "year": 
                df[k] = pd.to_datetime(raw[found], errors='coerce').dt.year
            elif k in ["area", "bed", "floor"]: 
                df[k] = raw[found].apply(clean_num)
            else: 
                df[k] = raw[found].astype(str).str.strip()

    # Tính toán cột giá mục tiêu (Target)
    ppm2_col = next((c for c in raw.columns if "Giá/m2" in c), None)
    if ppm2_col: 
        df["price"] = raw[ppm2_col].apply(clean_num) * df["area"]

    # LỌC: Chỉ giữ duy nhất năm 2020 và loại bỏ dữ liệu thiếu
    df = df[df["year"] == 2020].dropna(subset=["price", "area"])
    
    # Lọc bỏ giá trị ngoại lai để mô hình ổn định hơn
    df = df[(df["price"] > 5e8) & (df["price"] < 5e11)]

    # Mã hóa các biến phân loại (Quận, Loại hình)
    for col in ["dist", "type"]:
        df[col] = OrdinalEncoder().fit_transform(df[[col]].astype(str))

    return df


if __name__ == "__main__":
    start_time = time.time()
    
    if not pd.io.common.file_exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} không tồn tại.")
    else:
        raw = pd.read_csv(DATA_PATH)

        print("STAGE: Preparing 2020-only dataset...")
        df = fast_prepare(raw)

        # FEATURES: Loại bỏ 'year' vì toàn bộ là 2020
        FEATURES = ["area", "bed", "floor", "dist", "type"]
        X = df[FEATURES].fillna(0)
        y = df["price"]

        # Chia tập Train/Test (Không cần stratify theo năm nữa)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        model = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            random_state=SEED,
            n_jobs=-1
        )

        print("STAGE: Training on 2020 dataset...")
        # Sử dụng log1p để xử lý độ lệch (skewness) của giá nhà
        model.fit(X_train, np.log1p(y_train))
        
        preds = np.expm1(model.predict(X_test))
        mape  = mean_absolute_percentage_error(y_test, preds) * 100

        print("=" * 65)
        print("EVALUATION: DATASET 2020 ONLY")
        print("=" * 65)
        print(f"Total Records : {len(df)} rows")
        print(f"Train Size    : {len(X_train)} | Test Size: {len(X_test)}")
        print(f"Accuracy Rate : {100 - mape:.2f}%")
        print(f"MAPE          : {mape:.2f}%")
        print("-" * 65)

        # In kết quả dự đoán mẫu
        print(f"{'No':<4} {'Actual (Bn)':>15} {'Predicted (Bn)':>18} {'Error (%)':>12}")
        for i in range(min(10, len(preds))):
            actual_val = y_test.values[i] / 1e9
            pred_val   = preds[i] / 1e9
            error_pct  = abs(actual_val - pred_val) / actual_val * 100
            print(f"{i+1:<4} {actual_val:>15.2f} {pred_val:>18.2f} {error_pct:>11.1f}%")

        print("=" * 65)
        print(f"Total Execution Time: {time.time() - start_time:.2f}s")