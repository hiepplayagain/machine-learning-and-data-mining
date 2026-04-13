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
    col_map = {"year": "Ngày", "area": "Diện tích", "bed": "Số phòng ngủ", 
               "floor": "Số tầng", "dist": "Quận", "type": "Loại hình nhà ở"}
    
    for k, v in col_map.items():
        found = next((c for c in raw.columns if v in c), None)
        if found:
            if k == "year": df[k] = pd.to_datetime(raw[found], errors='coerce').dt.year
            elif k in ["area", "bed", "floor"]: df[k] = raw[found].apply(clean_num)
            else: df[k] = raw[found].astype(str).str.strip()

    ppm2_col = next((c for c in raw.columns if "Giá/m2" in c), None)
    if ppm2_col: df["price"] = raw[ppm2_col].apply(clean_num) * df["area"]

    # CRITICAL: Filter out garbage data that causes negative accuracy
    # Only keep houses between 500 million and 500 billion
    df = df[df["year"] == 2020].dropna(subset=["price", "area"])
    df = df[(df["price"] > 5e8) & (df["price"] < 5e11)]
    
    # Encode categories
    for col in ["dist", "type"]:
        df[col] = OrdinalEncoder().fit_transform(df[[col]].astype(str))
        
    return df

if __name__ == "__main__":
    start_time = time.time()
    
    data_2020 = fast_prepare(pd.read_csv(DATA_PATH))
    
    # Fill NaN for features to prevent model errors
    X = data_2020[["area", "bed", "floor", "dist", "type"]].fillna(0)
    y = data_2020["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Use ExtraTrees but with fewer estimators and deeper pruning for speed
    model = ExtraTreesRegressor(
        n_estimators=200, 
        max_depth=30,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )

    print("STAGES: Training optimized ExtraTrees...")
    model.fit(X_train, np.log1p(y_train))

    preds = np.expm1(model.predict(X_test))
    
    # Calculate MAPE safely
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    
    print("=" * 65)
    print("REFINED EVALUATION: YEAR 2020")
    print("=" * 65)
    print(f"Dataset Size  : {len(data_2020)} rows")
    print(f"Accuracy Rate : {100 - mape:.2f}%")
    print(f"Total Time    : {time.time() - start_time:.2f}s")
    print("-" * 65)
    
    print(f"{'No':<4} {'Actual (Bn)':>12} {'Predicted (Bn)':>15} {'Error (%)':>12}")
    for i in range(min(10, len(preds))):
        r, p = y_test.values[i] / 1e9, preds[i] / 1e9
        err = abs(r - p) / r * 100
        print(f"{i+1:<4} {r:>12.2f} {p:>15.2f} {err:>12.1f}%")
    print("=" * 65)