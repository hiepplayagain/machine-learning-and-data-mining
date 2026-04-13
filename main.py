import re, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── System Configuration ─────────────────────────────────────────────────────
DATA_PATH   = "data/VN_housing_dataset.csv"
FORECAST_YEARS = [2025, 2026, 2027]
TARGET, SEED = "price", 42
ANNUAL_GROWTH_RATE = 0.09  # Assumed annual market growth rate: 9%

ALIASES = {
    "year":          ["Ngay", "Date", "year"],
    "area_m2":       ["Dien tich", "area", "area m2"],
    "bedrooms":      ["So phong ngu", "bedrooms"],
    "floors":        ["So tang", "floors"],
    "district":      ["Quan", "district"],
    "property_type": ["Loai hinh nha o", "property type"],
    "legal_status":  ["Giay to phap ly", "legal"],
}

# ── Data Processing Utilities ────────────────────────────────────────────────

def norm(t):
    # Normalize text: remove diacritics, lowercase, and replace separators with spaces.
    s = pd.Series([str(t)]).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("utf-8").str.lower().str.strip().iloc[0]
    for p in "/,-._()": s = s.replace(p, " ")
    return " ".join(s.split())

def find_col(cols, aliases):
    # Resolve the real column name in the dataset from a list of possible aliases.
    m = {norm(c): c for c in cols}
    return next((m[norm(a)] for a in aliases if norm(a) in m), None)

def to_num(s):
    # Extract numeric value from free-form text (e.g., "86,96 trieu/m2" -> 86.96); invalid values become NaN.
    return pd.to_numeric(
        s.astype("string").str.lower().str.replace(" ","",regex=False)
         .str.replace(",",".",regex=False).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce")

def sanitize(name):
    # Convert feature names to safe ASCII identifiers for LightGBM column names.
    s = pd.Series([str(name)]).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("utf-8").iloc[0]
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def prepare(raw):
    # Preprocessing pipeline: map columns, cast types, create target/features, and filter outliers.
    df = raw.drop(columns=[c for c in raw.columns if norm(c).startswith("unnamed")], errors="ignore").copy()
    
    for feat, aliases in ALIASES.items():
        src = find_col(df.columns.tolist(), aliases)
        if not src: continue
        if feat == "year": df[feat] = pd.to_datetime(df[src], errors="coerce").dt.year
        elif feat in {"area_m2","bedrooms","floors"}: df[feat] = to_num(df[src])
        else: df[feat] = df[src].astype(str).str.strip().replace("nan", "Unknown")

    ppm2 = find_col(df.columns.tolist(), ["Gia/m2","price/m2","gia m2"])
    if TARGET not in df.columns and ppm2 and "area_m2" in df.columns:
        df[TARGET] = to_num(df[ppm2]) * 1_000_000 * df["area_m2"]
    
    if TARGET in df.columns:
        # Remove implausible values to stabilize model training.
        df = df[(df[TARGET] > 8e8) & (df[TARGET] < 5e11)].copy()
        if "area_m2" in df.columns:
            df = df[(df["area_m2"] > 15) & (df["area_m2"] < 1500)]

    if "year" in df.columns:
        df["year"] = df["year"].fillna(df["year"].median())
    
    # Feature engineering
    if {"area_m2","bedrooms"}.issubset(df.columns): 
        df["area_per_bed"] = df["area_m2"] / df["bedrooms"].replace(0, np.nan)
    if {"district","area_m2"}.issubset(df.columns):
        avg_area_dist = df.groupby("district")["area_m2"].transform("mean")
        df["rel_area_dist"] = df["area_m2"] / avg_area_dist

    keep = [c for c in [*ALIASES,"area_per_bed","rel_area_dist",TARGET] if c in df.columns]
    return df[keep].dropna(subset=[TARGET])

# ── Advanced Valuation Model ─────────────────────────────────────────────────

class ValuationModel:
    def __init__(self):
        # pre: feature transformer, lgbm: trained LightGBM model, fnames: encoded feature names.
        self.pre = None
        self.lgbm = None
        self.fnames = []
        self.min_price_log = 0

    def _build_pre(self, X):
        # Numeric -> median imputation + scaling; categorical -> mode imputation + one-hot encoding.
        num = X.select_dtypes("number").columns.tolist()
        cat = X.select_dtypes(exclude="number").columns.tolist()
        return ColumnTransformer([
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
            ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                            ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ], verbose_feature_names_out=False)

    def fit(self, X, y):
        # Train on log(price) to reduce outlier skew and stabilize relative error.
        log_y = np.log1p(y)
        self.min_price_log = log_y.min()
        
        self.pre = self._build_pre(X)
        Xe = self.pre.fit_transform(X)
        self.fnames = [sanitize(n) for n in self.pre.get_feature_names_out()]
        Xe_df = pd.DataFrame(Xe, columns=self.fnames)
        
        Xtr, Xva, ytr, yva = train_test_split(Xe_df, log_y, test_size=0.15, random_state=SEED)
        
        self.lgbm = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.01, num_leaves=127,
            max_depth=12, reg_alpha=1.0, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, verbose=-1
        )
        self.lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])

    def predict(self, X):
        # Inference flow: transform features -> predict log(price) -> map back to VND with expm1.
        Xe = pd.DataFrame(self.pre.transform(X), columns=self.fnames)
        pred_log = self.lgbm.predict(Xe)
        # Prevent unrealistic negative prices by clamping to the minimum log-price seen in training.
        return np.expm1(np.maximum(pred_log, self.min_price_log))

# ── Execution ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Processing dataset...")
    df = prepare(pd.read_csv(DATA_PATH))
    
    print(f"Clean rows: {len(df)}. Year range in file: {int(df['year'].min())}-{int(df['year'].max())}")
    
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    
    print("Evaluating valuation accuracy (3-fold CV)...")
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    mapes = []
    
    for tr_idx, va_idx in kf.split(X):
        m = ValuationModel()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict(X.iloc[va_idx])
        yt = y.iloc[va_idx]
        mapes.append(np.mean(np.abs((yt - p) / yt)) * 100)
        
    accuracy = 100 - np.mean(mapes)
    
    print("Training final model on 100% of data...")
    final_model = ValuationModel()
    final_model.fit(X, y)
    
    print(f"Done! Valuation accuracy: {accuracy:.2f}%")
    print(f"Applied expected annual growth rate: {ANNUAL_GROWTH_RATE*100}%")
    
    print(f"\n{'Year':<8} {'Forecast Price (VND)':>25} {'Billion':>10}")
    print("-" * 50)
    
    # Estimate a representative home at the latest year present in the dataset.
    base_row = X.mode().iloc[0].to_dict()
    current_year_in_data = int(df['year'].max())
    base_row['year'] = current_year_in_data
    
    price_base = final_model.predict(pd.DataFrame([base_row]))[0]
    print(f"{current_year_in_data} (Base) {price_base:>20,.0f} {price_base/1e9:>9.2f}")
    
    # Extrapolate future prices using compound growth.
    for yr in FORECAST_YEARS:
        years_diff = yr - current_year_in_data
        p_future = price_base * ((1 + ANNUAL_GROWTH_RATE) ** years_diff)
        print(f"{yr:<8} {p_future:>25,.0f} {p_future/1e9:>9.2f}")