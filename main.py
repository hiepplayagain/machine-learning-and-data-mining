import re, warnings, time, json, unicodedata, pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
import stopwordsiso as stopwords

DATA_PATH = "data/VN_housing_dataset.csv"
SEED      = 42
STOPWORDS_LIST = stopwords.stopwords("vi")
IMPORTANT_TOKENS = {
    "ngay", "dien", "tich", "phong", "ngu", "so", "tang", "quan",
    "loai", "hinh", "nha", "o", "gia", "m2"
}


def remove_vietnamese_accents(text):
    """Convert Vietnamese text to ASCII by stripping diacritics.

    This helps normalize heterogeneous column names before keyword matching.
    """
    normalized_text = unicodedata.normalize("NFD", str(text))
    return "".join(character for character in normalized_text if unicodedata.category(character) != "Mn")


def normalize_vietnamese_column_name(column_name, stopwords_list):
    """Normalize a raw column name into a compact token string.

    Steps:
    1) Lowercase and remove accents.
    2) Keep only alphanumeric tokens.
    3) Drop stopwords, except important domain tokens.
    4) Join tokens with underscore.
    """
    clean_text = remove_vietnamese_accents(column_name).lower()
    clean_text = re.sub(r"[^a-z0-9]+", " ", clean_text).strip()
    tokens = [
        token for token in clean_text.split()
        if token and (token not in stopwords_list or token in IMPORTANT_TOKENS)
    ]
    return "_".join(tokens) if tokens else "col"


def rename_columns_with_stopwords(raw_dataframe):
    """Rename all columns using normalized Vietnamese tokens.

    Returns:
    - renamed_dataframe: dataframe with normalized names.
    - rename_mapping: dict from original to normalized names.
    """
    renamed_dataframe = raw_dataframe.copy()
    rename_mapping = {}
    used_names = set()

    for original_column_name in raw_dataframe.columns:
        base_name = normalize_vietnamese_column_name(original_column_name, STOPWORDS_LIST)
        candidate_name = base_name
        suffix_index = 1

        while candidate_name in used_names:
            suffix_index += 1
            candidate_name = f"{base_name}_{suffix_index}"

        used_names.add(candidate_name)
        rename_mapping[original_column_name] = candidate_name

    renamed_dataframe = renamed_dataframe.rename(columns=rename_mapping)
    return renamed_dataframe, rename_mapping


def find_column_by_keywords(columns, required_keywords):
    """Find the first column containing all required keywords."""
    return next(
        (
            column_name
            for column_name in columns
            if all(keyword in column_name for keyword in required_keywords)
        ),
        None,
    )

def save_evaluation_plot(y_true, y_pred, output_path):
    """Save a single evaluation chart: Actual vs Predicted scatter."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_bn = y_true / 1e9
    y_pred_bn = y_pred / 1e9

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axis = plt.subplots(1, 1, figsize=(8, 7))

    sns.scatterplot(x=y_true_bn, y=y_pred_bn, ax=axis, s=30, alpha=0.65, color="#2a9d8f")
    lo = min(y_true_bn.min(), y_pred_bn.min())
    hi = max(y_true_bn.max(), y_pred_bn.max())
    axis.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Ideal line (y=x)")
    axis.set_title("Actual vs Predicted Prices (2020)")
    axis.set_xlabel("Actual Price (Bn VND)")
    axis.set_ylabel("Predicted Price (Bn VND)")
    axis.legend()

    fig.suptitle("Model Evaluation Scatter - VN Housing 2020", fontsize=13, weight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def clean_num(s):
    """Parse Vietnamese numeric strings and convert to VND float.

    Supports common forms such as:
    - "86,96 trieu/m2"
    - "2.5 ty"
    - "46"
    """
    if pd.isna(s): return np.nan
    s = str(s).lower().replace(',', '.')
    m = 1
    if 'triệu' in s or 'trieu' in s: m = 1e6
    elif 'tỷ' in s or 'ty' in s: m = 1e9
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(nums[0]) * m if nums else np.nan

def practice_prepare(raw_dataframe):
    """Stage 1/3 (practice): prepare model-ready data from raw CSV data.

    This stage normalizes column names, maps core features, computes target
    price from price-per-m2 and area, filters records, and encodes categories.
    """
    normalized_raw_dataframe, rename_mapping = rename_columns_with_stopwords(raw_dataframe)
    df = pd.DataFrame()
    column_rules = {
        "year": ["ngay"],
        "area": ["dien", "tich"],
        "bed": ["phong", "ngu"],
        "floor": ["so", "tang"],
        "dist": ["quan"],
        "type": ["loai", "hinh", "nha", "o"],
    }

    # Resolve source columns from normalized names using keyword rules.
    selected_columns = {
        feature_name: find_column_by_keywords(normalized_raw_dataframe.columns, keyword_group)
        for feature_name, keyword_group in column_rules.items()
    }

    for feature_name, source_column_name in selected_columns.items():
        if source_column_name is None:
            continue

        if feature_name == "year":
            df[feature_name] = pd.to_datetime(normalized_raw_dataframe[source_column_name], errors="coerce").dt.year
        elif feature_name in ["area", "bed", "floor"]:
            df[feature_name] = normalized_raw_dataframe[source_column_name].apply(clean_num)
        else:
            df[feature_name] = normalized_raw_dataframe[source_column_name].astype(str).str.strip()

    price_per_m2_column = find_column_by_keywords(normalized_raw_dataframe.columns, ["gia", "m2"])
    # Compute target price from unit price and area when both are available.
    if price_per_m2_column and "area" in df.columns:
        df["price"] = normalized_raw_dataframe[price_per_m2_column].apply(clean_num) * df["area"]

    # FILTER: Keep only 2020 records if the year column is available.
    if "year" in df.columns:
        df = df[df["year"] == 2020]

    required_columns = [column_name for column_name in ["price", "area"] if column_name in df.columns]
    if len(required_columns) < 2:
        missing_columns = sorted(set(["price", "area"]) - set(required_columns))
        raise ValueError(f"Missing required columns after preparation: {missing_columns}")

    df = df.dropna(subset=required_columns)

    # Remove outliers to make training more stable.
    df = df[(df["price"] > 5e8) & (df["price"] < 5e11)]

    # Encode categorical features (District, Property Type).
    for categorical_column_name in ["dist", "type"]:
        if categorical_column_name in df.columns:
            df[categorical_column_name] = OrdinalEncoder().fit_transform(df[[categorical_column_name]].astype(str))

    return df, rename_mapping


def train_model(X_train, y_train):
    """Stage 2/3 (train): fit ExtraTreesRegressor on log-transformed target."""
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, np.log1p(y_train))
    return model


def test_model(model, X_test, y_test):
    """Stage 3/3 (test): run inference and compute evaluation metrics."""
    preds = np.expm1(model.predict(X_test))
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return preds, {"mape": float(mape), "mae": float(mae), "r2": float(r2)}


def export_production_bundle(model, metrics, features, predictions_dataframe, rename_mapping):
    """Export model and metadata artifacts for production usage.

    Output files:
    - housing_price_model_2020.pkl
    - test_predictions.csv
    - model_metadata.json
    """
    production_dir = Path("production")
    production_dir.mkdir(parents=True, exist_ok=True)

    with open(production_dir / "housing_price_model_2020.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    predictions_dataframe.to_csv(production_dir / "test_predictions.csv", index=False)

    metadata = {
        "dataset": "VN_housing_dataset.csv",
        "year_scope": 2020,
        "features": features,
        "metrics": metrics,
        "rename_mapping": rename_mapping,
    }
    with open(production_dir / "model_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    return production_dir

def predict_manual_input(model, available_features, metrics):
    """
    Allow manual input of house features to get a price prediction with accuracy rate.
    """
    print("\n--- ENTER HOUSE INFORMATION FOR PRICE PREDICTION ---")
    print("\nType: (Apartment = 0, Streetfront house = 1)")
    input_data = {}
    
    for feature in available_features:
        val = input(f"Enter {feature}: ")
        input_data[feature] = float(val)

    input_df = pd.DataFrame([input_data])
    
    log_prediction = model.predict(input_df)
    final_price = np.expm1(log_prediction)[0]
    
    # Compute model accuracy and estimated prediction range.
    mape = metrics['mape']
    accuracy = 100 - mape
    price_variance = final_price * (mape / 100) # Estimated absolute deviation based on MAPE.
    
    lower_bound = (final_price - price_variance) / 1e9
    upper_bound = (final_price + price_variance) / 1e9
    
    print("-" * 50)
    print(f"PREDICTED PRICE    : {final_price / 1e9:.3f} Billion VND")
    print(f"MODEL ACCURACY     : ~{accuracy:.2f}%")
    print(f"ESTIMATED RANGE    : {lower_bound:.3f} - {upper_bound:.3f} Billion VND")
    print("-" * 50)

if __name__ == "__main__":
    start_time = time.time()
    
    if not pd.io.common.file_exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} does not exist.")
    else:
        raw = pd.read_csv(DATA_PATH)

        print("STAGE 1/3 - PRACTICE: Normalizing data and Vietnamese column names...")
        df, rename_mapping = practice_prepare(raw)

        # FEATURES: exclude year because all rows are 2020 after filtering.
        FEATURES = ["area", "bed", "floor", "dist", "type"]
        available_features = [feature_name for feature_name in FEATURES if feature_name in df.columns]

        X = df[available_features].fillna(0)
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        print("STAGE 2/3 - TRAIN: Training ExtraTreesRegressor...")
        model = train_model(X_train, y_train)

        print("STAGE 3/3 - TEST: Evaluating, visualizing, and exporting production artifacts...")
        preds, metrics = test_model(model, X_test, y_test)

        print("=" * 65)
        print("EVALUATION: DATASET 2020 ONLY")
        print("=" * 65)
        print(f"Total Records : {len(df)} rows")
        print(f"Train Size    : {len(X_train)} | Test Size: {len(X_test)}")
        print(f"Accuracy Rate : {100 - metrics['mape']:.2f}%")
        print(f"MAPE          : {metrics['mape']:.2f}%")
        print(f"MAE           : {metrics['mae'] / 1e9:.3f} Bn VND")
        print(f"R2 Score      : {metrics['r2']:.4f}")
        print("-" * 65)
        print("Sample Prediction Results on the Test Set")
        print("-" * 65)

        print(f"{'No':<4} {'Actual (Bn)':>15} {'Predicted (Bn)':>18} {'Error (%)':>12}")
        for row_index in range(min(10, len(preds))):
            actual_val = y_test.values[row_index] / 1e9
            pred_val = preds[row_index] / 1e9
            error_pct  = abs(actual_val - pred_val) / actual_val * 100
            print(f"{row_index+1:<4} {actual_val:>15.2f} {pred_val:>18.2f} {error_pct:>11.1f}%")

        plot_path = Path("reports/model_evaluation_dashboard_2020.png")
        save_evaluation_plot(y_test.values, preds, plot_path)
        print(f"Saved evaluation plot: {plot_path}")

        prediction_output = pd.DataFrame({
            "actual_price": y_test.values,
            "predicted_price": preds,
            "error_pct": np.abs((preds - y_test.values) / y_test.values) * 100,
        })
        production_dir = export_production_bundle(model, metrics, available_features, prediction_output, rename_mapping)
        print(f"Production artifacts exported to: {production_dir}")

        print("=" * 65)
        print(f"Total Execution Time: {time.time() - start_time:.2f}s")

    predict_manual_input(model, available_features, metrics)
