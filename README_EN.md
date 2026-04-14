# Machine Learning Project: Vietnam House Price Prediction

Vietnamese version: [README.md](README.md)

Goals:
- Normalize Vietnamese column names with stopwordsiso.
- Prepare housing data and compute target price from price-per-m2.
- Train an ExtraTrees regressor.
- Visualize model quality with matplotlib + seaborn.
- Export production-ready artifacts.

## 1. Project Structure

- `main.py`: 3-stage pipeline (`practice -> train -> test`).
- `data/VN_housing_dataset.csv`: Input dataset.
- `reports/model_evaluation_dashboard_2020.png`: Actual vs Predicted scatter chart.
- `production/housing_price_model_2020.pkl`: Trained model.
- `production/model_metadata.json`: Metadata (features, metrics, rename mapping).
- `production/test_predictions.csv`: Test-set predictions.
- `requirements.txt`: Python dependencies.

## 2. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Vietnamese Column Name Handling

The script uses `stopwordsiso` (`vi`) and applies:
- Accent removal.
- Alphanumeric tokenization.
- Stopword filtering, while preserving critical domain tokens (`ngay`, `dien`, `tich`, `gia`, `m2`, ...).
- Snake-case normalization.

Examples:
- `Thong tin Ngay dang` -> `ngay_dang`
- `Gia/m2 (trieu)` -> `gia_m2_trieu`

## 4. Three-Stage Pipeline

### Stage 1/3 - Practice

- Normalize Vietnamese column names.
- Detect required columns by keyword rules (`year`, `area`, `bed`, `floor`, `dist`, `type`).
- Compute `price = price_per_m2 * area`.
- Keep year 2020 records (if year column exists).
- Remove price outliers (`5e8` to `5e11`).
- Encode categorical columns (`dist`, `type`) with `OrdinalEncoder`.

### Stage 2/3 - Train

- Split data with `train_test_split(test_size=0.2, random_state=42)`.
- Train `ExtraTreesRegressor` on `log1p(price)`.

### Stage 3/3 - Test

- Predict and invert with `expm1`.
- Compute `MAPE`, `MAE`, and `R2`.
- Save one chart only: Actual vs Predicted scatter.
- Export production artifacts.

## 5. Run

```bash
python3 main.py
```

Console output includes:
- 3-stage progress logs.
- Train/test sizes.
- `Accuracy = 100 - MAPE`.
- `MAPE`, `MAE`, `R2`.
- Sample rows of actual vs predicted values.

## 6. Production Artifacts

After a successful run, `production` contains:
- `housing_price_model_2020.pkl`
- `model_metadata.json`
- `test_predictions.csv`

## 7. Notes

- `stopwordsiso` currently depends on `pkg_resources`, so `setuptools<81` is pinned in requirements.
- If your dataset schema changes significantly, update `column_rules` in `main.py`.
