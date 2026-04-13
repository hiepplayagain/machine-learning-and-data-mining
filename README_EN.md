# Machine Learning Project: Vietnam House Price Prediction

Vietnamese version: [README.md](README.md)

Goals:
- Preprocess real-estate data from CSV.
- Train a regression model to estimate house prices.
- Evaluate quickly on a held-out test split using MAPE.

## 1. Project Structure

- `main.py`: Data preprocessing, feature engineering, model training, and console reporting.
- `data/VN_housing_dataset.csv`: Input dataset.
- `requirements.txt`: Python dependencies.
- `train_report.json`, `reports/train_report.json`: Report files (if generated).

## 2. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Input Data and Supported Columns

The script maps input columns using aliases in `main.py`:
- `Ngay` or `year` -> `year`
- `Dien tich` or `area` -> `area_m2`
- `So phong ngu` or `bedrooms` -> `bedrooms`
- `So tang` or `floors` -> `floors`
- `Quan` or `district` -> `district`
- `Loai hinh nha o` or `property_type` -> `property_type`

Target `price` is computed from the `Gia/m2` column multiplied by `area_m2`.

The numeric parser (`clean_numeric_string`) supports values such as:
- `86,96 trieu/m2`
- `2.5 ty`
- `46 m2`

Unit conversion rules:
- `trieu` -> 1,000,000
- `ty` -> 1,000,000,000

## 4. Preprocessing and Feature Engineering

Inside `prepare_advanced`, the pipeline does:
- Extract required columns through alias matching.
- Convert date to year.
- Drop rows missing required fields: `price`, `year`, `area_m2`.
- Create `m2_per_bed = area_m2 / bedrooms` (replace 0 bedrooms by 1 to avoid division by zero).
- Create `district_rank` from district-wise mean of `log1p(price)`.
- Remove outliers for `price` and `area_m2` using 5th and 95th percentiles.

## 5. Model Training

The script splits data by year:
- Train: `year < 2020`
- Test: `year == 2020`

Model features:
- `area_m2`
- `bedrooms`
- `floors`
- `m2_per_bed`
- `district_rank`

Model pipeline:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `ExtraTreesRegressor` with:
	- `n_estimators=1000`
	- `max_depth=40`
	- `min_samples_split=2`
	- `random_state=42`
	- `n_jobs=-1`

Training target transformation:
- Train on `log1p(price)`
- Invert predictions with `expm1`

## 6. Run

```bash
python3 main.py
```

Console output includes:
- Training status.
- Accuracy computed as `100 - MAPE`.
- Processing time.
- A small sample of real vs predicted prices (in billion VND).

## 7. Notes

- If year 2020 is missing in your dataset, adjust the train/test split in `main.py`.
- Cleaner and larger datasets usually reduce MAPE.
- Current code focuses on evaluation on year 2020 and does not perform future-year forecasting.
