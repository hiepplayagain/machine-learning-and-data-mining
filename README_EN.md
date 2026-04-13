# Machine Learning Project: Hanoi House Price Prediction

Vietnamese version: [README.md](README.md)

Project goals:
- Predict house prices from Hanoi real-estate features.
- Produce forecasts for future years by default from 2025 to 2027.

## 1. Project Structure

- `main.py`: Data preprocessing, LightGBM model training, and future-year forecasting.
- `data/VN_housing_dataset.csv`: Input dataset.

## 2. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2.1 Push Policy (Source Code Only)

- Do not push the virtual environment folder `.venv/`.
- Do not push generated Python artifacts (`__pycache__/`, `*.pyc`).
- This repository should contain source code, required data, reports, and documentation only.

The `.gitignore` file is already configured to ignore those environment and cache files.

If someone forks or clones this repository, they should install dependencies locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Input Data Format

The script supports Vietnamese column names such as:
- Ngay
- Dia chi
- Quan
- Huyen
- Loai hinh nha o
- Giay to phap ly
- So tang
- So phong ngu
- Dien tich (example: 46 m2)
- Gia/m2 (example: 86,96 trieu/m2)

Automatic preprocessing includes:
- Extracting numeric values from text fields (for example area and bedrooms)
- Creating year from Ngay
- Creating `area_per_bed` (area per bedroom)
- Creating `rel_area_dist` (relative area compared to district average)
- Building the target price with:

$$
price = Dien\ tich \times Gia/m2 \times 1{,}000{,}000
$$

## 4. Run Training + Forecast + Report

```bash
python3 main.py
```

The current model is a LightGBM regressor trained on log-price targets.

The script prints:
- The training year range found in the dataset.
- Accuracy computed as `100 - MAPE` from 3-fold cross-validation.
- Forecasts for 2025, 2026, and 2027 in both VND and billion VND.

To change the default forecast horizon, edit `FORECAST_YEARS` in `main.py`.

## 4.1 How LightGBM Computes Predictions in This Project

In this codebase, the model learns:

$$
f(\mathbf{x}) \approx \log(1 + price)
$$

where $\mathbf{x}$ is the preprocessed feature vector (imputed numeric/categorical features, scaled numeric columns, and one-hot encoded categories).

Prediction is computed in 5 steps:

1. Transform raw input into model features
- Numeric features (`area_m2`, `bedrooms`, `floors`, `year`, ...): missing values are filled with median, then standardized.
- Categorical features (`district`, `property_type`, `legal_status`, ...): missing values are filled with most frequent category, then one-hot encoded.

2. Additive tree ensemble on log-price
- LightGBM builds a sequence of CART trees.
- At boosting round $t$, a new tree $h_t(\mathbf{x})$ is fit to reduce the current residual/gradient.
- The log-price prediction is the weighted sum of trees:

$$
\hat{y}_{log}(\mathbf{x}) = \sum_{t=1}^{T} \eta \cdot h_t(\mathbf{x})
$$

where $\eta$ is `learning_rate` and $T$ is the number of trees actually used (possibly less than `n_estimators` due to early stopping).

3. Optimal split search by gain
- Each tree chooses splits that maximize loss reduction (gain).
- LightGBM uses histogram binning, which makes split search faster and memory-efficient on larger datasets.

4. Early stopping for generalization
- The code holds out 15% validation data.
- With `early_stopping(200)`, training stops if validation does not improve for 200 rounds.

5. Convert prediction back to VND
- Model output is $\hat{y}_{log}$.
- The value is clamped by the minimum log-price seen in training to avoid unrealistic negative prices after inversion.
- Final prediction is:

$$
\hat{price} = \exp(\hat{y}_{log}) - 1
$$

After obtaining `price_base` at the latest year in the dataset, the script extrapolates future years using compound growth:

$$
price_{future} = price_{base} \cdot (1 + r)^{\Delta year}
$$

with $r = 0.09$ in the current configuration.

## 5. Outputs

After running, you get:
- Console output with dataset/feature summary and a sample future forecast.
