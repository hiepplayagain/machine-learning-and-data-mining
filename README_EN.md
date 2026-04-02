# Machine Learning Project: Hanoi House Price Prediction

Vietnamese version: [README.md](README.md)

Project goals:
- Predict house prices from Hanoi real-estate features.
- Produce forecasts for future years (for example 2027, 2028).

## 1. Project Structure

- main.py: Training, evaluation, report export, and chart visualization.
- data/VN_housing_dataset.csv: Input dataset.
- reports/train_report.json: Structured evaluation report.
- reports/train_report.png: Visualization chart.

## 2. Environment Setup

```bash
sudo apt update
sudo apt install -y python3.12-venv

cd "/home/thehiep/Machine Learning and Data Mining I"
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
- Building the target price with:

$$
price = Dien\ tich \times Gia/m2 \times 1{,}000{,}000
$$

## 4. Run Training + Forecast + Report

```bash
python3 main.py --data data/VN_housing_dataset.csv --future-year 2028 --report-out reports/train_report.json --chart-out reports/train_report.png
```

If you only want to save chart files and do not want a pop-up window:

```bash
python3 main.py --data data/VN_housing_dataset.csv --future-year 2028 --no-show-chart
```

## 5. Metrics Glossary (Abbreviations)

- MAE: Mean Absolute Error
Definition: Average absolute difference between predicted and actual values.
Interpretation: Lower is better.
Unit: VND in this project.

- RMSE: Root Mean Squared Error
Definition: Square root of average squared prediction error.
Interpretation: Lower is better; penalizes large errors more than MAE.
Unit: VND in this project.

- R2: Coefficient of Determination
Definition: Fraction of target variance explained by the model.
Range: Typically $(-\infty, 1]$.
Interpretation: Closer to 1 is better.

- MAPE: Mean Absolute Percentage Error
Definition: Average absolute percentage error.
Interpretation: Lower is better.
Unit: Percent (%).

Note: If you wrote "RMAE", the common metric name is usually "RMSE".

## 6. Outputs

After running, you get:
- Console output with MAE, RMSE, R2, MAPE, and a sample future forecast.
- JSON report at reports/train_report.json.
- Chart image at reports/train_report.png.
