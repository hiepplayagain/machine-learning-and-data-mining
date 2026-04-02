"""Train and use a house price prediction model for Hanoi.

This script supports Vietnamese CSV schemas such as:
- Ngay, Quan, Huyen, Loai hinh nha o, Giay to phap ly
- So tang, So phong ngu, Dien tich, Gia/m2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "price"
RANDOM_STATE = 42


@dataclass
class TrainArtifacts:
	model: Pipeline
	metrics: dict[str, float]
	train_size: int
	test_size: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Hanoi house price prediction with machine learning"
	)
	parser.add_argument(
		"--data",
		type=Path,
		default=Path("data/hanoi_house_prices.csv"),
		help="Path to training CSV file",
	)
	parser.add_argument(
		"--future-year",
		type=int,
		default=2027,
		help="Year to create an example future prediction",
	)
	parser.add_argument(
		"--report-out",
		type=Path,
		default=Path("reports/train_report.json"),
		help="Path to save the training report JSON",
	)
	parser.add_argument(
		"--chart-out",
		type=Path,
		default=Path("reports/train_report.png"),
		help="Path to save the visualization chart PNG",
	)
	parser.add_argument(
		"--no-show-chart",
		action="store_true",
		help="Disable showing chart window on screen",
	)
	return parser.parse_args()


def normalize_name(text: str) -> str:
	value = unicodedata.normalize("NFKD", str(text))
	value = value.encode("ascii", "ignore").decode("ascii")
	value = value.lower().strip()
	value = re.sub(r"[^a-z0-9]+", " ", value)
	return value.strip()


def find_column(columns: list[str], aliases: list[str]) -> str | None:
	normalized = {normalize_name(c): c for c in columns}
	for alias in aliases:
		match = normalized.get(normalize_name(alias))
		if match is not None:
			return match
	return None


def extract_number(value: object) -> float | None:
	if pd.isna(value):
		return None
	text = str(value).strip().lower().replace(" ", "")
	# Convert decimal comma style (86,96) to dot style (86.96)
	text = text.replace(",", ".")
	match = re.search(r"-?\d+(?:\.\d+)?", text)
	if not match:
		return None
	return float(match.group(0))


def to_numeric_series(series: pd.Series) -> pd.Series:
	return series.apply(extract_number).astype(float)


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
	df = raw_df.copy()

	# Drop auto index columns created by CSV export tools
	drop_cols = [c for c in df.columns if normalize_name(c).startswith("unnamed")]
	if drop_cols:
		df = df.drop(columns=drop_cols)

	date_col = find_column(df.columns.tolist(), ["Ngay", "Date"])
	if date_col:
		df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year

	area_col = find_column(df.columns.tolist(), ["Dien tich", "area", "area m2"])
	if area_col:
		df["area_m2"] = to_numeric_series(df[area_col])

	bedroom_col = find_column(df.columns.tolist(), ["So phong ngu", "bedrooms"])
	if bedroom_col:
		df["bedrooms"] = to_numeric_series(df[bedroom_col])

	floor_col = find_column(df.columns.tolist(), ["So tang", "floors"])
	if floor_col:
		df["floors"] = to_numeric_series(df[floor_col])

	district_col = find_column(df.columns.tolist(), ["Quan", "district"])
	if district_col:
		df["district"] = df[district_col].astype(str)

	type_col = find_column(df.columns.tolist(), ["Loai hinh nha o", "property type"])
	if type_col:
		df["property_type"] = df[type_col].astype(str)

	legal_col = find_column(df.columns.tolist(), ["Giay to phap ly", "legal"])
	if legal_col:
		df["legal_status"] = df[legal_col].astype(str)

	# Build target price from Gia/m2 and Dien tich when direct price is not available.
	if TARGET_COLUMN not in df.columns:
		price_m2_col = find_column(df.columns.tolist(), ["Gia/m2", "price/m2", "gia m2"])
		if price_m2_col and "area_m2" in df.columns:
			price_m2_million = to_numeric_series(df[price_m2_col])
			df[TARGET_COLUMN] = price_m2_million * 1_000_000 * df["area_m2"]

	feature_candidates = [
		"year",
		"area_m2",
		"bedrooms",
		"floors",
		"district",
		"property_type",
		"legal_status",
	]
	available_features = [c for c in feature_candidates if c in df.columns]
	if TARGET_COLUMN in df.columns:
		available_features.append(TARGET_COLUMN)

	prepared = df[available_features].dropna(subset=[TARGET_COLUMN])
	prepared = prepared.dropna(axis=0, how="all")
	return prepared


def load_data(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(
			f"Data file not found: {csv_path}. Create the CSV as documented in README.md"
		)

	raw_df = pd.read_csv(csv_path)
	df = prepare_dataframe(raw_df)

	if TARGET_COLUMN not in df.columns:
		raise ValueError(
			"Cannot build target 'price'. Need either a price column or both Dien tich and Gia/m2 columns."
		)
	if df.shape[0] < 30:
		raise ValueError("Dataset should have at least 30 rows for basic training")
	if df.drop(columns=[TARGET_COLUMN]).shape[1] == 0:
		raise ValueError("No valid feature columns found after preprocessing")
	return df


def build_model(feature_df: pd.DataFrame) -> Pipeline:
	numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
	categorical_features = [
		c for c in feature_df.columns if c not in numeric_features
	]

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_features),
			("cat", categorical_pipeline, categorical_features),
		]
	)

	model = Pipeline(
		steps=[
			("preprocess", preprocessor),
			("regressor", LinearRegression()),
		]
	)
	return model


def train_and_evaluate(df: pd.DataFrame) -> TrainArtifacts:
	x = df.drop(columns=[TARGET_COLUMN])
	y = df[TARGET_COLUMN]

	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.2, random_state=RANDOM_STATE
	)

	model = build_model(x_train)
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	rmse = math.sqrt(mse)
	r2 = r2_score(y_test, y_pred)
	# Guard MAPE when true values include zeros.
	non_zero_mask = y_test != 0
	if non_zero_mask.any():
		mape = (
			((y_test[non_zero_mask] - y_pred[non_zero_mask]).abs() / y_test[non_zero_mask])
			.mean()
			* 100
		)
	else:
		mape = float("nan")

	metrics = {
		"mae": float(mae),
		"rmse": float(rmse),
		"r2": float(r2),
		"mape_percent": float(mape),
	}

	return TrainArtifacts(
		model=model,
		metrics=metrics,
		train_size=len(x_train),
		test_size=len(x_test),
	)


def export_report(
	report_path: Path,
	artifacts: TrainArtifacts,
	row_count: int,
	feature_count: int,
	future_year: int,
	future_price: float,
) -> Path:
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report = {
		"dataset": {
			"rows": int(row_count),
			"features": int(feature_count),
			"train_size": int(artifacts.train_size),
			"test_size": int(artifacts.test_size),
		},
		"metrics": artifacts.metrics,
		"forecast": {
			"year": int(future_year),
			"predicted_price_vnd": float(future_price),
		},
	}
	with report_path.open("w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	return report_path


def export_visual_report(
	chart_path: Path,
	artifacts: TrainArtifacts,
	future_year: int,
	future_price: float,
	show_chart: bool,
) -> Path:
	chart_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	metric_names = ["MAE", "RMSE", "MAPE %"]
	metric_values = [
		artifacts.metrics["mae"] / 1_000_000_000,
		artifacts.metrics["rmse"] / 1_000_000_000,
		artifacts.metrics["mape_percent"],
	]
	colors = ["#4e79a7", "#f28e2b", "#59a14f"]
	axes[0].bar(metric_names, metric_values, color=colors)
	axes[0].set_title("Model Metrics")
	axes[0].set_ylabel("Value (Billion VND for MAE/RMSE)")

	axes[1].bar([str(future_year)], [future_price / 1_000_000_000], color="#e15759")
	axes[1].set_title("Forecast Price")
	axes[1].set_ylabel("Billion VND")

	fig.suptitle("Hanoi House Price Training Report", fontsize=13)
	fig.tight_layout()
	fig.savefig(chart_path, dpi=150)

	if show_chart:
		try:
			plt.show()
		except Exception as exc:
			print(f"Warning: cannot show chart window: {exc}")

	plt.close(fig)
	return chart_path


def make_future_sample(df: pd.DataFrame, future_year: int) -> pd.DataFrame:
	sample = df.drop(columns=[TARGET_COLUMN]).iloc[[0]].copy()
	if "year" in sample.columns:
		sample["year"] = future_year
	return sample


def main() -> None:
	args = parse_args()
	data = load_data(args.data)

	artifacts = train_and_evaluate(data)
	feature_count = data.drop(columns=[TARGET_COLUMN]).shape[1]
	print("Model training completed")
	print(f"Rows: {len(data)} | Features: {feature_count}")
	print(f"Train size: {artifacts.train_size} | Test size: {artifacts.test_size}")
	print(f"MAE (VND): {artifacts.metrics['mae']:,.0f}")
	print(f"RMSE (VND): {artifacts.metrics['rmse']:,.0f}")
	print(f"R2: {artifacts.metrics['r2']:.4f}")
	print(f"MAPE (%): {artifacts.metrics['mape_percent']:.2f}")

	future_sample = make_future_sample(data, args.future_year)
	future_price = artifacts.model.predict(future_sample)[0]
	print("\\nExample forecast")
	print(f"Year: {args.future_year}")
	print(f"Predicted price (VND): {future_price:,.0f}")

	report_path = export_report(
		report_path=args.report_out,
		artifacts=artifacts,
		row_count=len(data),
		feature_count=feature_count,
		future_year=args.future_year,
		future_price=float(future_price),
	)
	print(f"Report saved: {report_path}")

	chart_path = export_visual_report(
		chart_path=args.chart_out,
		artifacts=artifacts,
		future_year=args.future_year,
		future_price=float(future_price),
		show_chart=not args.no_show_chart,
	)
	print(f"Chart saved: {chart_path}")


if __name__ == "__main__":
	main()