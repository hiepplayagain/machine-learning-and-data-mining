"""Train and forecast Hanoi house prices from Vietnamese CSV data."""

import pandas as pd
from math import expm1, log1p
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "price"
RANDOM_STATE = 42
DATA_FILE = "data/VN_housing_dataset.csv"
FUTURE_YEAR = 2028
MAX_TRAIN_ROWS = 3000
COLUMN_ALIASES = {
	"year": ["Ngay", "Date"],
	"area_m2": ["Dien tich", "area", "area m2"],
	"bedrooms": ["So phong ngu", "bedrooms"],
	"floors": ["So tang", "floors"],
	"district": ["Quan", "district"],
	"property_type": ["Loai hinh nha o", "property type"],
	"legal_status": ["Giay to phap ly", "legal"],
}


def normalize_name(text: str) -> str:
	"""Normalize column text to a stable token for alias matching.

	The function removes accents, lowercases text, and converts common punctuation
	to spaces so headers like 'Diện tích', 'Dien tich', and 'dien_tich' can be
	resolved consistently.
	"""
	normalized_text = pd.Series([str(text)], dtype="string").str.normalize("NFKD")
	normalized_text = normalized_text.str.encode("ascii", "ignore").str.decode("utf-8")
	normalized_text = normalized_text.str.lower().str.strip().iloc[0]
	for punctuation in ["/", "-", "_", ",", ".", "(", ")"]:
		normalized_text = normalized_text.replace(punctuation, " ")
	return " ".join(normalized_text.split())


def find_column(columns: list[str], aliases: list[str]) -> str | None:
	"""Return the first real column that matches any alias.

	Args:
		columns: Raw column names from the CSV file.
		aliases: Candidate names for one logical feature.

	Returns:
		Matched original column name or None if no alias exists.
	"""
	normalized_columns = {normalize_name(column): column for column in columns}
	for alias in aliases:
		matched_column = normalized_columns.get(normalize_name(alias))
		if matched_column:
			return matched_column
	return None


def convert_to_number_series(series: pd.Series) -> pd.Series:
	"""Extract numeric values from mixed text and return float series.

	Examples:
		'46 m2' -> 46.0
		'86,96 trieu/m2' -> 86.96
	"""
	cleaned_values = series.astype("string").str.lower().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
	return pd.to_numeric(cleaned_values.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce")


def prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
	"""Build a model-ready dataframe from raw housing data.

	Main steps:
	1. Remove unnamed helper columns from CSV exports.
	2. Resolve feature columns by Vietnamese/English aliases.
	3. Convert numeric-like text fields into numeric values.
	4. Construct target `price` from area and price-per-square-meter when needed.
	5. Add a few helpful engineered features.
	"""
	data = raw_data.drop(columns=[column for column in raw_data.columns if normalize_name(column).startswith("unnamed")], errors="ignore").copy()
	resolved_columns = {feature_name: find_column(data.columns.tolist(), aliases) for feature_name, aliases in COLUMN_ALIASES.items()}

	for feature_name, source_name in resolved_columns.items():
		if not source_name:
			continue
		if feature_name == "year":
			data[feature_name] = pd.to_datetime(data[source_name], errors="coerce").dt.year
		elif feature_name in {"area_m2", "bedrooms", "floors"}:
			data[feature_name] = convert_to_number_series(data[source_name])
		else:
			data[feature_name] = data[source_name].astype("string").fillna("").astype(str)

	price_per_square_meter_column = find_column(data.columns.tolist(), ["Gia/m2", "price/m2", "gia m2"])
	if TARGET_COLUMN not in data.columns and price_per_square_meter_column and "area_m2" in data.columns:
		data[TARGET_COLUMN] = convert_to_number_series(data[price_per_square_meter_column]) * 1_000_000 * data["area_m2"]

	# Basic interaction feature that usually helps tree-based models.
	if {"area_m2", "bedrooms"}.issubset(data.columns):
		data["area_per_bedroom"] = data["area_m2"] / data["bedrooms"].replace(0, pd.NA)

	# Combine location and property type to provide richer categorical signal.
	if {"district", "property_type"}.issubset(data.columns):
		data["district_property_type"] = data["district"].str.strip() + "|" + data["property_type"].str.strip()

	selected_columns = [column for column in [*COLUMN_ALIASES, "area_per_bedroom", "district_property_type", TARGET_COLUMN] if column in data.columns]
	prepared_data = data[selected_columns]
	if TARGET_COLUMN in prepared_data.columns:
		prepared_data = prepared_data.dropna(subset=[TARGET_COLUMN])
	return prepared_data.dropna(axis=0, how="all")


def load_data(file_path: str) -> pd.DataFrame:
	"""Load CSV, preprocess data, and validate minimum training requirements."""
	try:
		raw_data = pd.read_csv(file_path)
	except FileNotFoundError as exception:
		raise FileNotFoundError(f"Data file not found: {file_path}. Create the CSV as documented in README.md") from exception

	prepared_data = prepare_data(raw_data)
	if TARGET_COLUMN not in prepared_data.columns:
		raise ValueError("Cannot build target 'price'. Need either a price column or both Dien tich and Gia/m2 columns.")
	if prepared_data.shape[0] < 30:
		raise ValueError("Dataset should have at least 30 rows for basic training")
	if len([column for column in prepared_data.columns if column != TARGET_COLUMN]) == 0:
		raise ValueError("No valid feature columns found after preprocessing")
	return prepared_data


def build_model(feature_data: pd.DataFrame) -> Pipeline:
	"""Create preprocessing + regressor pipeline.

	- Numeric columns: median imputation.
	- Categorical columns: mode imputation + one-hot encoding.
	- Regressor: ExtraTrees with log-transform target wrapper.
	"""
	numeric_columns = feature_data.select_dtypes(include=["number"]).columns.tolist()
	categorical_columns = [column for column in feature_data.columns if column not in numeric_columns]

	preprocessor = ColumnTransformer(
		transformers=[
			("numeric", SimpleImputer(strategy="median"), numeric_columns),
			(
				"categorical",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("one_hot", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				categorical_columns,
			),
		]
	)

	return Pipeline(
		steps=[
			("preprocess", preprocessor),
			(
				"regressor",
				TransformedTargetRegressor(
					regressor=ExtraTreesRegressor(
						random_state=RANDOM_STATE,
						n_estimators=160,
						max_features=0.7,
						n_jobs=-1,
					),
					func=lambda values: pd.DataFrame(values).astype(float).apply(lambda column: column.map(log1p)).to_numpy(),
					inverse_func=lambda values: pd.DataFrame(values).astype(float).apply(lambda column: column.map(expm1)).to_numpy(),
				),
			),
		]
	)


def train_model(prepared_data: pd.DataFrame) -> tuple[Pipeline, float]:
	"""Train/test split, fit model, and compute accuracy percent from MAPE.

	Accuracy formula used:
		accuracy = max(0, 100 - MAPE)
	"""
	features = prepared_data.drop(columns=[TARGET_COLUMN])
	target = prepared_data[TARGET_COLUMN]
	training_features, testing_features, training_target, testing_target = train_test_split(
		features, target, test_size=0.2, random_state=RANDOM_STATE
	)

	model = build_model(training_features)
	model.fit(training_features, training_target)
	predicted_target = model.predict(testing_features)
	valid_target_mask = testing_target != 0
	accuracy_percent = (
		max(
			0.0,
			100.0
			- float(
				((testing_target[valid_target_mask] - predicted_target[valid_target_mask]).abs() / testing_target[valid_target_mask]).mean() * 100
			),
		)
		if valid_target_mask.any()
		else float("nan")
	)
	return model, accuracy_percent


def main() -> None:
	"""Run full flow: load data, train model, forecast future price, print output."""
	prepared_data = load_data(DATA_FILE)
	# Keep runtime stable by training on a capped sample when dataset is large.
	if len(prepared_data) > MAX_TRAIN_ROWS:
		prepared_data = prepared_data.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

	model, accuracy_percent = train_model(prepared_data)
	future_sample = prepared_data.drop(columns=[TARGET_COLUMN]).iloc[[0]].copy()
	if "year" in future_sample.columns:
		future_sample["year"] = FUTURE_YEAR
	future_price = model.predict(future_sample)[0]

	print("House price prediction")
	print(f"Year: {FUTURE_YEAR}")
	print(f"Predicted price (VND): {future_price:,.0f}")
	print(f"Accuracy (%): {accuracy_percent:.2f}")


if __name__ == "__main__":
	main()
