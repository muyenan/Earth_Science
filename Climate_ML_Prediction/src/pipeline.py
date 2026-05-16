from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    FIGURE_DIR,
    GAS_LOOKBACK,
    OUTPUT_DIR,
    RANDOM_SEED,
    STRICT_TEST_YEARS,
    STRICT_VALIDATION_YEARS,
    TEMP_LOOKBACK,
)
from .data import build_datasets
from .features import (
    ENSO_COLUMNS,
    GAS_COLUMNS,
    fourier_time_features,
    gas_sequence_dataset,
    hybrid_temperature_features,
    random_decade_years,
    source_rows,
    temperature_dataset,
)
from .metrics import report
from .models import MLPRegressor
from .plots import bar_chart, line_chart, scatter_plot, workflow_diagram


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    gas_predictions: pd.DataFrame
    metrics: dict[str, float | str]


MODEL_COLUMNS = {
    "greenhouse_fourier_mlp": "greenhouse_fourier_temp_c",
    "greenhouse_fourier_enso_mlp": "greenhouse_fourier_enso_temp_c",
}


def _year_array(values: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
    return np.asarray(sorted({int(value) for value in values}), dtype=int)


def _available_years(merged: pd.DataFrame, use_enso: bool) -> np.ndarray:
    columns = ["temp_anomaly_c", *GAS_COLUMNS]
    if use_enso:
        columns.extend(ENSO_COLUMNS)
    frame = merged.dropna(subset=columns)
    return frame["year"].astype(int).to_numpy()


def _temperature_lookup(temperature: pd.DataFrame, years: np.ndarray) -> np.ndarray:
    lookup = temperature.set_index("year")["temp_anomaly_c"]
    return np.asarray([float(lookup.loc[int(year)]) for year in years], dtype=float)


def _fit_gas_model(greenhouse: pd.DataFrame, training_years: np.ndarray, unavailable_years: np.ndarray) -> MLPRegressor:
    unavailable = set(int(year) for year in unavailable_years)
    context = greenhouse[~greenhouse["year"].astype(int).isin(unavailable)].copy()
    target_years = _year_array([year for year in training_years if int(year) in set(context["year"].astype(int))])
    x_train, y_train, _ = gas_sequence_dataset(context, target_years=target_years)
    model = MLPRegressor(hidden_units=32, learning_rate=0.006, epochs=2200, l2=1e-5, random_state=RANDOM_SEED)
    return model.fit(x_train, y_train)


def _clip_gas_prediction(raw_prediction: np.ndarray, history: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_prediction, dtype=float).reshape(-1)
    if len(history) < 2:
        return raw
    recent = history[-min(len(history), 7) :]
    growth = np.nanmedian(np.diff(recent, axis=0), axis=0)
    growth = np.where(np.isfinite(growth), growth, 0.0)
    previous = recent[-1]
    lower = np.minimum(previous + 0.2 * growth, previous + 2.0 * growth)
    upper = np.maximum(previous + 0.2 * growth, previous + 2.0 * growth)
    return np.clip(raw, lower, upper)


def _predict_gas_recursive(
    model: MLPRegressor,
    initial_context: pd.DataFrame,
    target_years: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = initial_context[["year", *GAS_COLUMNS]].dropna().copy()
    context["year"] = context["year"].astype(int)
    rows: list[dict[str, float | int]] = []

    for year in sorted(int(value) for value in target_years):
        prior = context[context["year"] < year].sort_values("year")
        if len(prior) < GAS_LOOKBACK:
            raise ValueError(f"Not enough greenhouse history to predict {year}.")

        window = prior[GAS_COLUMNS].tail(GAS_LOOKBACK).to_numpy(dtype=float)
        features = np.concatenate([window.reshape(-1), fourier_time_features(year).reshape(-1)])
        raw_prediction = model.predict(features.reshape(1, -1)).reshape(-1)
        clipped = _clip_gas_prediction(raw_prediction, prior[GAS_COLUMNS].tail(7).to_numpy(dtype=float))

        row = {"year": year, **{column: float(value) for column, value in zip(GAS_COLUMNS, clipped)}}
        rows.append({f"{column}_predicted": value for column, value in row.items()})
        context = pd.concat([context, pd.DataFrame([row])], ignore_index=True)
        context = context.drop_duplicates(subset=["year"], keep="last").sort_values("year").reset_index(drop=True)

    gas_predictions = pd.DataFrame(rows)
    if not gas_predictions.empty:
        gas_predictions = gas_predictions.rename(columns={"year_predicted": "year"})
    return gas_predictions, context


def _recursive_baseline(temp_context: pd.DataFrame, target_years: np.ndarray) -> np.ndarray:
    context = temp_context[["year", "temp_anomaly_c"]].dropna().copy()
    context["year"] = context["year"].astype(int)
    predictions: list[float] = []
    for year in sorted(int(value) for value in target_years):
        prior = context[context["year"] < year].sort_values("year")
        if prior.empty:
            raise ValueError(f"Not enough temperature history to predict {year}.")
        prediction = float(prior.iloc[-1]["temp_anomaly_c"])
        predictions.append(prediction)
        context = pd.concat([context, pd.DataFrame([{"year": year, "temp_anomaly_c": prediction}])], ignore_index=True)
        context = context.drop_duplicates(subset=["year"], keep="last").sort_values("year").reset_index(drop=True)
    return np.asarray(predictions, dtype=float)


def _fit_temperature_model(
    model_name: str,
    training_years: np.ndarray,
    temp_context: pd.DataFrame,
    gas_context: pd.DataFrame,
    temperature: pd.DataFrame,
    enso_context: pd.DataFrame | None,
) -> MLPRegressor:
    x_train, y_train, _ = temperature_dataset(training_years, temp_context, gas_context, temperature, enso_context)
    hidden_units = 32 if enso_context is None else 48
    seed_offset = 0 if enso_context is None else 17
    model = MLPRegressor(
        hidden_units=hidden_units,
        learning_rate=0.006,
        epochs=2400,
        l2=1e-5,
        random_state=RANDOM_SEED + seed_offset,
    )
    return model.fit(x_train, y_train)


def _predict_temperature_recursive(
    model: MLPRegressor,
    temp_context: pd.DataFrame,
    gas_context: pd.DataFrame,
    target_years: np.ndarray,
    enso_context: pd.DataFrame | None,
) -> np.ndarray:
    context = temp_context[["year", "temp_anomaly_c"]].dropna().copy()
    context["year"] = context["year"].astype(int)
    predictions: list[float] = []

    for year in sorted(int(value) for value in target_years):
        features = hybrid_temperature_features(year, context, gas_context, enso_context)
        if features is None:
            raise ValueError(f"Could not build temperature features for {year}.")
        prediction = float(model.predict(features.reshape(1, -1))[0, 0])
        predictions.append(prediction)
        context = pd.concat([context, pd.DataFrame([{"year": year, "temp_anomaly_c": prediction}])], ignore_index=True)
        context = context.drop_duplicates(subset=["year"], keep="last").sort_values("year").reset_index(drop=True)

    return np.asarray(predictions, dtype=float)


def _run_temperature_forecast(
    model_name: str,
    split_name: str,
    temperature: pd.DataFrame,
    greenhouse: pd.DataFrame,
    merged: pd.DataFrame,
    training_years: np.ndarray,
    target_years: np.ndarray,
    use_enso: bool,
) -> ForecastResult:
    max_year = int(np.max(target_years))
    gas_model = _fit_gas_model(greenhouse, training_years, unavailable_years=target_years)
    gas_source = source_rows(greenhouse, GAS_COLUMNS, unavailable_years=target_years, max_year=max_year)
    gas_predictions, gas_context = _predict_gas_recursive(gas_model, gas_source, target_years)

    temp_source = source_rows(temperature, ["temp_anomaly_c"], unavailable_years=target_years, max_year=max_year)
    enso_context = None
    if use_enso:
        enso_context = merged[["year", *ENSO_COLUMNS]].dropna().copy()
        enso_context["year"] = enso_context["year"].astype(int)

    model = _fit_temperature_model(model_name, training_years, temp_source, gas_context, temperature, enso_context)
    predictions = _predict_temperature_recursive(model, temp_source, gas_context, target_years, enso_context)
    baseline = _recursive_baseline(temp_source, target_years)
    actual = _temperature_lookup(temperature, target_years)

    prediction_column = MODEL_COLUMNS[model_name]
    result = pd.DataFrame(
        {
            "year": target_years,
            "actual_temp_c": actual,
            "baseline_temp_c": baseline,
            prediction_column: predictions,
        }
    )
    result[f"{prediction_column}_error_c"] = result[prediction_column] - result["actual_temp_c"]
    result[f"{prediction_column}_abs_error_c"] = result[f"{prediction_column}_error_c"].abs()

    metric = report(model_name, split_name, actual, predictions, baseline)
    return ForecastResult(result, gas_predictions.assign(split=split_name, model=model_name), metric)


def _combine_forecasts(left: ForecastResult, right: ForecastResult) -> pd.DataFrame:
    left_frame = left.predictions.copy()
    right_column = [column for column in right.predictions.columns if column.endswith("_temp_c") and column not in {"actual_temp_c", "baseline_temp_c"}][0]
    right_frame = right.predictions[["year", right_column, f"{right_column}_error_c", f"{right_column}_abs_error_c"]]
    return left_frame.merge(right_frame, on="year", how="left")


def _add_baseline_errors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["baseline_error_c"] = result["baseline_temp_c"] - result["actual_temp_c"]
    result["baseline_abs_error_c"] = result["baseline_error_c"].abs()
    return result


def _run_split(
    split_name: str,
    temperature: pd.DataFrame,
    greenhouse: pd.DataFrame,
    merged: pd.DataFrame,
    training_years: np.ndarray,
    target_years: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], pd.DataFrame]:
    base_result = _run_temperature_forecast(
        "greenhouse_fourier_mlp",
        split_name,
        temperature,
        greenhouse,
        merged,
        training_years,
        target_years,
        use_enso=False,
    )
    enso_result = _run_temperature_forecast(
        "greenhouse_fourier_enso_mlp",
        split_name,
        temperature,
        greenhouse,
        merged,
        training_years,
        target_years,
        use_enso=True,
    )
    predictions = _add_baseline_errors(_combine_forecasts(base_result, enso_result))
    gas_predictions = pd.concat([base_result.gas_predictions, enso_result.gas_predictions], ignore_index=True)
    return predictions, [base_result.metrics, enso_result.metrics], gas_predictions


def _random_decade_split(greenhouse: pd.DataFrame, temperature: pd.DataFrame, merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    common_years = _available_years(merged, use_enso=True)
    first_gas_year = int(greenhouse["year"].min())
    first_temp_year = int(temperature["year"].min())
    common_years = common_years[common_years >= max(first_gas_year + GAS_LOOKBACK, first_temp_year + TEMP_LOOKBACK)]
    test_years = random_decade_years(common_years, random_state=RANDOM_SEED)
    training_years = np.asarray([year for year in common_years if int(year) not in set(test_years)], dtype=int)
    return training_years, test_years


def _write_outputs(
    validation_predictions: pd.DataFrame,
    strict_predictions: pd.DataFrame,
    random_predictions: pd.DataFrame,
    gas_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    validation_predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    strict_predictions.to_csv(OUTPUT_DIR / "strict_future_predictions.csv", index=False)
    random_predictions.to_csv(OUTPUT_DIR / "random_decade_predictions.csv", index=False)
    gas_predictions.to_csv(OUTPUT_DIR / "greenhouse_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)


def _draw_figures(validation_predictions: pd.DataFrame, strict_predictions: pd.DataFrame, random_predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    workflow_diagram(FIGURE_DIR / "model_workflow.png")
    line_chart(
        FIGURE_DIR / "validation_forecast.png",
        "Validation Forecast: 2019-2021",
        validation_predictions,
        [
            ("actual_temp_c", "Actual", "#222222"),
            ("baseline_temp_c", "Persistence baseline", "#8c8c8c"),
            ("greenhouse_fourier_temp_c", "Greenhouse + Fourier", "#2f6db3"),
            ("greenhouse_fourier_enso_temp_c", "With lagged ENSO", "#c44e52"),
        ],
    )
    line_chart(
        FIGURE_DIR / "strict_future_forecast.png",
        "Strict Future Forecast: 2022-2025",
        strict_predictions,
        [
            ("actual_temp_c", "Actual", "#222222"),
            ("baseline_temp_c", "Persistence baseline", "#8c8c8c"),
            ("greenhouse_fourier_temp_c", "Greenhouse + Fourier", "#2f6db3"),
            ("greenhouse_fourier_enso_temp_c", "With lagged ENSO", "#c44e52"),
        ],
    )
    scatter_plot(
        FIGURE_DIR / "random_decade_scatter.png",
        "Random Decade Holdout",
        random_predictions,
        "actual_temp_c",
        "greenhouse_fourier_enso_temp_c",
    )

    labels = {
        "greenhouse_fourier_mlp": "Fourier",
        "greenhouse_fourier_enso_mlp": "Fourier+ENSO",
    }
    rmse_frame = metrics.copy()
    rmse_frame["label"] = rmse_frame["model"].map(labels) + " " + rmse_frame["split"].str.replace(" ", "_", regex=False)
    bar_chart(FIGURE_DIR / "rmse_summary.png", "RMSE Summary", rmse_frame, "label", "rmse")


def main() -> None:
    temperature, greenhouse, _, merged = build_datasets()

    validation_years = _year_array(STRICT_VALIDATION_YEARS)
    strict_test_years = _year_array(STRICT_TEST_YEARS)
    validation_training = _available_years(merged, use_enso=True)
    validation_training = validation_training[validation_training < int(validation_years.min())]
    strict_training = _available_years(merged, use_enso=True)
    strict_training = strict_training[strict_training < int(strict_test_years.min())]
    random_training, random_test = _random_decade_split(greenhouse, temperature, merged)

    validation_predictions, validation_metrics, validation_gas = _run_split(
        "validation",
        temperature,
        greenhouse,
        merged,
        validation_training,
        validation_years,
    )
    strict_predictions, strict_metrics, strict_gas = _run_split(
        "strict future",
        temperature,
        greenhouse,
        merged,
        strict_training,
        strict_test_years,
    )
    random_predictions, random_metrics, random_gas = _run_split(
        "random decade",
        temperature,
        greenhouse,
        merged,
        random_training,
        random_test,
    )

    gas_predictions = pd.concat([validation_gas, strict_gas, random_gas], ignore_index=True)
    metrics = pd.DataFrame([*validation_metrics, *strict_metrics, *random_metrics])
    _write_outputs(validation_predictions, strict_predictions, random_predictions, gas_predictions, metrics)
    _draw_figures(validation_predictions, strict_predictions, random_predictions, metrics)

    print(f"Wrote outputs to {OUTPUT_DIR}")

