from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FOURIER_PERIODS, GAS_LOOKBACK, RANDOM_SEED, TEMP_LOOKBACK
from .data import forcing_proxy_from_values


GAS_COLUMNS = ["co2_ppm", "ch4_ppb", "n2o_ppb"]
ENSO_COLUMNS = [
    "nino34_mean_lag1",
    "nino34_abs_mean_lag1",
    "nino34_max_lag1",
    "nino34_min_lag1",
    "nino34_djf_lag1",
    "nino34_ndj_lag1",
    "nino34_mean_lag2",
    "nino34_djf_lag2",
    "nino34_change_lag1",
]


def fourier_time_features(year: int | np.ndarray, periods: tuple[int, ...] = FOURIER_PERIODS, origin_year: int = 1880) -> np.ndarray:
    years = np.asarray(year, dtype=float)
    if years.ndim == 0:
        years = years.reshape(1)
    centered = years - float(origin_year)
    features = [centered / 100.0]
    for period in periods:
        angle = 2.0 * np.pi * centered / period
        features.append(np.sin(angle))
        features.append(np.cos(angle))
    return np.vstack(features).T


def causal_fft_features(values: np.ndarray, keep: int = 3) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if len(series) == 0:
        return np.zeros(3 + keep * 3, dtype=float)
    scale = series.std()
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    spectrum = np.fft.rfft(series - series.mean())
    features = [float(series.mean()), float(series.std()), float(series[-1] - series[0]) if len(series) > 1 else 0.0]
    for index in range(1, keep + 1):
        if index < len(spectrum):
            coefficient = spectrum[index] / (len(series) * scale)
            features.extend([float(abs(coefficient)), float(coefficient.real), float(coefficient.imag)])
        else:
            features.extend([0.0, 0.0, 0.0])
    return np.asarray(features, dtype=float)


def random_decade_years(years: np.ndarray, random_state: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    for decade in sorted(set((years // 10) * 10)):
        decade_years = years[(years // 10) * 10 == decade]
        if len(decade_years) < 3:
            continue
        sample_size = min(int(rng.integers(1, 3)), max(1, len(decade_years) - 2))
        selected.extend(rng.choice(decade_years, size=sample_size, replace=False).astype(int).tolist())
    return np.asarray(sorted(set(selected)), dtype=int)


def gas_sequence_dataset(greenhouse: pd.DataFrame, target_years: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = greenhouse[["year", *GAS_COLUMNS]].dropna().sort_values("year").reset_index(drop=True)
    years = clean["year"].to_numpy(dtype=int)
    values = clean[GAS_COLUMNS].to_numpy(dtype=float)
    target_set = None if target_years is None else set(int(year) for year in target_years)
    x_rows = []
    y_rows = []
    kept_years = []
    for index in range(GAS_LOOKBACK, len(clean)):
        year = int(years[index])
        if target_set is not None and year not in target_set:
            continue
        window = values[index - GAS_LOOKBACK : index].reshape(-1)
        x_rows.append(np.concatenate([window, fourier_time_features(year).reshape(-1)]))
        y_rows.append(values[index])
        kept_years.append(year)
    return np.vstack(x_rows), np.vstack(y_rows), np.asarray(kept_years, dtype=int)


def gas_stats(gas_context: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = gas_context[GAS_COLUMNS].to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def source_rows(frame: pd.DataFrame, columns: list[str], unavailable_years: np.ndarray, max_year: int) -> pd.DataFrame:
    unavailable = set(int(year) for year in unavailable_years)
    result = frame[["year", *columns]].dropna().copy()
    result["year"] = result["year"].astype(int)
    result = result[(result["year"] <= int(max_year)) & (~result["year"].isin(unavailable))]
    return result.sort_values("year").reset_index(drop=True)


def hybrid_temperature_features(
    year: int,
    temp_context: pd.DataFrame,
    gas_context: pd.DataFrame,
    enso_context: pd.DataFrame | None = None,
) -> np.ndarray | None:
    temp_prior = (
        temp_context[temp_context["year"].astype(int) < int(year)]
        [["year", "temp_anomaly_c"]]
        .dropna()
        .sort_values("year")
        .tail(TEMP_LOOKBACK)
    )
    if len(temp_prior) < TEMP_LOOKBACK:
        return None

    gas_ordered = gas_context[gas_context["year"].astype(int) <= int(year)][["year", *GAS_COLUMNS]].dropna().sort_values("year")
    current = gas_ordered[gas_ordered["year"].astype(int) == int(year)]
    if current.empty:
        return None

    temp_window = temp_prior["temp_anomaly_c"].to_numpy(dtype=float)
    temp_slope = float(np.polyfit(np.arange(len(temp_window)), temp_window, 1)[0])
    temp_summary = np.array(
        [temp_window[-1], temp_window.mean(), temp_window.std(), temp_slope, temp_window[-1] - temp_window[-2]],
        dtype=float,
    )
    temp_fft = causal_fft_features(temp_window, keep=3)

    current_gas = current.iloc[-1][GAS_COLUMNS].to_numpy(dtype=float)
    gas_values = gas_ordered[GAS_COLUMNS].to_numpy(dtype=float)
    means, stds = gas_stats(gas_ordered)
    ghg_index = float(((current_gas - means) / stds).mean())
    forcing_proxy = float(forcing_proxy_from_values(current_gas.reshape(1, -1))[0])
    ghg_sequence = ((gas_values[-min(TEMP_LOOKBACK, len(gas_values)) :] - means) / stds).mean(axis=1)
    gas_fft = causal_fft_features(ghg_sequence, keep=3)

    extra = []
    if enso_context is not None:
        enso_row = enso_context[enso_context["year"].astype(int) == int(year)]
        if enso_row.empty or enso_row[ENSO_COLUMNS].isna().any(axis=None):
            return None
        extra.append(enso_row.iloc[0][ENSO_COLUMNS].to_numpy(dtype=float))

    return np.concatenate(
        [
            temp_window,
            temp_summary,
            temp_fft,
            current_gas,
            np.array([ghg_index, forcing_proxy], dtype=float),
            gas_fft,
            fourier_time_features(year).reshape(-1),
            *extra,
        ]
    )


def temperature_dataset(
    years: np.ndarray,
    temp_context: pd.DataFrame,
    gas_context: pd.DataFrame,
    target_temperature: pd.DataFrame,
    enso_context: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = target_temperature[["year", "temp_anomaly_c"]].dropna().assign(year=lambda frame: frame["year"].astype(int)).set_index("year")
    x_rows = []
    y_rows = []
    kept_years = []
    for year in sorted(int(year) for year in years):
        if year not in lookup.index:
            continue
        row = hybrid_temperature_features(year, temp_context, gas_context, enso_context)
        if row is None:
            continue
        x_rows.append(row)
        y_rows.append([float(lookup.loc[year, "temp_anomaly_c"])])
        kept_years.append(year)
    if not x_rows:
        raise ValueError("No temperature feature rows were built.")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=float), np.asarray(kept_years, dtype=int)
