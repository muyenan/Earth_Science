from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CH4_FILE, CO2_FILE, N2O_FILE, NINO34_FILE, PROCESSED_DIR, TEMPERATURE_FILE


def load_temperature(path: Path = TEMPERATURE_FILE) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    year_pattern = re.compile(r"^\d{4}$")
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 14 or not year_pattern.match(parts[0]):
                continue
            annual = parts[13]
            if "*" in annual:
                continue
            rows.append({"year": int(parts[0]), "temp_anomaly_c": float(annual) / 100.0})
    if not rows:
        raise ValueError(f"No annual temperature rows parsed from {path}")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def load_gas_csv(path: Path, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"year", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    frame["year"] = frame["year"].astype(int)
    return frame[["year", value_column]].sort_values("year").reset_index(drop=True)


def load_greenhouse() -> pd.DataFrame:
    co2 = load_gas_csv(CO2_FILE, "co2_ppm")
    ch4 = load_gas_csv(CH4_FILE, "ch4_ppb")
    n2o = load_gas_csv(N2O_FILE, "n2o_ppb")
    return co2.merge(ch4, on="year", how="outer").merge(n2o, on="year", how="outer").sort_values("year").reset_index(drop=True)


def load_nino34(path: Path = NINO34_FILE) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) != 13 or not parts[0].isdigit():
                continue
            year = int(parts[0])
            monthly = np.asarray([float(value) for value in parts[1:]], dtype=float)
            rows.append(
                {
                    "year": year,
                    "nino34_mean": float(monthly.mean()),
                    "nino34_abs_mean": float(np.abs(monthly).mean()),
                    "nino34_max": float(monthly.max()),
                    "nino34_min": float(monthly.min()),
                    "nino34_djf": float(np.mean([monthly[11], monthly[0], monthly[1]])),
                    "nino34_ndj": float(np.mean([monthly[10], monthly[11], monthly[0]])),
                }
            )
    if not rows:
        raise ValueError(f"No Nino 3.4 rows parsed from {path}")

    annual = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    lag_columns = ["nino34_mean", "nino34_abs_mean", "nino34_max", "nino34_min", "nino34_djf", "nino34_ndj"]
    for column in lag_columns:
        annual[f"{column}_lag1"] = annual[column].shift(1)
    annual["nino34_mean_lag2"] = annual["nino34_mean"].shift(2)
    annual["nino34_djf_lag2"] = annual["nino34_djf"].shift(2)
    annual["nino34_change_lag1"] = annual["nino34_mean_lag1"] - annual["nino34_mean_lag2"]
    annual["nino34_source"] = "NOAA PSL Nino 3.4 CPC monthly anomalies"
    return annual


def forcing_proxy_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    co2 = values[:, 0]
    ch4 = values[:, 1]
    n2o = values[:, 2]
    return (
        5.35 * np.log(co2 / 278.0)
        + 0.036 * (np.sqrt(ch4) - np.sqrt(722.0))
        + 0.12 * (np.sqrt(n2o) - np.sqrt(270.0))
    )


def add_greenhouse_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    gas_columns = ["co2_ppm", "ch4_ppb", "n2o_ppb"]
    gas_values = result[gas_columns].astype(float)
    standardized = (gas_values - gas_values.mean()) / gas_values.std().replace(0, 1.0)
    result["ghg_index"] = standardized.mean(axis=1)
    result["forcing_proxy"] = forcing_proxy_from_values(gas_values.to_numpy(dtype=float))
    return result


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    temperature = load_temperature()
    greenhouse = add_greenhouse_features(load_greenhouse().dropna(subset=["co2_ppm", "ch4_ppb", "n2o_ppb"]))
    nino34 = load_nino34()
    merged = temperature.merge(greenhouse, on="year", how="left").merge(nino34, on="year", how="left")

    temperature.to_csv(PROCESSED_DIR / "temperature_history.csv", index=False)
    greenhouse.to_csv(PROCESSED_DIR / "greenhouse_history.csv", index=False)
    nino34.to_csv(PROCESSED_DIR / "nino34_annual.csv", index=False)
    merged.to_csv(PROCESSED_DIR / "climate_merged.csv", index=False)
    return temperature, greenhouse, nino34, merged
