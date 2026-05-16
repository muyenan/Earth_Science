from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

TEMPERATURE_FILE = RAW_DIR / "GLB.Ts+dSST.txt"
CO2_FILE = RAW_DIR / "co2_annmean_gl.csv"
CH4_FILE = RAW_DIR / "ch4_annmean_gl.csv"
N2O_FILE = RAW_DIR / "n2o_annmean_gl.csv"
NINO34_FILE = RAW_DIR / "nino34_anom.data"

RANDOM_SEED = 42
TEMP_LOOKBACK = 8
GAS_LOOKBACK = 3
FOURIER_PERIODS = (3, 5, 11, 22, 60)
STRICT_VALIDATION_YEARS = (2019, 2020, 2021)
STRICT_TEST_YEARS = (2022, 2023, 2024, 2025)
