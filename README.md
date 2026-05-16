# Global Temperature Change Analysis

This repository contains the code and data used for a course project on global temperature change. The project has two parts: basic analysis and visualization of NASA GISS temperature records, and a small machine learning workflow for annual temperature anomaly prediction.

## Repository Structure

```text
Climate_ML_Prediction/   Machine learning workflow for annual anomaly prediction
Data_analysis/           Scripts for reading temperature data and drawing figures
Data_files/              Source files used by the analysis scripts
```

## Data

The main temperature datasets are stored in `Data_files/`:

- `GLB.Ts+dSST.txt`: NASA GISTEMP global mean temperature anomaly table
- `gistemp1200_GHCNv4_ERSSTv5.nc.gz`: NASA GISS gridded temperature anomaly data

The machine learning folder also includes processed and raw inputs for greenhouse gas concentrations and the Nino 3.4 index.

## Data Analysis

`Data_analysis` reads the text and gridded temperature files, then generates figures for the written report.

Main outputs are written to a local `result/` folder when the scripts are run:

- global annual mean temperature anomaly since 1880
- global temperature anomaly map for a selected year
- spatial trend map for the selected trend period

Run from the repository root:

```powershell
python Data_analysis/main.py
```

The switches in `Data_analysis/parameters.py` can be used to turn the text-file analysis or NetCDF analysis on and off.

## Machine Learning Prediction

`Climate_ML_Prediction` builds annual prediction datasets from temperature, greenhouse gas, and ENSO records. It compares a persistence baseline with neural-network models using Fourier time features, greenhouse gas histories, and lagged ENSO features.

Run from the machine learning folder:

```powershell
cd Climate_ML_Prediction
python -m pip install -r requirements.txt
python run.py
```

The workflow writes prediction tables, metrics, and figures to `Climate_ML_Prediction/outputs/`.

## Requirements

The project uses Python with common data-analysis packages. For the full analysis workflow, install:

```powershell
python -m pip install numpy pandas matplotlib pillow
```

The machine learning workflow can also be installed from `Climate_ML_Prediction/requirements.txt`.
