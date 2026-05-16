# Climate ML Prediction

This project evaluates global annual temperature anomaly prediction with a small, reproducible machine learning workflow.

The workflow combines:

- NASA GISTEMP global annual temperature anomaly data
- NOAA global annual CO2, CH4, and N2O concentration data
- NOAA PSL Nino 3.4 monthly anomaly data summarized as annual ENSO features
- causal Fourier features computed only from information available before the target year
- strict chronological testing and random decade holdout testing

## Run

```powershell
python run.py
```

The script writes processed tables, prediction tables, metrics, and PNG figures to `outputs/`.

## Main Outputs

- `outputs/model_metrics.csv`
- `outputs/validation_predictions.csv`
- `outputs/strict_future_predictions.csv`
- `outputs/random_decade_predictions.csv`
- `outputs/greenhouse_predictions.csv`
- `outputs/figures/validation_forecast.png`
- `outputs/figures/strict_future_forecast.png`
- `outputs/figures/random_decade_scatter.png`
- `outputs/figures/rmse_summary.png`
- `outputs/figures/model_workflow.png`
