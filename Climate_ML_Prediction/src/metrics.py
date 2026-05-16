from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    total = np.sum((true - true.mean()) ** 2)
    if total == 0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / total)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(true) < 2 or np.std(true) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def report(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> dict[str, float | str]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    base = np.asarray(baseline, dtype=float)
    model_rmse = rmse(true, pred)
    baseline_rmse = rmse(true, base)
    residual = pred - true
    return {
        "model": model,
        "split": split,
        "mae": mae(true, pred),
        "rmse": model_rmse,
        "max_abs_error": float(np.max(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": r2_score(true, pred),
        "pearson_corr": pearson_corr(true, pred),
        "baseline_rmse": baseline_rmse,
        "skill_vs_baseline_rmse": float(1.0 - model_rmse / baseline_rmse) if baseline_rmse else float("nan"),
    }
