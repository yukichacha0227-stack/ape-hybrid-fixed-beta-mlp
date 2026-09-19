"""Regression metrics reported by the experiment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    residual = truth - prediction
    rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
    mae = float(mean_absolute_error(truth, prediction))
    return {
        "n": int(len(truth)),
        "r2": float(r2_score(truth, prediction)),
        "rmse_eV": rmse,
        "rmse_meV": 1000.0 * rmse,
        "mae_eV": mae,
        "mae_meV": 1000.0 * mae,
        "mbe_truth_minus_prediction_meV": 1000.0 * float(np.mean(residual)),
    }
