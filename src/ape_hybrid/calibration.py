"""Validation-only affine calibration of predicted residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AffineResidualCalibrator:
    slope_bounds: tuple[float, float] = (0.0, 1.5)
    robust_sigma_multiplier: float = 3.0

    def fit(self, predicted_residual: np.ndarray, true_residual: np.ndarray):
        predicted = np.asarray(predicted_residual, dtype=float)
        truth = np.asarray(true_residual, dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(truth)
        if valid.sum() < 10:
            raise ValueError("At least ten finite calibration samples are required")
        slope, _ = np.polyfit(predicted[valid], truth[valid], 1)
        self.slope_ = float(np.clip(slope, *self.slope_bounds))
        self.intercept_ = float(np.mean(truth[valid] - self.slope_ * predicted[valid]))
        median = float(np.median(truth[valid]))
        mad = float(np.median(np.abs(truth[valid] - median)))
        self.clip_abs_ = max(self.robust_sigma_multiplier * 1.4826 * mad, 1.0e-6)
        self.n_samples_ = int(valid.sum())
        return self

    def transform(self, predicted_residual: np.ndarray) -> np.ndarray:
        calibrated = self.slope_ * np.asarray(predicted_residual, dtype=float) + self.intercept_
        return np.clip(calibrated, -self.clip_abs_, self.clip_abs_)

    def fit_transform(self, predicted_residual: np.ndarray, true_residual: np.ndarray) -> np.ndarray:
        return self.fit(predicted_residual, true_residual).transform(predicted_residual)
