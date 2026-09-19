"""Feature definitions used by the published hybrid model."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = ("am", "omega", "aod550", "totangstr")


def beta_design(data: pd.DataFrame) -> pd.DataFrame:
    """Build the surface-conditioned fixed-beta design matrix.

    HSR uses the common atmospheric coefficients. TSR adds an intercept shift
    and a surface interaction for every atmospheric term.
    """
    required = {*BASE_COLUMNS, "surface"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing beta-design columns: {sorted(missing)}")

    result = pd.DataFrame(index=data.index)
    result["AM"] = data["am"].astype(float)
    result["omega"] = data["omega"].astype(float)
    result["AOD550"] = data["aod550"].astype(float)
    result["TOTANGSTR"] = data["totangstr"].astype(float)
    if "to3" in data:
        result["TO3"] = data["to3"].astype(float)

    surface_tsr = data["surface"].eq("TSR").astype(float)
    result["TSR_label"] = surface_tsr
    for source, label in (
        ("am", "AM"),
        ("omega", "omega"),
        ("aod550", "AOD550"),
        ("totangstr", "TOTANGSTR"),
    ):
        result[f"TSR_x_{label}"] = surface_tsr * data[source].astype(float)
    if "to3" in data:
        result["TSR_x_TO3"] = surface_tsr * data["to3"].astype(float)

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("The beta design contains non-finite values")
    return result


def residual_static_features(
    data: pd.DataFrame,
    beta_prediction: np.ndarray,
) -> pd.DataFrame:
    """Build non-lag features for the residual MLP."""
    if "datetime" not in data or "surface" not in data:
        raise KeyError("Residual features require datetime and surface")
    if len(beta_prediction) != len(data):
        raise ValueError("beta_prediction must have one value per row")

    timestamps = pd.to_datetime(data["datetime"], utc=True).dt.tz_convert("Asia/Tokyo")
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    day = timestamps.dt.dayofyear.astype(float)

    result = beta_design(data)
    result["tilt_angle_deg"] = data["surface"].map({"HSR": 0.0, "TSR": 32.0})
    result["beta_prediction"] = np.asarray(beta_prediction, dtype=float)
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["doy_sin"] = np.sin(2.0 * np.pi * day / 365.2425)
    result["doy_cos"] = np.cos(2.0 * np.pi * day / 365.2425)
    result["AM_x_AOD550"] = data["am"] * data["aod550"]
    result["AM_x_TOTANGSTR"] = data["am"] * data["totangstr"]
    result["AOD550_x_TOTANGSTR"] = data["aod550"] * data["totangstr"]
    result["AM_x_omega"] = data["am"] * data["omega"]
    result["transmittance_proxy"] = np.exp(-np.clip(result["AM_x_AOD550"], 0, 50))
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Residual features contain non-finite values")
    return result


def add_empty_lags(features: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 6)) -> pd.DataFrame:
    """Add zero-valued lag inputs plus explicit availability indicators."""
    result = features.copy()
    for lag in lags:
        result[f"residual_lag_{lag}h"] = 0.0
        result[f"residual_lag_{lag}h_available"] = 0.0
    return result


def add_observed_lags(
    data: pd.DataFrame,
    features: pd.DataFrame,
    residual: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 3, 6),
) -> pd.DataFrame:
    """Add causal residual lags independently for HSR and TSR."""
    result = features.copy()
    work = data[["datetime", "surface"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True)
    work["residual"] = np.asarray(residual, dtype=float)
    for lag in lags:
        values = np.zeros(len(work), dtype=float)
        available = np.zeros(len(work), dtype=float)
        for surface in ("HSR", "TSR"):
            indices = np.flatnonzero(work["surface"].eq(surface).to_numpy())
            history = {
                work.iloc[index]["datetime"]: work.iloc[index]["residual"]
                for index in indices
                if np.isfinite(work.iloc[index]["residual"])
            }
            for index in indices:
                previous = work.iloc[index]["datetime"] - pd.Timedelta(hours=lag)
                if previous in history:
                    values[index] = history[previous]
                    available[index] = 1.0
        result[f"residual_lag_{lag}h"] = values
        result[f"residual_lag_{lag}h_available"] = available
    return result
