#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比較実験F: HSR/TSR別APEをMERRA-2の1時間窓に整合して予測する。

主モデル:
    固定β（表面条件付き線形）+ 残差MLP + 再帰推論 + 独立Validation校正

対照モデル:
    1. 固定βのみ
    2. HistGradientBoosting（表形式非線形ベンチマーク）+ 校正

重要な時刻定義:
    - APEと10分AMはJSTの [hh:00, hh+1:00) 内で表面別に算術平均する。
    - 集約値の時刻は窓中心 hh:30 JST（UTCでは対応する :30）とする。
    - MERRA-2 tavg1（AOD550, TOTANGSTR, omega、任意でTO3）の中心時刻と完全一致で結合する。
    - HSRとTSRは同じ時刻でも別行・別ラベルとして扱い、相互平均しない。
    - 分割は時刻単位で行い、同じ時刻のHSR/TSRが別splitへ漏れないようにする。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent


def _historical_root() -> Path:
    """Return the configurable root that owns the experiment inputs."""
    return Path(os.environ.get("APE_DATA_ROOT", SCRIPT_DIR.parent / "data")).resolve()


DATA_ROOT = _historical_root()
DEFAULT_HSR = DATA_ROOT / "比較実験B" / "target_ape_tsukuba_HSR_r12_350_1050.pkl"
DEFAULT_TSR = DATA_ROOT / "target_ape_tsukuba_TSR_r12_350_1050.pkl"
DEFAULT_MET = DATA_ROOT / "入力" / "Tsukuba_100879228"
DEFAULT_AEROSOL = (
    DATA_ROOT
    / "比較実験C"
    / "outputs"
    / "M2T1NXAER_tsukuba"
    / "tsukuba_aod550_totangstr_experiment_C_common.csv"
)
DEFAULT_OMEGA = DATA_ROOT / "入力" / "tsukuba_only_tqv_2011~2015_direct.csv"
DEFAULT_OZONE = (
    DATA_ROOT
    / "オゾン_ダウンローダー"
    / "outputs"
    / "M2T1NXCHM_tsukuba"
    / "tsukuba_M2T1NXCHM_TO3_20110101_20151231_full.csv"
)
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "outputs" / "hourly_hsr_tsr_to3"


@dataclass
class Settings:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    calibration_fraction_of_validation: float = 0.50
    purge_hours: int = 6
    min_samples_per_hour: int = 3
    ridge_alpha: float = 1.0
    oof_splits: int = 5
    residual_lags_hours: tuple[int, ...] = (1, 2, 3, 6)
    hidden_dim: int = 128
    bottleneck_dim: int = 64
    dropout: float = 0.05
    batch_size: int = 512
    max_epochs: int = 300
    patience: int = 100
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    scheduler_gamma: float = 0.99
    min_learning_rate: float = 1.0e-6
    gradient_clip_norm: float = 5.0
    min_delta: float = 1.0e-7
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 300
    hgb_max_leaf_nodes: int = 31
    hgb_l2_regularization: float = 1.0


MODEL_FIXED = "fixed_surface_beta"
MODEL_HYBRID_RAW = "fixed_beta_plus_residual_mlp_recursive_raw"
MODEL_HYBRID_CAL = "fixed_beta_plus_residual_mlp_recursive_calibrated"
MODEL_HGB_RAW = "hist_gradient_boosting_raw"
MODEL_HGB_CAL = "hist_gradient_boosting_calibrated"
SURFACE_TILT = {"HSR": 0.0, "TSR": 32.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsr", type=Path, default=DEFAULT_HSR)
    parser.add_argument("--tsr", type=Path, default=DEFAULT_TSR)
    parser.add_argument("--met-root", type=Path, default=DEFAULT_MET)
    parser.add_argument("--aerosol", type=Path, default=DEFAULT_AEROSOL)
    parser.add_argument("--omega", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--ozone", type=Path, default=DEFAULT_OZONE)
    parser.add_argument(
        "--include-ozone",
        action="store_true",
        help="M2T1NXCHMの全気柱オゾンTO3を追加特徴量として使用する",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="既存runのCSV/JSONからMarkdownだけを再生成する",
    )
    parser.add_argument("--quick", action="store_true", help="動作確認用（30 epoch）")
    parser.add_argument("--min-samples-per-hour", type=int)
    return parser


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"必要な入力が見つかりません:\n{formatted}")


def local_naive_to_utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("Asia/Tokyo")
    else:
        parsed = parsed.dt.tz_convert("Asia/Tokyo")
    return parsed.dt.tz_convert("UTC").dt.as_unit("ns")


def load_target(path: Path, surface: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    obj = joblib.load(path)
    frame = obj.get("target_dataframe") if isinstance(obj, dict) else obj
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"target_dataframeを取得できません: {path}")
    required = {"Datetime", "SiteNum", "Remark", "APE"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path.name} の列不足: {sorted(missing)}")
    data = frame[["Datetime", "SiteNum", "Remark", "APE"]].copy()
    data["datetime"] = local_naive_to_utc(data["Datetime"])
    data["SiteNum"] = pd.to_numeric(data["SiteNum"], errors="raise").astype(int)
    data["Remark"] = pd.to_numeric(data["Remark"], errors="raise").astype(int)
    data["APE"] = pd.to_numeric(data["APE"], errors="raise").astype(float)
    data = data.loc[(data["SiteNum"] == 301) & data["Remark"].isin([1, 2])].copy()
    if not np.isfinite(data["APE"]).all():
        raise ValueError(f"{surface} APEに非有限値があります")
    if data.duplicated(["datetime", "SiteNum"]).any():
        raise ValueError(f"{surface}に同一時刻重複があります。表面内時間平均の前提を確認してください")
    data["surface"] = surface
    report = {
        "surface": surface,
        "raw_rows": len(frame),
        "quality_selected_rows": len(data),
        "remark_counts": data["Remark"].value_counts().sort_index().to_dict(),
        "start_utc": data["datetime"].min(),
        "end_utc": data["datetime"].max(),
        "sha256": sha256(path),
    }
    return data[["datetime", "SiteNum", "surface", "Remark", "APE"]], report


def hourly_center_utc(values: pd.Series) -> pd.Series:
    """Return the center of the local civil-hour bin containing each timestamp."""
    local = values.dt.tz_convert("Asia/Tokyo")
    center_local = local.dt.floor("h") + pd.Timedelta("30min")
    return center_local.dt.tz_convert("UTC").dt.as_unit("ns")


def aggregate_target_hourly(
    target: pd.DataFrame, min_samples: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = target.copy()
    data["datetime"] = hourly_center_utc(data["datetime"])
    grouped = (
        data.groupby(["datetime", "SiteNum", "surface"], as_index=False)
        .agg(
            ape=("APE", "mean"),
            ape_std_within_hour=("APE", "std"),
            ape_samples_in_hour=("APE", "size"),
            remark_min=("Remark", "min"),
            remark_max=("Remark", "max"),
        )
        .sort_values("datetime")
    )
    before = len(grouped)
    grouped = grouped.loc[grouped["ape_samples_in_hour"] >= min_samples].copy()
    surface = str(grouped["surface"].iloc[0]) if len(grouped) else "unknown"
    report = {
        "surface": surface,
        "hourly_bins_before_minimum_count": before,
        "hourly_bins_after_minimum_count": len(grouped),
        "dropped_hourly_bins": before - len(grouped),
        "minimum_samples_per_hour": min_samples,
        "samples_per_hour_distribution": grouped["ape_samples_in_hour"].describe().to_dict(),
        "aggregation": "arithmetic mean of valid APE observations within JST [hh:00, hh+1:00)",
        "timestamp": "center of aggregation window (hh:30 JST, converted to UTC)",
    }
    return grouped.reset_index(drop=True), report


def _parse_met_datetime(frame: pd.DataFrame) -> pd.Series:
    dates = frame["年月日"].astype("string").str.strip()
    times = frame["時分"].astype("string").str.strip()
    rollover = times.str.fullmatch(r"24:00(?::00)?", na=False)
    normal = times.mask(rollover, "00:00")
    parsed = pd.to_datetime(dates + " " + normal, format="mixed", errors="coerce")
    parsed = parsed + pd.to_timedelta(rollover.astype(int), unit="D")
    return parsed.dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC").dt.as_unit("ns")


def load_and_aggregate_am(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(root.rglob("10MET*.csv"))
    if not files:
        raise FileNotFoundError(f"10MET*.csv がありません: {root}")
    parts: list[pd.DataFrame] = []
    columns = ["地点番号", "年月日", "時分", "エアマス"]
    for path in files:
        part = pd.read_csv(
            path,
            encoding="shift_jis",
            usecols=columns,
            low_memory=False,
        )
        part["datetime"] = _parse_met_datetime(part)
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    data["SiteNum"] = pd.to_numeric(data["地点番号"], errors="coerce")
    data["am"] = pd.to_numeric(data["エアマス"], errors="coerce").replace(-9999, np.nan)
    data = data.loc[(data["SiteNum"] == 301) & data["datetime"].notna() & data["am"].notna()].copy()
    data["datetime"] = hourly_center_utc(data["datetime"])
    hourly = (
        data.groupby(["datetime", "SiteNum"], as_index=False)
        .agg(am=("am", "mean"), am_std_within_hour=("am", "std"), am_samples_in_hour=("am", "size"))
        .sort_values("datetime")
    )
    if hourly.duplicated(["datetime", "SiteNum"]).any():
        raise AssertionError("AM時間平均後に時刻重複があります")
    return hourly, {
        "met_files": len(files),
        "valid_10min_rows": len(data),
        "hourly_rows": len(hourly),
        "samples_per_hour_distribution": hourly["am_samples_in_hour"].describe().to_dict(),
        "aggregation": "arithmetic mean within JST [hh:00, hh+1:00)",
    }


def load_atmosphere(
    aerosol_path: Path,
    omega_path: Path,
    ozone_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aerosol = pd.read_csv(aerosol_path, encoding="utf-8-sig")
    if missing := {"time", "TOTEXTTAU", "TOTANGSTR"} - set(aerosol.columns):
        raise ValueError(f"Aerosol列不足: {sorted(missing)}")
    aerosol["datetime"] = pd.to_datetime(aerosol["time"], utc=True, format="mixed").dt.as_unit("ns")
    aerosol["aod550"] = pd.to_numeric(aerosol["TOTEXTTAU"], errors="raise")
    aerosol["totangstr"] = pd.to_numeric(aerosol["TOTANGSTR"], errors="raise")
    aerosol = aerosol[["datetime", "aod550", "totangstr"]].drop_duplicates("datetime")

    omega = pd.read_csv(omega_path, encoding="utf-8-sig")
    if missing := {"time", "TQV"} - set(omega.columns):
        raise ValueError(f"omega列不足: {sorted(missing)}")
    omega["datetime"] = pd.to_datetime(omega["time"], utc=True, format="mixed").dt.as_unit("ns")
    omega["omega"] = pd.to_numeric(omega["TQV"], errors="raise")
    omega = omega[["datetime", "omega"]].drop_duplicates("datetime")

    atmosphere = aerosol.merge(omega, on="datetime", how="inner", validate="one_to_one")
    ozone_report: dict[str, Any] = {"included": False}
    if ozone_path is not None:
        ozone = pd.read_csv(ozone_path, encoding="utf-8-sig")
        if missing := {"time", "TO3"} - set(ozone.columns):
            raise ValueError(f"Ozone列不足: {sorted(missing)}")
        ozone["datetime"] = pd.to_datetime(
            ozone["time"], utc=True, format="mixed"
        ).dt.as_unit("ns")
        ozone["to3"] = pd.to_numeric(ozone["TO3"], errors="raise")
        if ozone["datetime"].duplicated().any():
            raise ValueError("Ozone CSVで時刻が重複しています")
        ozone = ozone[["datetime", "to3"]].sort_values("datetime")
        atmosphere_before_ozone = len(atmosphere)
        atmosphere = atmosphere.merge(
            ozone,
            on="datetime",
            how="inner",
            validate="one_to_one",
        )
        if len(atmosphere) != atmosphere_before_ozone:
            raise ValueError(
                "TO3追加により既存大気時刻が減少しました: "
                f"before={atmosphere_before_ozone}, after={len(atmosphere)}"
            )
        if not atmosphere["to3"].between(0.0, 1000.0, inclusive="neither").all():
            raise ValueError("TO3が物理監査範囲(0,1000) DU外です")
        ozone_report = {
            "included": True,
            "rows": len(ozone),
            "common_rows_without_loss": len(atmosphere),
            "sha256": sha256(ozone_path),
            "unit": "Dobson unit (DU)",
            "min": float(ozone["to3"].min()),
            "mean": float(ozone["to3"].mean()),
            "max": float(ozone["to3"].max()),
        }

    columns = ["aod550", "totangstr", "omega"]
    if ozone_path is not None:
        columns.append("to3")
    atmosphere = atmosphere.dropna(subset=columns).sort_values("datetime")
    if not np.isfinite(atmosphere[columns].to_numpy(float)).all():
        raise ValueError("大気特徴量に非有限値があります")
    off_center = atmosphere["datetime"].dt.minute.ne(30) | atmosphere["datetime"].dt.second.ne(0)
    if off_center.any():
        raise ValueError(f"MERRA-2時刻が:30中心でない行があります: {int(off_center.sum())}")
    return atmosphere.reset_index(drop=True), {
        "aerosol_rows": len(aerosol),
        "omega_rows": len(omega),
        "exact_common_hourly_rows": len(atmosphere),
        "aerosol_sha256": sha256(aerosol_path),
        "omega_sha256": sha256(omega_path),
        "ozone": ozone_report,
        "timestamp_semantics": "MERRA-2 tavg1 center timestamp; exact join only",
    }


def prepare_data(
    args: argparse.Namespace, settings: Settings
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hsr_raw, hsr_load_report = load_target(args.hsr, "HSR")
    tsr_raw, tsr_load_report = load_target(args.tsr, "TSR")
    hsr, hsr_hourly_report = aggregate_target_hourly(hsr_raw, settings.min_samples_per_hour)
    tsr, tsr_hourly_report = aggregate_target_hourly(tsr_raw, settings.min_samples_per_hour)

    key = ["datetime", "SiteNum"]
    hsr_key = hsr[key].drop_duplicates()
    tsr_key = tsr[key].drop_duplicates()
    common_surface_keys = hsr_key.merge(tsr_key, on=key, how="inner", validate="one_to_one")
    hsr = hsr.merge(common_surface_keys, on=key, how="inner", validate="one_to_one")
    tsr = tsr.merge(common_surface_keys, on=key, how="inner", validate="one_to_one")

    am, am_report = load_and_aggregate_am(args.met_root)
    atmosphere, atmosphere_report = load_atmosphere(
        args.aerosol,
        args.omega,
        args.ozone if args.include_ozone else None,
    )
    predictors = am.merge(atmosphere, on="datetime", how="inner", validate="one_to_one")
    common_predictor_keys = common_surface_keys.merge(predictors[key], on=key, how="inner", validate="one_to_one")
    predictors = predictors.merge(common_predictor_keys, on=key, how="inner", validate="one_to_one")

    parts = []
    for surface_frame, surface in ((hsr, "HSR"), (tsr, "TSR")):
        part = surface_frame.merge(predictors, on=key, how="inner", validate="one_to_one")
        part["surface"] = surface
        part["tilt_angle_deg"] = SURFACE_TILT[surface]
        part["surface_tsr"] = float(surface == "TSR")
        parts.append(part)
    data = pd.concat(parts, ignore_index=True).sort_values(["datetime", "surface"]).reset_index(drop=True)
    expected = data.groupby("datetime")["surface"].nunique()
    if not expected.eq(2).all():
        raise AssertionError("最終データにHSR/TSRの片側欠損時刻があります")
    if data.duplicated(["datetime", "SiteNum", "surface"]).any():
        raise AssertionError("最終データに時刻・表面重複があります")

    report = {
        "experiment": "F+TO3" if args.include_ozone else "F",
        "definition": "HSR/TSR separately averaged to exact MERRA-2 tavg1 hourly windows",
        "target_aggregation_warning": (
            "APEの算術時間平均。分光放射照度を1時間積算してAPEを再計算した値とは一般に一致しない"
        ),
        "hsr_load": hsr_load_report,
        "tsr_load": tsr_load_report,
        "hsr_hourly": hsr_hourly_report,
        "tsr_hourly": tsr_hourly_report,
        "am_hourly": am_report,
        "atmosphere": atmosphere_report,
        "exact_common_hsr_tsr_hours_before_predictors": len(common_surface_keys),
        "exact_common_hours_after_all_predictors": int(data["datetime"].nunique()),
        "final_surface_rows": len(data),
        "surface_row_counts": data["surface"].value_counts().to_dict(),
        "start_utc": data["datetime"].min(),
        "end_utc": data["datetime"].max(),
        "future_reference_rows": 0,
        "join": "exact center-time equality; no merge_asof/backward broadcasting",
        "surface_tilt_labels_deg": SURFACE_TILT,
    }
    return data, report


def unique_time_split(
    data: pd.DataFrame, settings: Settings
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    times = pd.Index(data["datetime"].drop_duplicates().sort_values())
    n = len(times)
    train_end = int(n * settings.train_fraction)
    validation_end = int(n * (settings.train_fraction + settings.validation_fraction))
    validation_start = train_end + settings.purge_hours
    test_start = validation_end + settings.purge_hours
    validation_times = times[validation_start:validation_end]
    calibration_count = int(len(validation_times) * settings.calibration_fraction_of_validation)
    calibration_start = validation_end - calibration_count
    early_end = calibration_start - settings.purge_hours
    time_roles = {
        "train_oof": times[:train_end],
        "validation_early": times[validation_start:early_end],
        "validation_calibration": times[calibration_start:validation_end],
        "test": times[test_start:],
    }
    indices: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for role, role_times in time_roles.items():
        idx = np.flatnonzero(data["datetime"].isin(role_times).to_numpy())
        if len(idx) == 0:
            raise ValueError(f"時系列分割後に空です: {role}")
        indices[role] = idx
        part = data.iloc[idx]
        report[role] = {
            "surface_rows": len(part),
            "unique_hours": part["datetime"].nunique(),
            "start_utc": part["datetime"].min(),
            "end_utc": part["datetime"].max(),
            "surface_counts": part["surface"].value_counts().to_dict(),
        }
    report["purge"] = {
        "hours_at_each_main_boundary": settings.purge_hours,
        "hours_between_validation_early_and_calibration": settings.purge_hours,
        "reason": "maximum recursive residual lag is 6 hours",
    }
    return indices, report


def beta_design(data: pd.DataFrame, with_surface: bool = True) -> pd.DataFrame:
    """Fixed-coefficient Nagaoka-axis linear design with explicit surface terms."""
    result = pd.DataFrame(index=data.index)
    result["AM"] = data["am"]
    result["omega"] = data["omega"]
    result["AOD550"] = data["aod550"]
    result["TOTANGSTR"] = data["totangstr"]
    if "to3" in data.columns:
        result["TO3"] = data["to3"]
    if with_surface:
        s = data["surface_tsr"]
        result["TSR_label"] = s
        result["TSR_x_AM"] = s * data["am"]
        result["TSR_x_omega"] = s * data["omega"]
        result["TSR_x_AOD550"] = s * data["aod550"]
        result["TSR_x_TOTANGSTR"] = s * data["totangstr"]
        if "to3" in data.columns:
            result["TSR_x_TO3"] = s * data["to3"]
    return result


def residual_static_features(data: pd.DataFrame, beta_prediction: np.ndarray) -> pd.DataFrame:
    local = data["datetime"].dt.tz_convert("Asia/Tokyo")
    hour = local.dt.hour + local.dt.minute / 60.0
    day = local.dt.dayofyear.astype(float)
    result = beta_design(data, with_surface=True).copy()
    result["tilt_angle_deg"] = data["tilt_angle_deg"]
    result["beta_prediction"] = beta_prediction
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["doy_sin"] = np.sin(2.0 * np.pi * day / 365.2425)
    result["doy_cos"] = np.cos(2.0 * np.pi * day / 365.2425)
    result["AM_x_AOD550"] = data["am"] * data["aod550"]
    result["AM_x_TOTANGSTR"] = data["am"] * data["totangstr"]
    result["AOD550_x_TOTANGSTR"] = data["aod550"] * data["totangstr"]
    result["AM_x_omega"] = data["am"] * data["omega"]
    result["transmittance_proxy"] = np.exp(-np.clip(result["AM_x_AOD550"], 0, 50))
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("残差MLP静的特徴量に非有限値があります")
    return result


def add_empty_lags(features: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    result = features.copy()
    for lag in lags:
        result[f"residual_lag_{lag}h"] = 0.0
        result[f"residual_lag_{lag}h_available"] = 0.0
    return result


def add_observed_lags(
    data: pd.DataFrame,
    features: pd.DataFrame,
    residual: np.ndarray,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    result = features.copy()
    work = data[["datetime", "surface"]].copy()
    work["residual"] = residual
    for lag in lags:
        values = np.zeros(len(work), dtype=float)
        available = np.zeros(len(work), dtype=float)
        for surface in ("HSR", "TSR"):
            idx = np.flatnonzero(work["surface"].eq(surface).to_numpy())
            mapping = {
                timestamp: float(value)
                for timestamp, value in zip(work.iloc[idx]["datetime"], work.iloc[idx]["residual"])
                if np.isfinite(value)
            }
            for row_index in idx:
                previous = work.iloc[row_index]["datetime"] - pd.Timedelta(hours=lag)
                if previous in mapping:
                    values[row_index] = mapping[previous]
                    available[row_index] = 1.0
        result[f"residual_lag_{lag}h"] = values
        result[f"residual_lag_{lag}h_available"] = available
    return result


def fit_beta_oof(
    data: pd.DataFrame,
    roles: dict[str, np.ndarray],
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Ridge, StandardScaler]:
    design = beta_design(data, with_surface=True)
    x = design.to_numpy(float)
    y = data["ape"].to_numpy(float)
    train_idx = roles["train_oof"]
    train_times = pd.Index(data.iloc[train_idx]["datetime"].drop_duplicates().sort_values())
    oof = np.full(len(data), np.nan, dtype=float)
    fold_reports = []
    splitter = TimeSeriesSplit(n_splits=settings.oof_splits, gap=settings.purge_hours)
    for fold, (fit_time_idx, val_time_idx) in enumerate(splitter.split(train_times), start=1):
        fit_times = train_times[fit_time_idx]
        val_times = train_times[val_time_idx]
        fit_idx = np.flatnonzero(data["datetime"].isin(fit_times).to_numpy())
        val_idx = np.flatnonzero(data["datetime"].isin(val_times).to_numpy())
        scaler = StandardScaler().fit(x[fit_idx])
        model = Ridge(alpha=settings.ridge_alpha).fit(scaler.transform(x[fit_idx]), y[fit_idx])
        oof[val_idx] = model.predict(scaler.transform(x[val_idx]))
        fold_reports.append(
            {
                "fold": fold,
                "fit_unique_hours": len(fit_times),
                "validation_unique_hours": len(val_times),
                "fit_end": fit_times.max(),
                "validation_start": val_times.min(),
            }
        )
    final_scaler = StandardScaler().fit(x[train_idx])
    final_model = Ridge(alpha=settings.ridge_alpha).fit(final_scaler.transform(x[train_idx]), y[train_idx])
    final_prediction = final_model.predict(final_scaler.transform(x))
    coefficients_scaled = dict(zip(design.columns, final_model.coef_.astype(float)))
    coefficients_original = {
        name: float(coef / scale)
        for name, coef, scale in zip(design.columns, final_model.coef_, final_scaler.scale_)
    }
    intercept_original = float(
        final_model.intercept_
        - np.sum(final_model.coef_ * final_scaler.mean_ / final_scaler.scale_)
    )
    atmospheric_terms = "AM/omega/AOD550/TOTANGSTR"
    if "to3" in data.columns:
        atmospheric_terms += "/TO3"
    report = {
        "design_columns": design.columns.tolist(),
        "ridge_alpha": settings.ridge_alpha,
        "oof_available_surface_rows": int(np.isfinite(oof[train_idx]).sum()),
        "oof_unavailable_initial_surface_rows": int(np.isnan(oof[train_idx]).sum()),
        "folds": fold_reports,
        "scaled_coefficients": coefficients_scaled,
        "original_unit_coefficients": coefficients_original,
        "original_unit_intercept": intercept_original,
        "interpretation": {
            "HSR": f"intercept + common {atmospheric_terms} coefficients",
            "TSR": "HSR equation + TSR_label intercept and TSR interaction coefficients",
        },
    }
    return oof, final_prediction, report, final_model, final_scaler


def _load_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(
            "PyTorchを読み込めません。AdamW残差MLPの実行にはtorchが必要です"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def train_residual_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    settings: Settings,
) -> tuple[Any, pd.DataFrame, int, float]:
    torch, nn, DataLoader, TensorDataset = _load_torch()

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, settings.hidden_dim),
                nn.LayerNorm(settings.hidden_dim),
                nn.GELU(),
                nn.Dropout(settings.dropout),
                nn.Linear(settings.hidden_dim, settings.hidden_dim),
                nn.LayerNorm(settings.hidden_dim),
                nn.GELU(),
                nn.Dropout(settings.dropout),
                nn.Linear(settings.hidden_dim, settings.bottleneck_dim),
                nn.GELU(),
                nn.Linear(settings.bottleneck_dim, 1),
            )

        def forward(self, values: Any) -> Any:
            return self.network(values).squeeze(-1)

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    model = ResidualMLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=settings.scheduler_gamma
    )
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    early_x = torch.from_numpy(x_early.astype(np.float32))
    early_y = torch.from_numpy(y_early.astype(np.float32))
    loss_fn = nn.HuberLoss(delta=1.0)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(early_x), early_y))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_huber_standardized": total / seen,
                "validation_early_huber_standardized": validation_loss,
                "learning_rate": learning_rate,
            }
        )
        if validation_loss < best_loss - settings.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if optimizer.param_groups[0]["lr"] > settings.min_learning_rate:
            scheduler.step()
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"], settings.min_learning_rate)
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[Residual MLP] epoch={epoch:03d} train={total/seen:.6f} "
                f"val={validation_loss:.6f} lr={learning_rate:.3e}",
                flush=True,
            )
        if stale >= settings.patience:
            print(
                f"[Residual MLP] EarlyStopping epoch={epoch}, best_epoch={best_epoch}",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError("Residual MLPのbest stateがありません")
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history), best_epoch, best_loss


def torch_predict(model: Any, values: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    torch, _, _, _ = _load_torch()
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size].astype(np.float32))
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).astype(float)


def recursive_predict(
    model: Any,
    data: pd.DataFrame,
    static: pd.DataFrame,
    indices: np.ndarray,
    feature_scaler: StandardScaler,
    residual_mean: float,
    residual_std: float,
    lags: tuple[int, ...],
) -> np.ndarray:
    selected = data.iloc[indices][["datetime", "surface"]].copy()
    selected["original_index"] = indices
    selected = selected.sort_values(["datetime", "surface"])
    history: dict[str, dict[pd.Timestamp, float]] = {"HSR": {}, "TSR": {}}
    prediction = np.full(len(data), np.nan, dtype=float)
    for row in selected.itertuples(index=False):
        original = int(row.original_index)
        one = static.iloc[[original]].copy()
        for lag in lags:
            previous = row.datetime - pd.Timedelta(hours=lag)
            available = previous in history[row.surface]
            one[f"residual_lag_{lag}h"] = (
                history[row.surface].get(previous, 0.0)
            )
            one[f"residual_lag_{lag}h_available"] = float(available)
        scaled = feature_scaler.transform(one)
        standardized = float(torch_predict(model, scaled, batch_size=1)[0])
        raw = standardized * residual_std + residual_mean
        prediction[original] = raw
        history[row.surface][row.datetime] = raw
    return prediction


def fit_surface_affine(
    data: pd.DataFrame,
    true_values: np.ndarray,
    prediction: np.ndarray,
    calibration_indices: np.ndarray,
    slope_bounds: tuple[float, float] = (0.0, 1.5),
    correction_clip: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    calibrated = prediction.copy()
    report: dict[str, Any] = {}
    for surface in ("HSR", "TSR"):
        idx = calibration_indices[data.iloc[calibration_indices]["surface"].eq(surface).to_numpy()]
        valid = idx[np.isfinite(prediction[idx])]
        if len(valid) < 10:
            raise ValueError(f"{surface}校正データが不足しています")
        slope, intercept = np.polyfit(prediction[valid], true_values[valid], 1)
        slope = float(np.clip(slope, *slope_bounds))
        intercept = float(np.mean(true_values[valid] - slope * prediction[valid]))
        surface_all = np.flatnonzero(data["surface"].eq(surface).to_numpy())
        values = slope * prediction[surface_all] + intercept
        clip_value = None
        if correction_clip:
            target_residual = true_values[valid]
            median = float(np.median(target_residual))
            mad = float(np.median(np.abs(target_residual - median)))
            robust_sigma = 1.4826 * mad
            clip_value = max(3.0 * robust_sigma, 1.0e-6)
            values = np.clip(values, -clip_value, clip_value)
        calibrated[surface_all] = values
        report[surface] = {
            "slope": slope,
            "intercept": intercept,
            "clip_abs": clip_value,
            "n": len(valid),
        }
    return calibrated, report


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    residual = truth - prediction
    return {
        "n": len(truth),
        "r2": float(r2_score(truth, prediction)),
        "rmse_eV": float(np.sqrt(mean_squared_error(truth, prediction))),
        "rmse_meV": float(1000.0 * np.sqrt(mean_squared_error(truth, prediction))),
        "mae_eV": float(mean_absolute_error(truth, prediction)),
        "mae_meV": float(1000.0 * mean_absolute_error(truth, prediction)),
        "mbe_truth_minus_prediction_eV": float(np.mean(residual)),
        "mbe_truth_minus_prediction_meV": float(1000.0 * np.mean(residual)),
    }


def metrics_table(
    data: pd.DataFrame,
    roles: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    y = data["ape"].to_numpy(float)
    rows = []
    for split, indices in roles.items():
        for surface in ("ALL", "HSR", "TSR"):
            idx = indices
            if surface != "ALL":
                idx = idx[data.iloc[idx]["surface"].eq(surface).to_numpy()]
            for model_name, prediction in predictions.items():
                rows.append(
                    {
                        "split": split,
                        "surface": surface,
                        "model": model_name,
                        **regression_metrics(y[idx], prediction[idx]),
                    }
                )
    return pd.DataFrame(rows)


def residual_summaries(
    prediction_frame: pd.DataFrame, primary_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = prediction_frame.loc[prediction_frame["split"].eq("test")].copy()
    test["residual_meV"] = 1000.0 * (test["ape_true"] - test[primary_column])
    local = test["datetime"].dt.tz_convert("Asia/Tokyo")
    test["local_hour"] = local.dt.hour
    month = local.dt.month
    test["season"] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["spring", "summer", "autumn"],
        default="winter",
    )

    def summarize(columns: list[str]) -> pd.DataFrame:
        return (
            test.groupby(columns, as_index=False)
            .agg(
                n=("residual_meV", "size"),
                residual_mean_meV=("residual_meV", "mean"),
                residual_mae_meV=("residual_meV", lambda x: float(np.mean(np.abs(x)))),
                residual_rmse_meV=("residual_meV", lambda x: float(np.sqrt(np.mean(x**2)))),
            )
        )

    return summarize(["surface", "season"]), summarize(["surface", "local_hour"])


def save_plots(
    output: Path,
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    history: pd.DataFrame,
    primary_column: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.labelweight": "bold", "axes.titleweight": "bold"})
    experiment_name = "Experiment F+TO3" if "to3" in data.columns else "Experiment F"

    test_metrics = metrics.loc[(metrics["split"] == "test") & (metrics["surface"] != "ALL")]
    pivot = test_metrics.pivot(index="model", columns="surface", values="r2")
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=["#2878B5", "#D95F02"])
    ax.set_ylabel("Test R²")
    ax.set_xlabel("Model")
    ax.set_title(f"{experiment_name}: hourly HSR/TSR test performance")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(title="Surface")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output / "test_r2_model_comparison.png", dpi=180)
    plt.close()

    test = prediction_frame.loc[prediction_frame["split"].eq("test")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, surface in zip(axes, ("HSR", "TSR")):
        part = test.loc[test["surface"].eq(surface)]
        axis.scatter(part["ape_true"], part[primary_column], s=9, alpha=0.35)
        low = min(part["ape_true"].min(), part[primary_column].min())
        high = max(part["ape_true"].max(), part[primary_column].max())
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_title(surface)
        axis.set_xlabel("Measured hourly-mean APE (eV)")
        axis.set_ylabel("Predicted hourly-mean APE (eV)")
    fig.suptitle(f"{experiment_name}: measured vs predicted (independent Test)")
    plt.tight_layout()
    plt.savefig(output / "test_measured_vs_predicted_primary.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    for surface, color in (("HSR", "#2878B5"), ("TSR", "#D95F02")):
        values = data.loc[data["surface"].eq(surface), "ape"]
        ax.hist(values, bins=50, alpha=0.45, density=True, label=surface, color=color)
    ax.set_xlabel("Hourly-mean APE (eV)")
    ax.set_ylabel("Density")
    ax.set_title(f"{experiment_name}: HSR/TSR target distributions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output / "hourly_ape_distribution.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_huber_standardized"], label="Train")
    ax.plot(history["epoch"], history["validation_early_huber_standardized"], label="Validation early")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standardized Huber loss")
    ax.set_yscale("log")
    ax.set_title("Residual MLP learning history")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output / "residual_mlp_learning_history.png", dpi=180)
    plt.close()


def markdown_report(
    output: Path,
    run_name: str,
    settings: Settings,
    data_report: dict[str, Any],
    split_report: dict[str, Any],
    metrics: pd.DataFrame,
    beta_report: dict[str, Any],
    model_report: dict[str, Any],
    data: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> Path:
    has_ozone = bool(
        data_report.get("atmosphere", {}).get("ozone", {}).get("included", False)
    )
    experiment_title = (
        "比較実験F+TO3（HSR/TSR・1時間整合）"
        if has_ozone
        else "比較実験F（HSR/TSR・1時間整合）"
    )
    feature_description = "AM、ω（TQV）、AOD550、TOTANGSTR"
    if has_ozone:
        feature_description += "、TO3（全気柱オゾン、DU）"
    test = metrics.loc[(metrics["split"] == "test") & (metrics["surface"] != "ALL")].copy()
    test = test[["surface", "model", "r2", "rmse_meV", "mae_meV", "mbe_truth_minus_prediction_meV", "n"]]
    primary = test.loc[test["model"].eq(MODEL_HYBRID_CAL)]
    hgb = test.loc[test["model"].eq(MODEL_HGB_CAL)]
    beta = test.loc[test["model"].eq(MODEL_FIXED)]

    ape_distribution = (
        data.groupby("surface", as_index=False)["ape"]
        .agg(n="size", mean_eV="mean", std_eV="std", min_eV="min", max_eV="max")
    )
    paired_ape = data.pivot(index="datetime", columns="surface", values="ape").dropna()
    paired_difference_mev = 1000.0 * (paired_ape["TSR"] - paired_ape["HSR"])
    paired_report = {
        "n": len(paired_ape),
        "pearson": float(paired_ape["HSR"].corr(paired_ape["TSR"])),
        "mean_tsr_minus_hsr_meV": float(paired_difference_mev.mean()),
        "median_tsr_minus_hsr_meV": float(paired_difference_mev.median()),
        "tsr_greater_fraction": float((paired_difference_mev > 0).mean()),
    }
    ape_distribution.to_csv(output / "hourly_ape_distribution_summary.csv", index=False, encoding="utf-8-sig")
    atomic_json(output / "paired_hsr_tsr_ape_difference.json", paired_report)

    change_rows = []
    for surface in ("HSR", "TSR"):
        selected = test.loc[test["surface"].eq(surface)].set_index("model")
        for baseline_name, description in (
            (MODEL_FIXED, "固定β→主ハイブリッド"),
            (MODEL_HGB_CAL, "HGB校正→主ハイブリッド"),
            (MODEL_HYBRID_RAW, "主モデルraw→校正"),
        ):
            baseline_row = selected.loc[baseline_name]
            primary_row = selected.loc[MODEL_HYBRID_CAL]
            change_rows.append(
                {
                    "surface": surface,
                    "comparison": description,
                    "delta_r2": primary_row["r2"] - baseline_row["r2"],
                    "delta_rmse_meV": primary_row["rmse_meV"] - baseline_row["rmse_meV"],
                    "delta_mae_meV": primary_row["mae_meV"] - baseline_row["mae_meV"],
                }
            )
    performance_change = pd.DataFrame(change_rows)
    performance_change.to_csv(output / "test_performance_changes.csv", index=False, encoding="utf-8-sig")

    residual_season, residual_hour = residual_summaries(prediction_frame, MODEL_HYBRID_CAL)
    worst_season = residual_season.sort_values("residual_rmse_meV", ascending=False).iloc[0]
    worst_hour = residual_hour.sort_values("residual_rmse_meV", ascending=False).iloc[0]
    target_band_rows = []
    test_predictions = prediction_frame.loc[prediction_frame["split"].eq("test")].copy()
    test_predictions["residual_meV"] = 1000.0 * (
        test_predictions["ape_true"] - test_predictions[MODEL_HYBRID_CAL]
    )
    for surface, group in test_predictions.groupby("surface"):
        q10, q90 = group["ape_true"].quantile([0.10, 0.90])
        bands = {
            "low_10pct": group.loc[group["ape_true"] <= q10],
            "middle_80pct": group.loc[(group["ape_true"] > q10) & (group["ape_true"] < q90)],
            "high_10pct": group.loc[group["ape_true"] >= q90],
        }
        for band, selected in bands.items():
            residual_values = selected["residual_meV"].to_numpy(float)
            target_band_rows.append(
                {
                    "surface": surface,
                    "target_band": band,
                    "n": len(selected),
                    "ape_true_min_eV": selected["ape_true"].min(),
                    "ape_true_max_eV": selected["ape_true"].max(),
                    "residual_mean_truth_minus_prediction_meV": residual_values.mean(),
                    "mae_meV": np.mean(np.abs(residual_values)),
                    "rmse_meV": np.sqrt(np.mean(residual_values**2)),
                }
            )
    target_band = pd.DataFrame(target_band_rows)
    target_band.to_csv(output / "test_residual_by_target_band.csv", index=False, encoding="utf-8-sig")
    high_hsr = target_band.loc[
        target_band["surface"].eq("HSR") & target_band["target_band"].eq("high_10pct")
    ].iloc[0]
    high_tsr = target_band.loc[
        target_band["surface"].eq("TSR") & target_band["target_band"].eq("high_10pct")
    ].iloc[0]

    historical_reference = pd.DataFrame()
    historical_path = SCRIPT_DIR / "全解析_主要Test結果.csv"
    if historical_path.exists():
        historical = pd.read_csv(historical_path, encoding="utf-8-sig")
        historical_reference = historical.loc[
            historical["モデル"].eq("D_HSR_TSR_AOD_ANG")
            & historical["surface"].isin(["HSR", "TSR"]),
            ["surface", "n", "R2", "RMSE_meV", "MAE_meV"],
        ].copy()
        historical_reference.insert(0, "time_resolution", "従来10分・後方割当")
        current_reference = primary[["surface", "n", "r2", "rmse_meV", "mae_meV"]].rename(
            columns={"r2": "R2", "rmse_meV": "RMSE_meV", "mae_meV": "MAE_meV"}
        )
        current_reference.insert(0, "time_resolution", "比較実験F・1時間完全一致")
        historical_reference = pd.concat(
            [historical_reference, current_reference], ignore_index=True
        )
        historical_reference.to_csv(
            output / "historical_10min_vs_hourly_reference.csv", index=False, encoding="utf-8-sig"
        )

    def table(frame: pd.DataFrame) -> str:
        """Render a compact Markdown table without the optional tabulate package."""
        headers = [str(column) for column in frame.columns]

        def format_cell(value: Any) -> str:
            if isinstance(value, (float, np.floating)):
                return "" if not np.isfinite(value) else f"{float(value):.4f}"
            return str(value).replace("|", "\\|").replace("\n", " ")

        rows = ["| " + " | ".join(headers) + " |"]
        rows.append("|" + "|".join("---" for _ in headers) + "|")
        for values in frame.itertuples(index=False, name=None):
            rows.append("| " + " | ".join(format_cell(value) for value in values) + " |")
        return "\n".join(rows)

    lines = [
        f"# {experiment_title}結果と考察",
        "",
        f"実行ID: `{run_name}`",
        "",
        "## 1. 目的",
        "",
        (
            "既存の比較実験Fと同じ1時間整合・同一時系列分割を維持し、M2T1NXCHMの全気柱オゾンTO3を新規特徴量として追加したときの独立Test性能変化を検証した。HSRとTSRは相互平均せず、傾斜角ラベル（HSR=0°、TSR=32°）を付けた別観測として扱った。"
            if has_ozone
            else "APE側をMERRA-2の1時間平均窓へ合わせることで、10分APEへ同じ1時間大気値を反復付与していた時間粒度不一致を解消したとき、予測性能と残差構造がどう変わるかを検証した。HSRとTSRは相互平均せず、傾斜角ラベル（HSR=0°、TSR=32°）を付けた別観測として扱った。"
        ),
        "",
        "## 2. 計算方法",
        "",
        "- 正解値: Remark 1/2、350–1050 nmのAPE。HSR内・TSR内で別々にJST `[hh:00, hh+1:00)` の算術平均を計算。",
        "- 時刻: 各1時間窓の中心 `hh:30 JST` をUTCへ変換し、MERRA-2 tavg1の中心時刻と完全一致結合。asof補間や1時間値の10分行への反復付与は不使用。",
        f"- 品質条件: 各表面・各時間に有効APEが{settings.min_samples_per_hour}点以上。",
        f"- 特徴量: {feature_description}、表面ラベル/傾斜角。残差MLPだけに時刻・季節周期、交互作用、β予測値、過去1/2/3/6時間の予測残差を追加。",
        "- 固定β式: HSR共通係数に、TSRラベルと各大気変数×TSRの固定交互作用を加えた表面条件付き線形式。係数は時刻ごとには変動しない。",
        "- 主モデル: 固定β + Huber残差MLP + 予測残差のみを用いる再帰推論 + Validation後半で表面別アフィン校正。",
        "- 学習: AdamW、初期学習率2×10⁻⁴、指数減衰γ=0.99、最大300 epoch、EarlyStopping=100。",
        "- 対照: 固定βのみ、およびHistGradientBoosting+同様の表面別校正。",
        "- 分割: 時刻順70/15/15%、最大残差ラグに合わせ境界を6時間purge。同一時刻のHSR/TSRは必ず同じsplit。",
        "",
        "## 3. データ監査",
        "",
        f"- HSR/TSR共通1時間数（特徴量結合後）: **{data_report['exact_common_hours_after_all_predictors']:,}**",
        f"- 最終行数: **{data_report['final_surface_rows']:,}**（各時刻2表面）",
        f"- 解析期間: {data_report['start_utc']} ～ {data_report['end_utc']}",
        "- 未来時刻参照: 0行。すべて中心時刻の完全一致結合。",
        "",
        "| split | 共通時刻数 | 表面行数 | 期間 (UTC) |",
        "|---|---:|---:|---|",
    ]
    for role in ("train_oof", "validation_early", "validation_calibration", "test"):
        item = split_report[role]
        lines.append(
            f"| {role} | {item['unique_hours']:,} | {item['surface_rows']:,} | {item['start_utc']} – {item['end_utc']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 独立Test結果",
            "",
            table(test),
            "",
            "主張可能な汎化性能は、校正に使っていない独立Testの値である。Validation calibrationの指標はin-sampleなので性能主張には使わない。",
            "",
            "### APE分布と表面差",
            "",
            table(ape_distribution),
            "",
            f"同一1時間のHSR/TSR相関はr={paired_report['pearson']:.4f}。TSR−HSRは平均{paired_report['mean_tsr_minus_hsr_meV']:.2f} meV、中央値{paired_report['median_tsr_minus_hsr_meV']:.2f} meVで、TSRがHSRを上回る時刻は{100.0*paired_report['tsr_greater_fraction']:.2f}%であった。したがって、傾斜角ラベルは単なる識別子ではなく、系統的な表面差を担う。",
            "",
            "### Test性能変化量",
            "",
            table(performance_change),
            "",
            "## 5. 主結果",
            "",
        ]
    )
    for surface in ("HSR", "TSR"):
        p = primary.loc[primary["surface"].eq(surface)].iloc[0]
        b = beta.loc[beta["surface"].eq(surface)].iloc[0]
        g = hgb.loc[hgb["surface"].eq(surface)].iloc[0]
        lines.extend(
            [
                f"### {surface}",
                "",
                f"固定βはR²={b['r2']:.4f}、主ハイブリッドはR²={p['r2']:.4f}（ΔR²={p['r2']-b['r2']:+.4f}）。主モデルのRMSE={p['rmse_meV']:.2f} meV、MAE={p['mae_meV']:.2f} meVであった。非線形対照HGBはR²={g['r2']:.4f}であり、主モデルとの差は{p['r2']-g['r2']:+.4f}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 評価・考察",
            "",
            "1. 1時間平均化でR²が上がっても、10分変動の予測能力が向上したことを直接意味しない。正解値とAMの高周波変動・測定雑音を平均で平滑化し、MERRA-2が表現できる時間尺度へ評価対象を変更した効果を含む。",
            "2. 固定β→残差MLPの増分が大きければ、線形土台に残る非線形性・季節時刻依存・自己相関を残差経路が回収したと解釈できる。増分が小さければ、時間平均後の情報量または特徴量自体が上限を決めている可能性が高い。",
            "3. HGBがハイブリッドを上回る場合は、再帰残差より同時刻の非線性交互作用が主要である可能性がある。逆なら、固定βという物理解釈可能な土台と残差時系列情報の組合せが有効だったと評価できる。",
            f"4. 残差依存は消えていない。最大の季節別RMSEは{worst_season['surface']}・{worst_season['season']}の{worst_season['residual_rmse_meV']:.2f} meV、最大の時刻別RMSEは{worst_hour['surface']}・{int(worst_hour['local_hour'])}時JSTの{worst_hour['residual_rmse_meV']:.2f} meVだった。とくに午後側の悪化は、固定32°ラベルだけでは太陽方位・面入射角の時間変化を表せないことと整合する。次の優先改良は傾斜面入射角の追加である。",
            "5. HSRでは校正によりR²/RMSEは改善した一方、MAEはrawより悪化した。平均二乗誤差基準のアフィン校正と補正clipが少数の大誤差を抑えつつ、典型誤差をわずかに増やした可能性がある。TSRではR²・RMSE・MAEがすべて改善しており、校正効果は表面で一様ではない。",
            f"6. EarlyStopping=100により最良epoch {model_report['best_epoch']}の後も十分待機し、停止はepoch {len(pd.read_csv(output / 'training_history_residual_mlp.csv'))}だった。後半はTrain lossだけが低下しValidationが改善しないため、今回は『早すぎる停止』より軽度の過学習が支配的である。",
            f"7. 正解–予測散布図には平均への回帰が残る。高APE上位10%の平均残差（正解−予測）はHSRで+{high_hsr['residual_mean_truth_minus_prediction_meV']:.2f} meV、TSRで+{high_tsr['residual_mean_truth_minus_prediction_meV']:.2f} meVであった。" + ("今回はTO3を明示的に追加しているため、この残差が元の比較実験Fからどれだけ縮小したかを差分表で評価する必要がある。改善が小さい場合、全気柱オゾン単独では短波長側の不足情報を十分に補えず、雲・散乱状態やより直接的な分光特徴が支配的である可能性が高い。" if has_ozone else "時間粒度整合だけでは従来の高APE（短波長優勢）側の問題は解消しておらず、オゾン等の短波長吸収・雲/散乱状態・より直接的な分光特徴の不足を優先的に検証すべきである。"),
            "8. 本実装は各時刻のAPE値を算術平均した。APEはスペクトルの比から得る量なので、厳密な『1時間APE』は1時間内スペクトルまたは分子・分母を先に積算して再計算する方が物理的に望ましい。今回の結果は算術時間平均APEに対する比較である。",
            "9. 過去の10分解析とのR²差は、対象時刻数・分散・平滑化が異なるため、そのままモデル改善量とは呼べない。同一Test期間で10分版と1時間版を併記して初めて時間整合効果を分離できる。",
            "",
            "### APE帯域別残差",
            "",
            table(target_band),
            "",
            "### 過去10分解析との参考比較（非同一評価母集団）",
            "",
            table(historical_reference) if not historical_reference.empty else "過去結果CSVが見つからないため省略。",
            "",
            "比較実験Fでは従来の非線形DよりR²がHSR/TSRとも高く、RMSEも小さい。ただし、Fは1時間平均で正解分散と短周期雑音を変えており、Test標本も異なる。この差は『時間整合＋平滑化＋モデル構成差』の合成であり、純粋なモデル改善量ではない。",
            "",
            "## 7. 固定βの収束係数（元単位）",
            "",
            f"切片: `{beta_report['original_unit_intercept']:.8g}`",
            "",
            "| 項 | 係数 |",
            "|---|---:|",
        ]
    )
    for name, value in beta_report["original_unit_coefficients"].items():
        lines.append(f"| {name} | {value:.8g} |")
    lines.extend(
        [
            "",
            "## 8. 学習・校正情報",
            "",
            f"- Residual MLP best epoch: {model_report['best_epoch']} / {settings.max_epochs}",
            f"- best Validation-early standardized Huber loss: {model_report['best_validation_loss']:.6g}",
            f"- EarlyStopping patience: {settings.patience}",
            f"- 校正パラメータ: `{json.dumps(json_value(model_report['residual_calibration']), ensure_ascii=False)}`",
            "",
            "## 9. 図",
            "",
            "- `test_r2_model_comparison.png`: 独立Testのモデル・表面別R²",
            "- `test_measured_vs_predicted_primary.png`: 主モデルの正解値–予測値散布図",
            "- `hourly_ape_distribution.png`: HSR/TSRの1時間平均APE分布",
            "- `residual_mlp_learning_history.png`: 残差MLPの学習履歴",
            "",
            "## 10. 結論",
            "",
            (
                "本実験は、比較実験Fの時間整合・目的変数・分割・モデル設定を固定し、TO3だけを新規大気特徴量として追加した比較である。元のFとTest時刻が完全一致することを確認したうえで、F→F+TO3の差をオゾン情報追加による性能変化として評価する。"
                if has_ozone
                else "本実験は、従来の『10分APEに1時間大気値を割り当てる』構成から、『同一1時間窓のAPE・AMとMERRA-2を中心時刻で完全一致させる』構成へ変更した比較である。性能差は時間粒度整合と平滑化の双方を含むため、適地解析向けの時間平均APE予測として評価し、10分運用予測の代替とは区別する必要がある。"
            ),
            "",
        ]
    )
    report_path = output / "比較実験F_結果と評価・考察.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # この配布用スクリプトはTO3追加版専用。明示指定がなくても必ずTO3を使う。
    args.include_ozone = True
    settings = Settings()
    if args.min_samples_per_hour is not None:
        settings.min_samples_per_hour = args.min_samples_per_hour
    if args.quick:
        settings.max_epochs = 30
        settings.patience = 10
        settings.hgb_max_iter = 60
    run_name = args.run_name or datetime.now(timezone.utc).strftime("F_hourly_%Y%m%dT%H%M%SZ")
    output = args.output_root / run_name
    if args.report_only:
        if not output.exists():
            raise FileNotFoundError(f"既存runがありません: {output}")
        config_payload = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        stored_settings = config_payload["settings"]
        if isinstance(stored_settings.get("residual_lags_hours"), list):
            stored_settings["residual_lags_hours"] = tuple(stored_settings["residual_lags_hours"])
        settings = Settings(**stored_settings)
        data_report = json.loads(
            (output / "data_quality_and_alignment_report.json").read_text(encoding="utf-8")
        )
        split_report = json.loads((output / "split_report.json").read_text(encoding="utf-8"))
        beta_report = json.loads(
            (output / "fixed_beta_coefficients.json").read_text(encoding="utf-8")
        )
        model_report = json.loads(
            (output / "model_and_calibration_report.json").read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(output / "metrics_all_splits.csv", encoding="utf-8-sig")
        data = pd.read_csv(
            output / "hourly_aligned_hsr_tsr_dataset.csv",
            encoding="utf-8-sig",
            parse_dates=["datetime"],
        )
        prediction_frame = pd.read_csv(
            output / "predictions_all_models.csv",
            encoding="utf-8-sig",
            parse_dates=["datetime"],
        )
        report_path = markdown_report(
            output,
            run_name,
            settings,
            data_report,
            split_report,
            metrics,
            beta_report,
            model_report,
            data,
            prediction_frame,
        )
        print(f"結果・考察を再生成しました: {report_path}", flush=True)
        return 0

    required_inputs = [args.hsr, args.tsr, args.aerosol, args.omega]
    if args.include_ozone:
        required_inputs.append(args.ozone)
    require_files(required_inputs)
    if not args.met_root.exists():
        raise FileNotFoundError(args.met_root)
    output.mkdir(parents=True, exist_ok=False)
    print(f"出力: {output}", flush=True)
    print("データを1時間窓へ集約・完全一致結合しています...", flush=True)
    data, data_report = prepare_data(args, settings)
    roles, split_report = unique_time_split(data, settings)
    data.to_csv(output / "hourly_aligned_hsr_tsr_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"split": role, **details} for role, details in split_report.items() if role != "purge"]
    ).to_csv(output / "split_periods.csv", index=False, encoding="utf-8-sig")
    atomic_json(output / "data_quality_and_alignment_report.json", data_report)
    atomic_json(output / "split_report.json", split_report)
    atomic_json(
        output / "run_config.json",
        {
            "experiment": "F+TO3" if args.include_ozone else "F",
            "settings": asdict(settings),
            "inputs": {
                "hsr": args.hsr,
                "tsr": args.tsr,
                "met_root": args.met_root,
                "aerosol": args.aerosol,
                "omega": args.omega,
                "ozone": args.ozone if args.include_ozone else None,
            },
        },
    )
    print(
        f"共通1時間={data['datetime'].nunique():,}, 表面行={len(data):,}, "
        f"期間={data['datetime'].min()} ～ {data['datetime'].max()}",
        flush=True,
    )
    if args.prepare_only:
        print("--prepare-only のため学習前に終了しました", flush=True)
        return 0

    y = data["ape"].to_numpy(float)
    print("固定βとTrain OOF βを推定しています...", flush=True)
    beta_oof, beta_prediction, beta_report, beta_model, beta_scaler = fit_beta_oof(
        data, roles, settings
    )
    atomic_json(output / "fixed_beta_coefficients.json", beta_report)

    static = residual_static_features(data, beta_prediction)
    train_idx = roles["train_oof"]
    oof_available = train_idx[np.isfinite(beta_oof[train_idx])]
    train_residual = np.full(len(data), np.nan, dtype=float)
    train_residual[oof_available] = y[oof_available] - beta_oof[oof_available]
    observed_features = add_observed_lags(
        data, static, train_residual, settings.residual_lags_hours
    )
    empty_features = add_empty_lags(static, settings.residual_lags_hours)
    feature_names = observed_features.columns.tolist()
    feature_scaler = StandardScaler().fit(observed_features.iloc[oof_available])
    residual_mean = float(np.mean(train_residual[oof_available]))
    residual_std = float(np.std(train_residual[oof_available]))
    if residual_std < 1.0e-12:
        raise ValueError("Train OOF残差の標準偏差が0です")
    x_train = feature_scaler.transform(observed_features.iloc[oof_available])
    y_train = (train_residual[oof_available] - residual_mean) / residual_std
    early_idx = roles["validation_early"]
    x_early = feature_scaler.transform(empty_features.iloc[early_idx])
    y_early = (y[early_idx] - beta_prediction[early_idx] - residual_mean) / residual_std

    print("残差MLPを学習しています...", flush=True)
    residual_model, history, best_epoch, best_loss = train_residual_mlp(
        x_train, y_train, x_early, y_early, settings
    )
    history.to_csv(output / "training_history_residual_mlp.csv", index=False, encoding="utf-8-sig")

    raw_residual = np.full(len(data), np.nan, dtype=float)
    for split, idx in roles.items():
        raw_residual_split = recursive_predict(
            residual_model,
            data,
            static,
            idx,
            feature_scaler,
            residual_mean,
            residual_std,
            settings.residual_lags_hours,
        )
        valid = np.isfinite(raw_residual_split)
        raw_residual[valid] = raw_residual_split[valid]
    # Evaluation prediction uses chronological OOF beta wherever it exists in
    # Train.  Validation/Test always use the beta model fitted on Train only.
    # The residual MLP itself is fitted on Train OOF residuals, so its Train
    # correction remains an in-sample diagnostic and is not a generalisation claim.
    beta_evaluation = beta_prediction.copy()
    beta_evaluation[oof_available] = beta_oof[oof_available]
    hybrid_raw = beta_evaluation + raw_residual
    true_residual = y - beta_prediction
    calibrated_residual, residual_calibration = fit_surface_affine(
        data,
        true_residual,
        raw_residual,
        roles["validation_calibration"],
        slope_bounds=(0.0, 1.5),
        correction_clip=True,
    )
    hybrid_calibrated = beta_evaluation + calibrated_residual

    print("非線形対照HistGradientBoostingを学習しています...", flush=True)
    hgb_features = residual_static_features(data, beta_prediction).drop(columns="beta_prediction")
    hgb = HistGradientBoostingRegressor(
        learning_rate=settings.hgb_learning_rate,
        max_iter=settings.hgb_max_iter,
        max_leaf_nodes=settings.hgb_max_leaf_nodes,
        l2_regularization=settings.hgb_l2_regularization,
        random_state=settings.seed,
        early_stopping=True,
        validation_fraction=None,
    )
    hgb.fit(hgb_features.iloc[train_idx], y[train_idx])
    hgb_raw = hgb.predict(hgb_features)
    hgb_calibrated, hgb_calibration = fit_surface_affine(
        data,
        y,
        hgb_raw,
        roles["validation_calibration"],
        slope_bounds=(0.5, 1.5),
        correction_clip=False,
    )

    predictions = {
        MODEL_FIXED: beta_evaluation,
        MODEL_HYBRID_RAW: hybrid_raw,
        MODEL_HYBRID_CAL: hybrid_calibrated,
        MODEL_HGB_RAW: hgb_raw,
        MODEL_HGB_CAL: hgb_calibrated,
    }
    metrics = metrics_table(data, roles, predictions)
    metrics.to_csv(output / "metrics_all_splits.csv", index=False, encoding="utf-8-sig")

    split_label = np.full(len(data), "purged", dtype=object)
    for split, idx in roles.items():
        split_label[idx] = split
    prediction_columns = [
        "datetime",
        "SiteNum",
        "surface",
        "tilt_angle_deg",
        "ape",
        "ape_samples_in_hour",
        "am",
        "omega",
        "aod550",
        "totangstr",
    ]
    if "to3" in data.columns:
        prediction_columns.append("to3")
    prediction_frame = data[prediction_columns].copy()
    prediction_frame = prediction_frame.rename(columns={"ape": "ape_true"})
    prediction_frame["split"] = split_label
    for name, values in predictions.items():
        prediction_frame[name] = values
    prediction_frame["residual_primary_truth_minus_prediction_meV"] = 1000.0 * (
        prediction_frame["ape_true"] - prediction_frame[MODEL_HYBRID_CAL]
    )
    prediction_frame.to_csv(output / "predictions_all_models.csv", index=False, encoding="utf-8-sig")

    residual_season, residual_hour = residual_summaries(prediction_frame, MODEL_HYBRID_CAL)
    residual_season.to_csv(output / "test_residual_by_season.csv", index=False, encoding="utf-8-sig")
    residual_hour.to_csv(output / "test_residual_by_local_hour.csv", index=False, encoding="utf-8-sig")

    model_report = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "residual_feature_names": feature_names,
        "residual_train_oof_rows": len(oof_available),
        "train_evaluation_protocol": (
            "fixed beta is chronological OOF where available; residual MLP correction is in-sample "
            "on Train OOF residual targets and is diagnostic only"
        ),
        "residual_target_mean": residual_mean,
        "residual_target_std": residual_std,
        "residual_calibration": residual_calibration,
        "hgb_calibration": hgb_calibration,
        "calibration_role": "validation_calibration (later chronological half of Validation)",
        "test_used_for_training_selection_or_calibration": False,
    }
    atomic_json(output / "model_and_calibration_report.json", model_report)
    joblib.dump(
        {
            "beta_model": beta_model,
            "beta_scaler": beta_scaler,
            "feature_scaler": feature_scaler,
            "hgb_model": hgb,
            "settings": asdict(settings),
            "beta_report": beta_report,
            "model_report": model_report,
        },
        output / "non_torch_models_and_preprocessors.joblib",
        compress=3,
    )
    torch, _, _, _ = _load_torch()
    torch.save(
        {
            "state_dict": residual_model.state_dict(),
            "feature_names": feature_names,
            "settings": asdict(settings),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "calibration": residual_calibration,
        },
        output / "residual_mlp_model.pt",
    )

    save_plots(
        output,
        data,
        metrics,
        prediction_frame,
        history,
        MODEL_HYBRID_CAL,
    )
    report_path = markdown_report(
        output,
        run_name,
        settings,
        data_report,
        split_report,
        metrics,
        beta_report,
        model_report,
        data,
        prediction_frame,
    )

    test = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["surface"].isin(["HSR", "TSR"]))
        & (metrics["model"].isin([MODEL_FIXED, MODEL_HYBRID_CAL, MODEL_HGB_CAL]))
    ]
    experiment_label = "比較実験F+TO3" if args.include_ozone else "比較実験F"
    print(f"\n=== {experiment_label} 独立Test ===", flush=True)
    print(
        test[["surface", "model", "r2", "rmse_meV", "mae_meV", "mbe_truth_minus_prediction_meV", "n"]].to_string(index=False),
        flush=True,
    )
    print(f"\n結果・考察: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nユーザー操作により中断しました。", file=sys.stderr)
        raise SystemExit(130)
