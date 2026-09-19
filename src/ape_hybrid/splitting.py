"""Chronological split helpers with explicit purge gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_roles(
    timestamps,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.50,
    purge_hours: int = 6,
) -> dict[str, pd.DatetimeIndex]:
    """Split unique timestamps without separating surfaces at the same time."""
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).unique().sort_values()
    if len(times) < 100:
        raise ValueError("Too few unique timestamps for a four-role chronological split")
    train_end = int(np.floor(len(times) * train_fraction))
    validation_end = int(np.floor(len(times) * (train_fraction + validation_fraction)))
    validation = times[train_end:validation_end]
    calibration_count = int(np.floor(len(validation) * calibration_fraction))
    calibration_start = validation_end - calibration_count
    delta = pd.Timedelta(hours=purge_hours)

    train = times[:train_end]
    validation_early = times[train_end:calibration_start]
    validation_calibration = times[calibration_start:validation_end]
    test = times[validation_end:]

    if len(validation_early):
        validation_early = validation_early[validation_early > train.max() + delta]
    if len(validation_calibration) and len(validation_early):
        validation_early = validation_early[
            validation_early < validation_calibration.min() - delta
        ]
    if len(test):
        test = test[test > times[validation_end - 1] + delta]

    roles = {
        "train_oof": train,
        "validation_early": validation_early,
        "validation_calibration": validation_calibration,
        "test": test,
    }
    if any(len(values) == 0 for values in roles.values()):
        raise ValueError("A chronological role became empty")
    return roles
