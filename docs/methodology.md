# Methodology

## Target and alignment

- Target: APE derived independently for HSR and 32-degree TSR spectra over 350-1050 nm.
- Instantaneous valid APE samples are arithmetically averaged within each JST hour.
- At least three valid samples are required for a surface-hour.
- The hourly target is assigned to `hh:30 JST`, converted to UTC, and exactly joined to the MERRA-2 `tavg1` center time.
- No nearest-time join or target replication is used.

## Atmospheric inputs

The fixed-beta branch uses air mass (`AM`), total precipitable water (`TQV`, named `omega` in the historical implementation), aerosol optical depth at 550 nm (`AOD550`), Angstrom exponent (`TOTANGSTR`), total-column ozone (`TO3`), a TSR label, and surface interactions.

The residual MLP additionally uses tilt angle, the beta prediction, local-hour and day-of-year cycles, selected interactions, a transmittance proxy, and predicted residual lags at 1, 2, 3, and 6 hours.

## Training

1. Fit a standardized Ridge model on the training period.
2. Generate five-fold time-series out-of-fold beta predictions with a six-hour gap.
3. Train the MLP on out-of-fold residual targets with AdamW and Huber loss.
4. Select the MLP epoch on the earlier validation interval.
5. Roll residual predictions forward causally by surface for validation and test.
6. Fit surface-specific affine residual calibration on the later validation interval.
7. Score the untouched chronological test period.

The MLP has hidden widths 128, 128, and 64 with LayerNorm, GELU, and 0.05 dropout. The maximum is 300 epochs with patience 100.

## Important rollout detail

Training residual lags use observed out-of-fold residuals, while validation and test use previously predicted residuals. This prevents test-label leakage but introduces a teacher-forcing-versus-rollout mismatch that should be considered when interpreting lag utility.
