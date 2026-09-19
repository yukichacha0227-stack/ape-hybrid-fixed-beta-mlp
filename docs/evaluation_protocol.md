# Evaluation protocol

## Chronological roles

There are 12,181 common hourly timestamps and 24,362 surface rows from June 2011 through December 2015.

| Role | Unique hours | Surface rows |
|---|---:|---:|
| Train / OOF | 8,526 | 17,052 |
| Validation for early stopping | 905 | 1,810 |
| Validation for calibration | 910 | 1,820 |
| Independent test | 1,822 | 3,644 |

HSR and TSR at the same timestamp always belong to the same role. Six-hour purge gaps are used at boundaries because the longest residual lag is six hours.

## Reported metrics

- R2
- RMSE in meV
- MAE in meV
- mean bias error defined as truth minus prediction

Only the independent test values support generalization claims. Calibration-set metrics are in-sample and are not headline results.

## TO3 comparison

The F and F+TO3 experiments use identical timestamps, targets, and split labels. Day-block paired bootstrap resamples the 236 JST test days 2,000 times. This quantifies uncertainty over test-day composition, not variation from retraining the neural network with different seeds.

## Non-controlled comparison

The three-feature demonstration and the final five-atmospheric-variable model do not differ only by features. They also differ in joint versus separate surface training, recursive residual lags, engineered features, and random seed. Their performance gap is descriptive, not a controlled feature ablation.
