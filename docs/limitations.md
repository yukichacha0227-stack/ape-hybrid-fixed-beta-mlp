# Limitations

1. **Single site.** All headline results come from Tsukuba. Geographic transfer has not been tested.
2. **Single neural-network seed for the TO3 ablation.** The paired bootstrap resamples test days but does not measure retraining variance.
3. **High-APE underprediction.** The upper target decile remains systematically underestimated.
4. **Hourly target definition.** Arithmetic averaging of instantaneous APE is not identical to computing APE after integrating the numerator and denominator spectra over the hour.
5. **Lag training mismatch.** Training uses observed OOF residual lags; deployment uses recursively predicted lags.
6. **Associational coefficients.** Fixed-beta coefficients are regularized regression estimates, not physical constants or causal effects.
7. **Confounded three-feature comparison.** The conference before/after scatter plots are not a strict nested-feature ablation.
8. **Data redistribution.** Raw spectrum and observation files are omitted until their redistribution conditions are confirmed.

The next strongest experiment is a same-pipeline, same-split, multi-seed nested ablation: `AM/TQV/AOD550`, then `+TOTANGSTR`, then `+TO3`, followed by evaluation at at least one external site.
