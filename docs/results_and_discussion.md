# Results and discussion

## Main test result

The calibrated hybrid reached R2 = 0.6567 for HSR and 0.7795 for TSR. Relative to fixed beta, the gain was +0.1109 for HSR and +0.1301 for TSR. The residual MLP accounts for most of that gain.

Affine calibration improved R2 and RMSE for both surfaces. Its effect was metric-dependent: HSR MAE changed from 10.3222 meV before calibration to 10.6759 meV after calibration, whereas TSR MAE improved from 10.5333 to 10.3116 meV. It should therefore be described as a bias/scale tradeoff, not a universal accuracy gain.

## TO3 contribution

With rows, splits, and targets fixed, adding TO3 improved R2 by +0.0142 for HSR and +0.0104 for TSR. Paired day-bootstrap 95% intervals were +0.0027 to +0.0257 and +0.0052 to +0.0158, respectively.

The increments are small and the experiment used one training seed. TO3 may contain residual seasonal or spectral information, but the result does not support a causal interpretation of the fitted ozone coefficient.

## High-APE regime

For the top 10% of measured APE, mean truth-minus-prediction residual was +21.74 meV for HSR and +18.74 meV for TSR. TO3 did not remove this regression-to-the-mean behavior. Additional cloud/scattering information, incidence geometry, or more direct short-wavelength descriptors are plausible next features.

## Surface dependence

TSR was consistently easier to predict in R2 terms than HSR. The shared-plus-interaction linear design lets both surfaces use the same atmospheric backbone while retaining surface-specific slopes. This is preferable to interpreting two separately trained coefficient sets as directly comparable physical responses.
