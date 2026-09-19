# Model philosophy

## Preserve a legible baseline

The fixed-beta branch captures the large-scale relationship between APE, atmospheric state, and receiving surface. Its coefficients remain constant over time after training. This makes the baseline inspectable, but not causal: Ridge coefficients can reflect collinearity, seasonal structure, and site-specific sampling.

## Learn only what the baseline misses

The MLP targets the out-of-fold residual of the linear branch rather than replacing it. This separates the roles of the two components:

- the linear branch expresses a compact global tendency;
- the nonlinear branch models interactions, periodic structure, and recent residual dynamics;
- the final calibration corrects validation-observed amplitude and offset.

The decomposition does not guarantee physical identifiability. It is a modeling constraint intended to preserve interpretability while retaining nonlinear capacity.

## Keep calibration honest

Early stopping and affine calibration use different chronological portions of validation. A six-hour purge matches the longest residual lag. Test labels are unavailable until final scoring.

## Treat transfer as a hypothesis

A transparent baseline may be easier to transport than an unconstrained black-box model, but both the residual learner and calibration were estimated from Tsukuba observations. Claims about regions without spectral labels require external-site validation.
