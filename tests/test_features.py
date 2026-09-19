import numpy as np
import pandas as pd

from ape_hybrid.features import add_observed_lags, beta_design, residual_static_features


def sample_frame():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T01:00Z"]
            ),
            "surface": ["HSR", "TSR", "HSR", "TSR"],
            "am": [1.1, 1.1, 1.2, 1.2],
            "omega": [20.0, 20.0, 21.0, 21.0],
            "aod550": [0.1, 0.1, 0.2, 0.2],
            "totangstr": [1.0, 1.0, 1.1, 1.1],
            "to3": [300.0, 300.0, 301.0, 301.0],
        }
    )


def test_beta_design_uses_surface_interactions():
    design = beta_design(sample_frame())
    assert design.loc[0, "TSR_label"] == 0.0
    assert design.loc[1, "TSR_label"] == 1.0
    assert design.loc[1, "TSR_x_AM"] == design.loc[1, "AM"]


def test_residual_lags_do_not_cross_surfaces():
    frame = sample_frame()
    static = residual_static_features(frame, np.full(len(frame), 1.8))
    lagged = add_observed_lags(frame, static, np.array([0.01, 0.02, 0.03, 0.04]), (1,))
    assert lagged.loc[2, "residual_lag_1h"] == 0.01
    assert lagged.loc[3, "residual_lag_1h"] == 0.02
