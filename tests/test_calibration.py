import numpy as np

from ape_hybrid.calibration import AffineResidualCalibrator


def test_affine_calibration_recovers_scale_and_offset():
    predicted = np.linspace(-0.01, 0.01, 40)
    truth = 0.8 * predicted + 0.002
    calibrator = AffineResidualCalibrator().fit(predicted, truth)
    assert np.isclose(calibrator.slope_, 0.8)
    assert np.isclose(calibrator.intercept_, 0.002)
    assert calibrator.n_samples_ == 40


def test_slope_is_bounded():
    predicted = np.linspace(-0.01, 0.01, 40)
    truth = 3.0 * predicted
    calibrator = AffineResidualCalibrator().fit(predicted, truth)
    assert calibrator.slope_ == 1.5
