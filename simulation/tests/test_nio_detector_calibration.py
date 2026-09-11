import unittest

import numpy as np
from eels_sim.nio_detector_calibration import (
    _fit_response,
    _histogram_subtracted_quantiles,
)


class NiODetectorCalibrationTests(unittest.TestCase):
    def test_dark_subtracted_quantiles_remove_scaled_background(self):
        signal = np.concatenate((np.linspace(40.0, 100.0, 1000), np.full(100, 20.0)))
        dark = np.full(50, 20.0)
        quantiles, _, _ = _histogram_subtracted_quantiles(
            signal,
            10,
            dark,
            5,
            np.asarray([0.5]),
            (0.0, 120.0),
            bins=240,
        )
        self.assertAlmostEqual(float(quantiles[0]), 70.0, delta=1.0)

    def test_response_fit_recovers_effective_gain_and_blur(self):
        rng = np.random.default_rng(11)
        templates = np.zeros((300, 7, 7), dtype=np.float64)
        totals = rng.lognormal(np.log(450.0), 0.35, size=len(templates))
        templates[:, 3, 3] = totals
        measured_peak = np.asarray([48.0, 58.0, 76.0, 105.0, 145.0])
        measured_sum = np.asarray([70.0, 88.0, 120.0, 165.0, 230.0])
        fit = _fit_response(
            templates,
            measured_peak,
            measured_sum,
            np.asarray([0.10, 0.25, 0.50, 0.75, 0.90]),
            noise_sigma_adu=4.0,
            threshold_adu=35.0,
            xray_threshold_adu=800.0,
            random_seed=9,
        )
        self.assertGreater(fit["gain_adu_per_pair"], 0.05)
        self.assertLess(fit["gain_adu_per_pair"], 0.60)
        self.assertGreaterEqual(fit["additional_blur_sigma_pixels"], 0.0)
        self.assertLessEqual(fit["additional_blur_sigma_pixels"], 1.2)
        self.assertGreater(fit["detection_efficiency_above_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
