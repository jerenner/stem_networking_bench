import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from eels_sim.dark_calibration import calibrate_dark_stack, detect_storage_lsb
from eels_sim.io import read_readout_calibration


class DarkCalibrationTests(unittest.TestCase):
    def test_lsb_detection_and_held_out_maps(self):
        rng = np.random.default_rng(4)
        pedestal = np.arange(48, dtype=np.float32).reshape(6, 8) + 1000.0
        frames = []
        for index in range(12):
            common = 2.0 if index >= 6 else 0.0
            codes = np.rint(pedestal + common + rng.normal(0.0, 1.0, pedestal.shape))
            frames.append(codes * 64.0)
        frames = np.asarray(frames, dtype=np.float32)
        frames[:, 2, 3] += rng.normal(0.0, 1000.0, size=len(frames))

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "dark.h5"
            output_path = Path(directory) / "calibration.h5"
            with h5py.File(input_path, "w") as h5:
                dataset = h5.create_dataset("frames", data=frames)
                self.assertEqual(detect_storage_lsb(dataset), 64)
            summary = calibrate_dark_stack(
                input_path,
                output_path,
                calibration_frames=6,
                validation_frames=6,
                max_noise_raw=500.0,
                batch_frames=2,
                row_stride=1,
                column_stride=1,
                write_plot=False,
                known_defect_regions=[(0, 2, 4, 6)],
            )
            calibration = read_readout_calibration(output_path)
            self.assertEqual(calibration["pedestal_adu"].shape, (6, 8))
            self.assertFalse(calibration["valid_pixel_mask"][2, 3])
            self.assertTrue(np.all(calibration["known_defect_mask"][0:2, 4:6]))
            self.assertFalse(np.any(calibration["known_defect_mask"][2:, :]))
            self.assertTrue(np.all(calibration["signal_efficiency"][0:2, 4:6] == 0.0))
            self.assertTrue(np.all(np.isfinite(calibration["healthy_pedestal_adu"])))
            self.assertEqual(summary["known_defect_pixels"], 4)
            self.assertGreater(summary["validation_bias_median_adu"], 1.0)


if __name__ == "__main__":
    unittest.main()
