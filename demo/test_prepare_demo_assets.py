#!/usr/bin/env python3
"""Synthetic-data tests for the demo asset generator."""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class DemoAssetTest(unittest.TestCase):
    def test_generates_complete_asset_manifest_from_small_hdf5_bucket(self):
        try:
            import h5py
            import numpy as np
        except ImportError as error:
            self.skipTest("scientific demo dependencies unavailable: {}".format(error))
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is unavailable")

        from prepare_demo_assets import generate_assets

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.h5"
            dark_path = root / "dark.h5"
            output_path = root / "assets"
            config_path = root / "config.json"

            rows, columns, frames = 8, 12, 4
            raw = np.arange(frames * rows * columns, dtype=np.float32).reshape(
                frames, rows, columns
            )
            raw[:, 2, 3] += np.array([0, 1000, 0, 1000], dtype=np.float32)
            with h5py.File(raw_path, "w") as handle:
                handle.create_dataset("frames", data=raw)
            with h5py.File(dark_path, "w") as handle:
                handle.create_dataset(
                    "processed", data=np.zeros((1, rows, columns), np.float32)
                )
                valid = np.ones((1, rows, columns), np.uint8)
                valid[0, 1, 1] = 0
                handle.create_dataset("valid_pixel_mask", data=valid)

            config = {
                "raw_file": str(raw_path),
                "raw_dataset": "/frames",
                "dark_file": str(dark_path),
                "dark_dataset": "/processed",
                "valid_mask_dataset": "/valid_pixel_mask",
                "output_directory": str(output_path),
                "start_frame": 0,
                "bucket_size": frames,
                "representative_frame_index": 1,
                "chunk_size": 2,
                "detector": {"edge_rows": 1, "zlp_width": 4, "zlp_period": 2},
                "processor": {
                    "subtract_dark_frame": True,
                    "apply_valid_pixel_mask": True,
                    "apply_blr_correction": True,
                    "blr_rows": 1,
                    "blr_zlp_width": 4,
                    "blr_zlp_group_columns": 2,
                    "blr_core_group_columns": 2,
                    "apply_dynamic_half_column_mask": True,
                    "dynamic_mask_median_window_pixels": 3,
                    "dynamic_mask_threshold_ratio": 1.0,
                    "dynamic_mask_threshold_offset": 100.0,
                    "dynamic_mask_excluded_edge_rows": 1,
                    "dynamic_mask_two_sided": True,
                },
                "performance": {"measured_input_gbps": 1.0},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            generated = generate_assets(config_path, force=True)

            expected = {
                "raw_frame.png",
                "dark_subtracted_frame.png",
                "blr_corrected_frame.png",
                "mask_overlay.png",
                "corrected_frame.png",
                "bucket_sum.png",
                "spectrum_linear.png",
                "spectrum_log.png",
                "spectrum_data.npz",
                "demo_metadata.json",
            }
            self.assertEqual(expected, {path.name for path in generated.iterdir()})
            metadata = json.loads((generated / "demo_metadata.json").read_text())
            self.assertEqual(metadata["detector_shape"], [rows, columns])
            self.assertEqual(metadata["source"]["bucket_size"], frames)
            self.assertEqual(metadata["spectrum"]["channels"], 10)
            self.assertGreaterEqual(metadata["mask"]["static_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
