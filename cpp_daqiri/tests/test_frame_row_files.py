#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HDF5 row orientation, calibration metadata and streaming conversion checks."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "scripts/conversion/flip_detector_rows.py"
spec = importlib.util.spec_from_file_location("flip_detector_rows", CONVERTER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RowFileTests(unittest.TestCase):
    def test_numpy_ranges_and_inverse(self):
        data = np.broadcast_to(np.arange(1024)[:, None], (1024, 3)).copy()
        for top in (False, True):
            for bottom in (False, True):
                expected = data.copy()
                if top:
                    expected[32:512] = data[32:512][::-1]
                if bottom:
                    expected[992:] = data[992:][::-1]
                got = module.flip_rows(data, top, bottom)
                np.testing.assert_array_equal(got, expected)
                np.testing.assert_array_equal(module.flip_rows(got, top, bottom), data)
                self.assertFalse(np.shares_memory(got, data))

    def test_all_calibration_datasets_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest, restored = [Path(tmp) / name for name in ("in.h5", "out.h5", "restored.h5")]
            values = np.broadcast_to(np.arange(1024, dtype=np.int16)[:, None], (1024, 3840)).copy()
            with h5py.File(source, "w") as file:
                file.attrs["note"] = "preserve root metadata"
                file.create_group("nested").attrs["note"] = "preserve group metadata"
                file.create_dataset("nested/values", data=[1, 2, 3])
                for name, data in (("frames", np.stack((values, values + 1))),
                                   ("processed", values.astype(np.float32)),
                                   ("dark_stddev", values.astype(np.float32) / 2),
                                   ("valid_pixel_mask", (values % 2).astype(np.uint8))):
                    ds = file.create_dataset(name, data=data, compression="gzip")
                    ds.attrs["note"] = name
            module.convert(source, dest, True, True)
            with h5py.File(source) as a, h5py.File(dest) as b:
                self.assertEqual(b.attrs["note"], a.attrs["note"])
                self.assertEqual(b["nested"].attrs["note"], a["nested"].attrs["note"])
                np.testing.assert_array_equal(b["nested/values"], a["nested/values"])
                for name in ("frames", "processed", "dark_stddev", "valid_pixel_mask"):
                    before, after = a[name], b[name]
                    self.assertEqual(before.dtype, after.dtype)
                    self.assertEqual(after.compression, "gzip")
                    self.assertEqual(module.read_orientation(after), (True, True))
                    frame_a = before[0] if before.ndim == 3 else before[...]
                    frame_b = after[0] if after.ndim == 3 else after[...]
                    np.testing.assert_array_equal(frame_b, module.flip_rows(frame_a, True, True))
            module.convert(dest, restored, True, True)
            with h5py.File(source) as a, h5py.File(restored) as b:
                for name in ("frames", "processed", "dark_stddev", "valid_pixel_mask"):
                    np.testing.assert_array_equal(b[name], a[name])
                    self.assertEqual(module.read_orientation(b[name]), (False, False))
            with self.assertRaises(FileExistsError):
                module.convert(source, dest, True, True)

    def test_dark_builder_retains_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, oriented, dark = [Path(tmp) / name for name in ("in.h5", "oriented.h5", "dark.h5")]
            values = np.broadcast_to(np.arange(1024, dtype=np.int16)[:, None], (1024, 3840))
            with h5py.File(source, "w") as file:
                file.create_dataset("frames", data=np.stack((values, values + 1)), compression="gzip")
            module.convert(source, oriented, True, True)
            subprocess.run([sys.executable, str(ROOT / "scripts/dark/make_dark_frame.py"),
                            str(oriented), str(dark), "--input-dataset", "/frames",
                            "--chunk-size", "1", "--blinker-std-threshold", "1000000"], check=True)
            with h5py.File(dark) as file:
                for name in ("processed", "raw_dark_mean", "dark_stddev", "blinker_mask", "valid_pixel_mask"):
                    self.assertEqual(module.read_orientation(file[name]), (True, True))
                np.testing.assert_array_equal(file["processed"][0],
                                              module.flip_rows(values.astype(np.float32) + 0.5, True, True))

    def test_invalid_orientation_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "in.h5", Path(tmp) / "out.h5"
            with h5py.File(source, "w") as file:
                ds = file.create_dataset("frames", shape=(1, 1024, 3840), dtype="int16")
                ds.attrs["flip_top_imaging_rows"] = 1
            with self.assertRaisesRegex(ValueError, "both row-orientation"):
                module.convert(source, output, True, False)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                module.convert(source, source, True, True)
            with self.assertRaises(ValueError):
                module.flip_rows(np.zeros((960, 1)), True, True)


if __name__ == "__main__":
    unittest.main()
