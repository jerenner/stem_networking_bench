"""Host-only tests of the independent GPU-validation reference and log reader."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_synthetic_rx.py"
SPEC = importlib.util.spec_from_file_location("validate_synthetic_rx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReferenceTests(unittest.TestCase):
    def test_native_coverage_and_coordinates(self):
        frame = MODULE.reference_frame(0, 0)
        self.assertEqual(frame.dtype, np.uint16)
        self.assertTrue((frame >= 64).all())
        self.assertEqual(frame[0, 0], 64)
        self.assertEqual(frame[0, 32], 64 + 17)  # Tile 1.
        self.assertEqual(frame[0, 768], 64 + (192 * 17) % 4096)
        self.assertEqual(frame[1023, 3839], 64 + (959 * 17 + 4095 * 5) % 4096)

    def test_legacy_prefix_and_sparse_mask(self):
        frame = MODULE.reference_frame(0, 0, mask=13, legacy=True)
        # Final eight rows of ZLP tile 0 repeat its first eight rows.
        np.testing.assert_array_equal(frame[120:128, :32], frame[:8, :32])
        self.assertTrue((frame[224:, 768:] == 0).all())  # Tile 360 onward absent.
        self.assertEqual(np.count_nonzero(frame), 3 * 120 * 4096)

    def test_wrap_and_receiver_independence(self):
        np.testing.assert_array_equal(MODULE.reference_frame(0, 128), MODULE.reference_frame(0, 0))
        self.assertFalse(np.array_equal(MODULE.reference_frame(1, 0), MODULE.reference_frame(0, 0)))

    def test_walking_dot(self):
        frame = MODULE.reference_frame(0, 0, pattern="walking_dot")
        self.assertEqual(np.count_nonzero(frame == 20000), 960)
        self.assertEqual(frame[0, 0], 20000)

    def test_result_parser(self):
        result = MODULE.parse_results("log\nsynthetic complete rx=0 frames=6 mismatches=0\n")
        self.assertEqual(result[0]["frames"], "6")
        with self.assertRaises(AssertionError):
            MODULE.parse_results("synthetic complete rx=0\nsynthetic complete rx=0\n")


if __name__ == "__main__":
    unittest.main()
