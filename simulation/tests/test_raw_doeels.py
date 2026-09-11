import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from eels_sim.raw_doeels import (
    RAW_DOEELS_SCHEMA,
    RESPONSE_KERNEL_SCHEMA,
    _sample_detector_charge,
    simulate_raw_doeels_scan,
)


class RawDOEEELSTests(unittest.TestCase):
    def test_exact_event_templates_conserve_interior_charge(self):
        templates = np.zeros((2, 3, 3), dtype=np.float64)
        templates[0, 1, 1] = 5.0
        templates[1, 1, 1] = 6.0
        templates[1, 1, 2] = 1.0
        repeated_pixels = np.array([4 * 9 + 4] * 12, dtype=int)
        charge, mode = _sample_detector_charge(
            repeated_pixels,
            (9, 9),
            templates.sum(axis=(1, 2)),
            templates,
            templates.sum(axis=0) / templates.sum(),
            100,
            np.random.default_rng(7),
        )
        self.assertEqual(mode, "exact_event_template")
        self.assertGreaterEqual(charge.sum(), 12 * 5)
        self.assertLessEqual(charge.sum(), 12 * 7)
        self.assertEqual(np.count_nonzero(charge), 2)

    def test_small_raw_scan_writes_frames_truth_and_stage_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path = root / "response.h5"
            spectrum_path = root / "spectrum.h5"
            config_path = root / "raw.toml"
            output_path = root / "raw.h5"
            templates = np.zeros((2, 3, 3), dtype=np.uint32)
            templates[0, 1, 1] = 5
            templates[1, 1, 1] = 6
            templates[1, 1, 2] = 1
            totals = templates.sum(axis=(1, 2))
            with h5py.File(response_path, "w") as h5:
                h5.attrs["schema"] = RESPONSE_KERNEL_SCHEMA
                h5.attrs["summary_json"] = json.dumps({"event_count": 2})
                h5.create_dataset("event_total_pairs", data=totals)
                h5.create_dataset("event_spatial_pairs", data=templates)
                h5.create_dataset(
                    "mean_spatial_fraction",
                    data=templates.sum(axis=0) / templates.sum(),
                )

            component_profiles = np.zeros((3, 16), dtype=np.float64)
            component_profiles[0, 1] = 1.0
            component_profiles[1, 4:7] = 1.0 / 3.0
            component_profiles[2, 10:13] = 1.0 / 3.0
            component_totals = np.zeros((1, 2, 2, 3), dtype=np.uint32)
            component_totals[..., 0] = 80
            component_totals[..., 1] = 15
            component_totals[..., 2] = np.array([[[5, 6], [7, 8]]])
            component_totals[..., 0] -= component_totals[..., 2] - 5
            counts = np.einsum("dyxc,ce->dyxe", component_totals, component_profiles).astype(
                np.uint32
            )
            with h5py.File(spectrum_path, "w") as h5:
                h5.attrs["schema"] = "eels-sim-spectrum-image-v1"
                h5.attrs["summary_json"] = json.dumps({"selected_element_edges": {"Ti": ["Ti_K"]}})
                axes = h5.create_group("axes")
                axes.create_dataset("energy_edges_eV", data=np.arange(17))
                axes.create_dataset("energy_centers_eV", data=np.arange(16) + 0.5)
                axes.create_dataset("focal_depth_A", data=[0.0])
                axes.create_dataset("scan_y_A", data=[0.0, 1.0])
                axes.create_dataset("scan_x_A", data=[0.0, 1.0])
                spectrum = h5.create_group("spectrum")
                spectrum.attrs["component_labels_json"] = json.dumps(
                    ["zero_loss", "low_loss", "Ti_K"]
                )
                spectrum.create_dataset("counts", data=counts)
                spectrum.create_dataset("component_profiles", data=component_profiles)
                spectrum.create_dataset("component_total_counts", data=component_totals)

            config_path.write_text(
                f"""[experiment]
beam_energy_keV = 300.0
frame_rate_hz = 87000.0
beam_current_pA = 30.0

[readout]
gain_adu_per_electron = 1.0
pedestal_adu = 10.0
read_noise_adu = 0.0
adc_bits = 16

[spectrometer]
dispersion_eV_per_column = 1.0
zero_loss_stitched_column = 2.0
detector_rows = 8
detector_columns = 32
zero_y_row = 3.5
zlp_repeats = 2
zlp_lane_width_columns = 4

[raw_doeels]
response_kernel_file = "{response_path}"
electrons_per_integration = 20
depth_indices = [0]
detector_row_sigma_pixels = 1.2
energy_mapping_subsamples = 4
exact_response_max_electrons = 100
response_blur_sigma_pixels = 0.5
keep_hit_counts = true
random_seed = 12
""",
                encoding="utf-8",
            )
            summary = simulate_raw_doeels_scan(config_path, spectrum_path, output_path)
            with h5py.File(output_path, "r") as h5:
                self.assertEqual(h5.attrs["schema"], RAW_DOEELS_SCHEMA)
                self.assertEqual(h5["frames/raw"].shape, (4, 8, 32))
                np.testing.assert_array_equal(
                    h5["truth/incident_spectrum_counts"][:].sum(axis=1), 20
                )
                self.assertEqual(
                    h5["reconstruction/scan_maps/expected_component_counts"].shape,
                    (1, 2, 2, 3),
                )
                self.assertIn("Ti", h5["reconstruction/scan_maps/elements"])
                self.assertTrue(np.isfinite(h5["reconstruction/component_counts"][:]).all())
                self.assertTrue(
                    np.isfinite(h5["reconstruction/pre_readout_component_counts"][:]).all()
                )
            self.assertEqual(summary["frame_count"], 4)
            self.assertEqual(summary["detector_response_modes"], ["exact_event_template"])
            self.assertEqual(summary["response_blur_sigma_pixels"], 0.5)


if __name__ == "__main__":
    unittest.main()
