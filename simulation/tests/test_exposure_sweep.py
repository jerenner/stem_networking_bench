import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from eels_sim.exposure_sweep import (
    EXPOSURE_SWEEP_SCHEMA,
    simulate_exposure_sweep,
    write_exposure_slider_demo,
)
from eels_sim.raw_doeels import RESPONSE_KERNEL_SCHEMA


class ExposureSweepTests(unittest.TestCase):
    def test_nested_sweep_retains_compact_spectra_not_raw_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path = root / "response.h5"
            spectrum_path = root / "spectrum.h5"
            config_path = root / "sweep.toml"
            output_path = root / "sweep.h5"

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

            profiles = np.zeros((3, 16), dtype=np.float64)
            profiles[0, 1] = 1.0
            profiles[1, 4:7] = 1.0 / 3.0
            profiles[2, 10:13] = 1.0 / 3.0
            totals_by_position = np.zeros((1, 2, 2, 3), dtype=np.uint32)
            totals_by_position[..., 0] = 80
            totals_by_position[..., 1] = 15
            totals_by_position[..., 2] = np.array([[[5, 6], [7, 8]]])
            totals_by_position[..., 0] -= totals_by_position[..., 2] - 5
            counts = np.einsum("dyxc,ce->dyxe", totals_by_position, profiles).astype(np.uint32)
            with h5py.File(spectrum_path, "w") as h5:
                h5.attrs["schema"] = "eels-sim-spectrum-image-v1"
                h5.attrs["summary_json"] = json.dumps({"selected_element_edges": {"Ti": ["Ti_K"]}})
                axes = h5.create_group("axes")
                axes.create_dataset("energy_edges_eV", data=np.arange(17))
                axes.create_dataset("focal_depth_A", data=[0.0])
                axes.create_dataset("scan_y_A", data=[0.0, 1.0])
                axes.create_dataset("scan_x_A", data=[0.0, 1.0])
                spectrum = h5.create_group("spectrum")
                spectrum.attrs["component_labels_json"] = json.dumps(
                    ["zero_loss", "low_loss", "Ti_K"]
                )
                spectrum.create_dataset("counts", data=counts)
                spectrum.create_dataset("component_profiles", data=profiles)
                spectrum.create_dataset("component_total_counts", data=totals_by_position)
                truth = h5.create_group("truth")
                truth.create_dataset(
                    "haadf_expected_counts",
                    data=np.array([[[3.0, 4.0], [5.0, 6.0]]]),
                )

            config_path.write_text(
                f"""[experiment]
beam_energy_keV = 300.0
frame_rate_hz = 1000.0
beam_current_pA = 1.0

[readout]
gain_adu_per_electron = 1.0
pedestal_adu = 10.0
read_noise_adu = 0.5
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
random_seed = 12
""",
                encoding="utf-8",
            )

            summary = simulate_exposure_sweep(
                config_path,
                spectrum_path,
                output_path,
                exposures=[4, 1],
            )
            with h5py.File(output_path, "r") as h5:
                self.assertEqual(h5.attrs["schema"], EXPOSURE_SWEEP_SCHEMA)
                np.testing.assert_array_equal(h5["axes/integrations_per_position"][:], [1, 4])
                self.assertNotIn("frames", h5)
                incident = h5["truth/incident_spectrum_counts"][:]
                np.testing.assert_array_equal(incident.sum(axis=-1)[0], 20)
                np.testing.assert_array_equal(incident.sum(axis=-1)[1], 80)
                self.assertTrue(np.all(incident[1] >= incident[0]))
                expected = h5["truth/expected_component_counts"][:]
                np.testing.assert_allclose(expected[1], 4.0 * expected[0])
                self.assertIn("Ti", h5["reconstruction/elements"])
                self.assertIn("counted_counts", h5["reconstruction/elements/Ti"])
                self.assertEqual(
                    h5["reconstruction/energy_counts"].shape,
                    (2, 1, 2, 2, 16),
                )
                self.assertEqual(
                    h5["reconstruction/counted_energy_counts"].shape,
                    (2, 1, 2, 2, 16),
                )
            self.assertEqual(summary["raw_frames_retained"], 0)
            self.assertEqual(summary["scan_shape"], [1, 2, 2])

            html_path = root / "slider.html"
            png_path = root / "slider.png"
            demo = write_exposure_slider_demo(
                output_path,
                html_path,
                png_path,
                elements=("Ti",),
                display_sigma_pixels=0.0,
            )
            self.assertEqual(demo["channels"], ["HAADF", "Ti"])
            self.assertTrue(png_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Monte Carlo silicon charge-cloud response", html)
            self.assertIn("counted fit", html)


if __name__ == "__main__":
    unittest.main()
