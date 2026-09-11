import tempfile
import unittest
from pathlib import Path

import numpy as np
from eels_sim.config import SpectrometerConfig
from eels_sim.phase_space import read_phase_space_hdf5, write_phase_space_hdf5
from eels_sim.spectrometer import transfer_phase_space, write_geant4_source_csv


def phase_space(
    energies_eV: list[float],
    theta_x_mrad: list[float] | None = None,
    theta_y_mrad: list[float] | None = None,
) -> dict[str, np.ndarray]:
    count = len(energies_eV)
    theta_x = np.zeros(count) if theta_x_mrad is None else np.asarray(theta_x_mrad)
    theta_y = np.zeros(count) if theta_y_mrad is None else np.asarray(theta_y_mrad)
    slopes = np.column_stack((theta_x, theta_y)) * 1.0e-3
    directions = np.column_stack((slopes, np.ones(count)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return {
        "frame_id": np.arange(count, dtype=np.uint64) // 2,
        "electron_id": np.arange(count, dtype=np.uint64),
        "x_um": np.zeros(count),
        "y_um": np.zeros(count),
        "z_um": np.zeros(count),
        "dir_x": directions[:, 0],
        "dir_y": directions[:, 1],
        "dir_z": directions[:, 2],
        "kinetic_energy_eV": np.asarray(energies_eV),
        "time_ns": np.arange(count, dtype=float),
        "weight": np.ones(count),
        "loss_channel": np.zeros(count, dtype=np.int32),
    }


class PhaseSpaceTests(unittest.TestCase):
    def test_hdf5_round_trip(self):
        electrons = phase_space([100.0, 99.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_space.h5"
            write_phase_space_hdf5(path, electrons, {"purpose": "test"})
            restored = read_phase_space_hdf5(path)
        for name, values in electrons.items():
            np.testing.assert_array_equal(restored[name], values)

    def test_four_zlp_lanes_and_core_offset(self):
        electrons = phase_space([100.0, 100.0, 100.0, 100.0, 97.0])
        config = SpectrometerConfig(
            dispersion_eV_per_column=1.0,
            zero_loss_stitched_column=1.0,
            detector_rows=4,
            detector_columns=20,
            pixel_pitch_um=1.0,
            zero_y_row=1.5,
            maximum_energy_loss_eV=10.0,
            zlp_repeats=4,
            zlp_lane_width_columns=4,
            random_seed=7,
        )
        entries, diagnostics, summary = transfer_phase_space(
            electrons, config, reference_energy_eV=100.0
        )
        self.assertEqual(summary["accepted_electrons"], 5)
        np.testing.assert_allclose(entries["stitched_column"], [1, 1, 1, 1, 4])
        np.testing.assert_allclose(entries["raw_column"][:4] % 4, 1)
        self.assertTrue(np.all((entries["zlp_lane"][:4] >= 0)))
        self.assertEqual(entries["raw_column"][4], 16.0)
        self.assertEqual(entries["zlp_lane"][4], -1)
        np.testing.assert_array_equal(diagnostics["rejection_code"], 0)

    def test_position_angle_energy_mapping_and_acceptance(self):
        electrons = phase_space(
            [100.0, 98.0, 100.0],
            theta_x_mrad=[0.0, 1.0, 60.0],
            theta_y_mrad=[0.0, 2.0, 0.0],
        )
        electrons["x_um"][1] = 2.0
        electrons["y_um"][1] = 3.0
        config = SpectrometerConfig(
            dispersion_eV_per_column=1.0,
            zero_loss_stitched_column=5.0,
            detector_rows=40,
            detector_columns=30,
            pixel_pitch_um=1.0,
            zero_y_row=19.5,
            x_magnification=2.0,
            y_magnification=2.0,
            x_angle_to_position_um_per_mrad=3.0,
            y_angle_to_position_um_per_mrad=4.0,
            collection_semiangle_mrad=50.0,
            maximum_energy_loss_eV=10.0,
        )
        entries, diagnostics, summary = transfer_phase_space(
            electrons, config, reference_energy_eV=100.0
        )
        self.assertEqual(summary["accepted_electrons"], 2)
        self.assertAlmostEqual(entries["stitched_column"][1], 14.0)
        self.assertAlmostEqual(entries["raw_row"][1], 33.5)
        self.assertEqual(diagnostics["rejection_code"][2], 2)

    def test_geant4_csv_rejects_statistical_weights(self):
        entries = {
            name: values
            for name, values in {
                "event_id": np.array([0]),
                "frame_id": np.array([0]),
                "electron_id": np.array([0]),
                "x_um": np.array([0.0]),
                "y_um": np.array([0.0]),
                "z_um": np.array([-5.0]),
                "dir_x": np.array([0.0]),
                "dir_y": np.array([0.0]),
                "dir_z": np.array([1.0]),
                "kinetic_energy_eV": np.array([100.0]),
                "time_ns": np.array([0.0]),
                "weight": np.array([2.0]),
                "loss_channel": np.array([0]),
            }.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "individual electrons"):
                write_geant4_source_csv(Path(directory) / "source.csv", entries)


if __name__ == "__main__":
    unittest.main()
