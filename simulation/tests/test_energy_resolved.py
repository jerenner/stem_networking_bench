import tempfile
import unittest
from pathlib import Path

import numpy as np
from eels_sim.energy_resolved import EnergyResolvedSpecimenKernel
from eels_sim.materialize import materialize_phase_space
from eels_sim.specimen import run_specimen_kernel
from eels_sim.spectral_library import (
    CoreLossComponent,
    LowLossComponent,
    SpectralDistribution,
    SpectralLibrary,
    read_spectral_library,
    write_spectral_library,
)


def incident_electrons(count: int) -> dict[str, np.ndarray]:
    return {
        "frame_id": np.arange(count, dtype=np.uint64) // 100,
        "electron_id": np.arange(count, dtype=np.uint64),
        "x_um": np.zeros(count),
        "y_um": np.zeros(count),
        "z_um": np.zeros(count),
        "dir_x": np.zeros(count),
        "dir_y": np.zeros(count),
        "dir_z": np.ones(count),
        "kinetic_energy_eV": np.full(count, 300_000.0),
        "time_ns": np.zeros(count),
        "weight": np.ones(count),
        "loss_channel": np.zeros(count, dtype=np.int32),
    }


def toy_library() -> SpectralLibrary:
    return SpectralLibrary(
        material="toy",
        beam_energy_eV=300_000.0,
        specimen_thickness_A=100.0,
        elastic_angular_sigma_mrad=0.1,
        low_loss=LowLossComponent(
            SpectralDistribution(np.array([9.9, 10.1]), np.array([1.0, 1.0])),
            mean_events=0.5,
            angular_sigma_mrad=0.2,
        ),
        core_loss=(
            CoreLossComponent(
                name="X_K",
                element="X",
                edge="K",
                onset_energy_eV=99.9,
                integrated_probability=0.2,
                angular_sigma_mrad=0.5,
                distribution=SpectralDistribution(np.array([99.9, 100.1]), np.array([1.0, 1.0])),
            ),
        ),
        metadata={"purpose": "unit test"},
    )


class EnergyResolvedTests(unittest.TestCase):
    def test_library_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.h5"
            write_spectral_library(path, toy_library())
            restored = read_spectral_library(path)
        self.assertEqual(restored.material, "toy")
        self.assertAlmostEqual(restored.low_loss.mean_events, 0.5)
        self.assertEqual(restored.core_loss[0].name, "X_K")
        self.assertAlmostEqual(
            np.trapezoid(
                restored.core_loss[0].distribution.probability_density_per_eV,
                restored.core_loss[0].distribution.energy_loss_eV,
            ),
            1.0,
        )

    def test_kernel_samples_one_electron_per_parent_and_plural_losses(self):
        count = 20_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.h5"
            write_spectral_library(path, toy_library())
            result = run_specimen_kernel(
                EnergyResolvedSpecimenKernel(),
                incident_electrons(count),
                reference_energy_eV=300_000.0,
                parameters={"library_file": str(path)},
                random_seed=41,
            )
        self.assertEqual(result.weight_semantics, "individual_electrons")
        self.assertEqual(len(result.electrons["electron_id"]), count)
        np.testing.assert_array_equal(result.electrons["parent_electron_id"], np.arange(count))
        self.assertTrue(np.all(result.electrons["weight"] == 1.0))
        self.assertAlmostEqual(np.mean(result.electrons["loss_channel"] == 3), 0.2, delta=0.015)
        self.assertAlmostEqual(np.mean(result.electrons["plural_order"]), 0.7, delta=0.025)
        self.assertTrue(np.any(result.electrons["plural_order"] >= 2))
        np.testing.assert_allclose(result.electrons["z_um"], 0.01)
        direction_norm = np.sqrt(
            result.electrons["dir_x"] ** 2
            + result.electrons["dir_y"] ** 2
            + result.electrons["dir_z"] ** 2
        )
        np.testing.assert_allclose(direction_norm, 1.0)

    def test_passthrough_preserves_sampled_component_fields(self):
        electrons = incident_electrons(4)
        electrons["spectral_component_id"] = np.array([0, 1, 2, 2], dtype=np.uint16)
        electrons["plural_order"] = np.array([0, 1, 1, 2], dtype=np.uint16)
        output, summary = materialize_phase_space(
            electrons, mode="passthrough", random_seed=5, max_electrons=10
        )
        self.assertEqual(summary["output_electrons"], 4)
        np.testing.assert_array_equal(
            output["spectral_component_id"], electrons["spectral_component_id"]
        )
        np.testing.assert_array_equal(output["plural_order"], electrons["plural_order"])


if __name__ == "__main__":
    unittest.main()
