import tempfile
import unittest
from pathlib import Path

import numpy as np
from eels_sim.abtem_adapter import (
    AbTEMScatteringModel,
    AbTEMSpecimenKernel,
    AngularDistribution,
)
from eels_sim.config import SpectrometerConfig
from eels_sim.materialize import materialize_phase_space
from eels_sim.phase_space import read_phase_space_hdf5, write_phase_space_hdf5
from eels_sim.specimen import SpecimenResult, load_specimen_kernel, run_specimen_kernel
from eels_sim.spectrometer import transfer_phase_space


def individual_electrons(count: int = 4) -> dict[str, np.ndarray]:
    return {
        "frame_id": np.arange(count, dtype=np.uint64) // 2,
        "electron_id": np.arange(count, dtype=np.uint64),
        "x_um": np.zeros(count),
        "y_um": np.zeros(count),
        "z_um": np.zeros(count),
        "dir_x": np.zeros(count),
        "dir_y": np.zeros(count),
        "dir_z": np.ones(count),
        "kinetic_energy_eV": np.full(count, 300_000.0),
        "time_ns": np.arange(count, dtype=float),
        "weight": np.ones(count),
        "loss_channel": np.zeros(count, dtype=np.int32),
    }


class ToyBranchKernel:
    name = "toy-branches"

    def simulate(self, incident_electrons, context):
        count = len(incident_electrons["electron_id"])
        selected = np.repeat(np.arange(count), 2)
        outgoing = {
            name: np.asarray(values)[selected].copy() for name, values in incident_electrons.items()
        }
        outgoing["electron_id"] = np.arange(2 * count, dtype=np.uint64)
        outgoing["parent_electron_id"] = np.repeat(incident_electrons["electron_id"], 2)
        outgoing["branch_id"] = np.tile(np.array([0, 1], dtype=np.uint32), count)
        outgoing["weight"] = np.tile(np.array([0.75, 0.25]), count)
        outgoing["kinetic_energy_eV"][1::2] -= 25.0
        outgoing["loss_channel"][1::2] = 2
        return SpecimenResult(
            outgoing,
            "branch_probability",
            {"reference_energy_eV": context.reference_energy_eV},
        )


class FakeAbTEMBackend:
    def calculate(self, beam_energy_eV, parameters):
        self.beam_energy_eV = beam_energy_eV
        self.parameters = parameters
        return AbTEMScatteringModel(
            elastic=AngularDistribution(
                theta_x_mrad=np.array([1.0]),
                theta_y_mrad=np.array([0.0]),
                probability=np.array([3.0]),
            ),
            core=AngularDistribution(
                theta_x_mrad=np.array([0.0]),
                theta_y_mrad=np.array([-2.0]),
                probability=np.array([7.0]),
            ),
            specimen_thickness_A=12.5,
            core_probability=0.25,
            core_energy_loss_eV=532.0,
            metadata={"engine": "fake-abTEM"},
        )


class SpecimenPluginTests(unittest.TestCase):
    def test_abtem_adapter_emits_weighted_elastic_and_core_branches(self):
        backend = FakeAbTEMBackend()
        incident = individual_electrons(6)
        result = run_specimen_kernel(
            AbTEMSpecimenKernel(backend),
            incident,
            reference_energy_eV=300_000.0,
            parameters={"energy_tolerance_eV": 0.1},
            random_seed=17,
        )
        self.assertEqual(result.weight_semantics, "branch_probability")
        self.assertEqual(len(result.electrons["electron_id"]), 12)
        np.testing.assert_allclose(result.electrons["weight"][:6], 0.75)
        np.testing.assert_allclose(result.electrons["weight"][6:], 0.25)
        np.testing.assert_array_equal(result.electrons["branch_id"][:6], 0)
        np.testing.assert_array_equal(result.electrons["branch_id"][6:], 1)
        np.testing.assert_array_equal(
            result.electrons["parent_electron_id"],
            np.concatenate((incident["electron_id"], incident["electron_id"])),
        )
        np.testing.assert_allclose(result.electrons["kinetic_energy_eV"][6:], 299_468.0)
        np.testing.assert_array_equal(result.electrons["loss_channel"][:6], 0)
        np.testing.assert_array_equal(result.electrons["loss_channel"][6:], 3)
        np.testing.assert_allclose(result.electrons["z_um"], 12.5e-4)
        direction_norm = np.sqrt(
            result.electrons["dir_x"] ** 2
            + result.electrons["dir_y"] ** 2
            + result.electrons["dir_z"] ** 2
        )
        np.testing.assert_allclose(direction_norm, 1.0)
        self.assertEqual(backend.beam_energy_eV, 300_000.0)

        individual, summary = materialize_phase_space(
            result.electrons,
            mode="categorical",
            random_seed=19,
            max_electrons=20,
        )
        self.assertEqual(summary["output_electrons"], 6)
        self.assertEqual(len(np.unique(individual["parent_electron_id"])), 6)

    def test_abtem_adapter_rejects_mixed_beam_energies(self):
        incident = individual_electrons(2)
        incident["kinetic_energy_eV"][1] += 2.0
        with self.assertRaisesRegex(ValueError, "one incident beam energy"):
            run_specimen_kernel(
                AbTEMSpecimenKernel(FakeAbTEMBackend()),
                incident,
                reference_energy_eV=300_000.0,
                parameters={"energy_tolerance_eV": 1.0},
                random_seed=23,
            )

    def test_identity_kernel_adds_lineage(self):
        incident = individual_electrons()
        result = run_specimen_kernel(
            load_specimen_kernel("identity"),
            incident,
            reference_energy_eV=300_000.0,
            parameters={},
            random_seed=3,
        )
        self.assertEqual(result.weight_semantics, "branch_probability")
        np.testing.assert_array_equal(
            result.electrons["parent_electron_id"], incident["electron_id"]
        )
        np.testing.assert_array_equal(result.electrons["branch_id"], 0)

    def test_module_object_plugin_and_categorical_sampling(self):
        incident = individual_electrons(20)
        kernel = load_specimen_kernel(f"{__name__}:ToyBranchKernel")
        result = run_specimen_kernel(
            kernel,
            incident,
            reference_energy_eV=300_000.0,
            parameters={},
            random_seed=5,
        )
        individual, summary = materialize_phase_space(
            result.electrons,
            mode="categorical",
            random_seed=7,
            max_electrons=100,
        )
        self.assertEqual(summary["output_electrons"], 20)
        self.assertEqual(len(np.unique(individual["parent_electron_id"])), 20)
        self.assertTrue(np.all(individual["weight"] == 1.0))
        self.assertEqual(len(np.unique(individual["electron_id"])), 20)
        entries, _, _ = transfer_phase_space(
            individual,
            SpectrometerConfig(
                dispersion_eV_per_column=1.0,
                zero_loss_stitched_column=5.0,
                detector_rows=10,
                detector_columns=100,
                pixel_pitch_um=10.0,
                zero_y_row=4.5,
                maximum_energy_loss_eV=30.0,
            ),
            reference_energy_eV=300_000.0,
        )
        np.testing.assert_array_equal(
            entries["parent_electron_id"], individual["parent_electron_id"]
        )
        np.testing.assert_array_equal(entries["branch_id"], individual["branch_id"])
        np.testing.assert_array_equal(entries["source_record_id"], individual["source_record_id"])

    def test_poisson_sampling_matches_seeded_counts(self):
        weighted = individual_electrons(2)
        weighted["weight"] = np.array([2.0, 3.0])
        expected_counts = np.random.default_rng(11).poisson(weighted["weight"])
        individual, summary = materialize_phase_space(
            weighted,
            mode="poisson",
            random_seed=11,
            max_electrons=100,
        )
        actual_counts = np.array(
            [
                np.count_nonzero(individual["source_record_id"] == source_id)
                for source_id in weighted["electron_id"]
            ]
        )
        np.testing.assert_array_equal(actual_counts, expected_counts)
        self.assertEqual(summary["output_electrons"], int(expected_counts.sum()))
        self.assertTrue(np.all(individual["weight"] == 1.0))

    def test_categorical_rejects_probability_sum_above_one(self):
        weighted = individual_electrons(2)
        weighted["electron_id"] = np.array([10, 11], dtype=np.uint64)
        weighted["parent_electron_id"] = np.array([4, 4], dtype=np.uint64)
        weighted["branch_id"] = np.array([0, 1], dtype=np.uint32)
        weighted["weight"] = np.array([0.8, 0.4])
        with self.assertRaisesRegex(ValueError, "greater than one"):
            materialize_phase_space(
                weighted,
                mode="categorical",
                random_seed=2,
                max_electrons=10,
            )

    def test_optional_lineage_fields_round_trip(self):
        electrons = individual_electrons(3)
        electrons["parent_electron_id"] = np.array([8, 9, 10], dtype=np.uint64)
        electrons["branch_id"] = np.array([0, 1, 2], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weighted.h5"
            write_phase_space_hdf5(
                path,
                electrons,
                {"weight_semantics": "branch_probability"},
            )
            restored = read_phase_space_hdf5(path)
        np.testing.assert_array_equal(
            restored["parent_electron_id"], electrons["parent_electron_id"]
        )
        np.testing.assert_array_equal(restored["branch_id"], electrons["branch_id"])


if __name__ == "__main__":
    unittest.main()
