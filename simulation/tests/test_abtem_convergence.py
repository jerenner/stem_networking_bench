import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from eels_sim.abtem_adapter import AbTEMScatteringModel, AngularDistribution
from eels_sim.abtem_convergence import (
    load_convergence_definition,
    run_abtem_convergence,
)


class FakeConvergenceBackend:
    def calculate(self, beam_energy_eV, parameters):
        repetitions = parameters["repetitions"]
        sampling = float(parameters["potential_sampling_A"])
        cutoff = float(parameters["max_scattering_angle_mrad"])
        double_channel = bool(parameters["core_loss"]["double_channel"])
        thickness_A = 4.17 * repetitions[2]
        angular_scale = min(0.4 * cutoff, 30.0)
        elastic = AngularDistribution(
            np.array([0.0, angular_scale]),
            np.array([0.0, 0.0]),
            np.array([0.8, 0.2]),
        )
        core = AngularDistribution(
            np.array([0.0, 0.5 * angular_scale]),
            np.array([0.0, 0.5 * angular_scale]),
            np.array([0.65, 0.35]),
        )
        core_probability = (
            1.0e-7
            * repetitions[2]
            * (1.0 + 0.02 * repetitions[0])
            * (1.0 + 0.01 / sampling)
            * (1.05 if double_channel else 1.0)
        )
        return AbTEMScatteringModel(
            elastic=elastic,
            specimen_thickness_A=thickness_A,
            core=core,
            core_probability=core_probability,
            core_energy_loss_eV=532.0,
            metadata={
                "atom_count": 8 * int(np.prod(repetitions)),
                "potential_gpts": [
                    round(4.17 * repetitions[0] / sampling),
                    round(4.17 * repetitions[1] / sampling),
                ],
                "potential_slices": repetitions[2] * 2,
                "elastic_captured_intensity": 0.999,
            },
        )


class AbTEMConvergenceTests(unittest.TestCase):
    config_path = Path(__file__).resolve().parents[1] / "config" / "abtem_nio_convergence.toml"

    def test_definition_deduplicates_baseline_and_selects_references(self):
        definition = load_convergence_definition(self.config_path)
        self.assertEqual(len(definition.cases), 14)
        self.assertEqual(
            definition.reference_case_ids["lateral_repetitions"],
            "lateral_repetitions_3x3",
        )
        self.assertEqual(
            definition.reference_case_ids["potential_sampling_A"],
            "potential_sampling_A_0p09A",
        )
        self.assertEqual(
            definition.reference_case_ids["max_scattering_angle_mrad"],
            "max_scattering_angle_mrad_80mrad",
        )
        self.assertEqual(
            definition.reference_case_ids["double_channel"],
            "double_channel_double",
        )
        self.assertEqual(definition.reference_case_ids["thickness_repetitions"], "baseline")
        self.assertEqual(definition.axis_case_ids["potential_sampling_A"].count("baseline"), 1)
        self.assertEqual(definition.histogram_max_angle_mrad, 100.0)
        self.assertEqual(definition.comparison_collection_angle_mrad, 50.0)
        self.assertEqual(
            definition.combined_case_ids,
            {
                "reference": "combined_reference",
                "confirmation": "larger_finer_confirmation",
            },
        )
        combined = {case.case_id: case for case in definition.cases}
        self.assertEqual(combined["combined_reference"].parameters["repetitions"], [3, 3, 4])
        self.assertEqual(
            combined["larger_finer_confirmation"].parameters["repetitions"],
            [4, 4, 4],
        )
        self.assertEqual(
            combined["larger_finer_confirmation"].parameters["max_scattering_angle_mrad"],
            100.0,
        )

    def test_runner_writes_machine_readable_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output_json = Path(directory) / "study.json"
            output_csv = Path(directory) / "study.csv"
            output_combined_plot = Path(directory) / "combined.png"
            report = run_abtem_convergence(
                self.config_path,
                output_json,
                output_csv,
                output_combined_plot=output_combined_plot,
                backend=FakeConvergenceBackend(),
                progress=None,
            )
            restored = json.loads(output_json.read_text())
            with output_csv.open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            combined_plot_exists = output_combined_plot.exists()

        self.assertEqual(report["summary"]["case_count"], 14)
        self.assertEqual(report["summary"]["failed_cases"], 0)
        self.assertEqual(restored["schema"], "eels-sim-abtem-convergence-v2")
        self.assertEqual(len(csv_rows), 18)
        self.assertTrue(combined_plot_exists)
        self.assertEqual(report["combined_check"]["reference_case_id"], "combined_reference")
        self.assertIsNotNone(
            report["combined_check"]["confirmation_relative_to_reference"][
                "core_radial_wasserstein_mrad"
            ]
        )
        self.assertIsNotNone(
            report["combined_check"]["confirmation_relative_to_reference"][
                "core_collection_radial_wasserstein_mrad"
            ]
        )
        reference = next(
            case for case in report["cases"] if case["case_id"] == "combined_reference"
        )
        self.assertAlmostEqual(
            reference["core_probability"],
            reference["core_probability_within_collection"],
        )
        thickness_axis = report["axes"]["thickness_repetitions"]
        self.assertIn("physical thickness trend", thickness_axis["interpretation"])
        lateral_first = report["axes"]["lateral_repetitions"]["points"][0]
        self.assertIsNotNone(
            lateral_first["comparison_to_reference"]["core_radial_total_variation"]
        )
        self.assertIsNotNone(
            lateral_first["comparison_to_reference"]["core_radial_wasserstein_mrad"]
        )


if __name__ == "__main__":
    unittest.main()
