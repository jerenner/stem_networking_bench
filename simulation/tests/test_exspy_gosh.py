import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from eels_sim.exspy_gosh import (
    DFT_GOSH_DOI,
    LMTO_GOS_COMPONENTS,
    exclusive_core_probabilities,
    formula_unit_areal_density_cm2,
    lmto_gosh_library,
)


class _Parameter:
    def __init__(self, value):
        self.value = value


class FakeEELSCLEdge:
    onsets = {
        "Ti_M3": 35.0,
        "Ti_M2": 36.0,
        "Mn_M3": 51.0,
        "Mn_M2": 52.0,
        "Li_K": 55.0,
        "Ti_L3": 456.0,
        "Ti_L2": 462.0,
        "O_K": 532.0,
        "Mn_L3": 640.0,
        "Mn_L2": 651.0,
    }

    def __init__(self, element_subshell, GOS, gos_file_path):
        del GOS, gos_file_path
        self.name = element_subshell
        self.onset_energy = _Parameter(self.onsets[element_subshell])
        self.effective_angle = _Parameter(0.0)

    def set_microscope_parameters(self, E0, alpha, beta, energy_scale):
        del E0, alpha, energy_scale
        self.effective_angle.value = 0.95 * beta

    def function(self, energy_eV):
        energy = np.asarray(energy_eV)
        excess = energy - self.onset_energy.value
        shell_factor = 2.0 if self.name.endswith(("M3", "L3")) else 1.0
        return np.where(excess >= 0.0, shell_factor * np.exp(-excess / 80.0), 0.0)


class ExSpyGOSHTests(unittest.TestCase):
    def test_optical_depth_probability_conversion(self):
        optical_depth = np.array([0.1, 0.2, 0.3])
        probabilities = exclusive_core_probabilities(optical_depth)
        self.assertAlmostEqual(probabilities.sum(), 1.0 - np.exp(-0.6))
        np.testing.assert_allclose(probabilities / probabilities.sum(), [1 / 6, 2 / 6, 3 / 6])

    def test_formula_unit_areal_density(self):
        value = formula_unit_areal_density_cm2(
            density_g_cm3=4.0,
            molar_mass_g_mol=80.0,
            thickness_A=500.0,
        )
        self.assertAlmostEqual(value, 1.50553519e17, delta=1.0e8)

    def test_lmto_provider_combines_all_six_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "fake.gosh"
            with h5py.File(database_path, "w") as h5:
                reference = h5.create_group("metadata/data_ref")
                reference.attrs["data_doi"] = DFT_GOSH_DOI
            library = lmto_gosh_library(
                database_path,
                verify_gosh_hash=False,
                edge_class=FakeEELSCLEdge,
            )
        self.assertEqual(
            [component.name for component in library.core_loss],
            [definition.name for definition in LMTO_GOS_COMPONENTS],
        )
        self.assertTrue(
            all(component.integrated_probability > 0.0 for component in library.core_loss)
        )
        self.assertLess(sum(c.integrated_probability for c in library.core_loss), 1.0)
        ti_m = library.core_loss[0]
        self.assertEqual(
            [item["subshell"] for item in ti_m.metadata["subshells"]],
            ["Ti_M3", "Ti_M2"],
        )
        self.assertEqual(ti_m.onset_energy_eV, 35.0)
        self.assertTrue(ti_m.metadata["quantitative_atomic_cross_section"])
        self.assertFalse(ti_m.metadata["material_specific_fine_structure"])


if __name__ == "__main__":
    unittest.main()
