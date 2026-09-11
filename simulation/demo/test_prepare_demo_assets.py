from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))

from prepare_demo_assets import resolve_path  # noqa: E402
from prepare_demo_assets import (
    load_config,
    select_probe_points,
)


class LMTOAssetTests(unittest.TestCase):
    def test_probe_selection_is_unique_and_element_sensitive(self):
        mn = np.zeros((4, 4))
        mn[1, 2] = 9
        oxygen = np.zeros((4, 4))
        oxygen[0, 1] = 8
        titanium = np.zeros((4, 4))
        titanium[3, 0] = 7
        points = select_probe_points({"Mn": mn, "O": oxygen, "Ti": titanium}, np)
        self.assertEqual(points["Mn"], (1, 2))
        self.assertEqual(points["O"], (0, 1))
        self.assertEqual(points["Ti"], (3, 0))
        self.assertEqual(len(set(points.values())), 3)

    def test_configured_sources_exist(self):
        config = load_config(DEMO_ROOT / "demo_config.json")
        source_keys = (
            "simulation_config",
            "spatial_response",
            "spectrum_image",
            "spectral_library",
            "physical_raw_scan",
            "exposure_sweep",
            "detector_response",
            "dark_calibration",
            "sparse_calibration",
        )
        missing = [key for key in source_keys if not resolve_path(config[key]).exists()]
        if missing:
            self.skipTest(
                "generated physics products are not present; run "
                "scripts/reproduce_200kev_demo.sh before validating demo inputs"
            )

    def test_generated_manifest_records_primary_unreplicated_demo(self):
        manifest = DEMO_ROOT / "assets" / "generated_200keV" / "demo_metadata.json"
        if not manifest.exists():
            self.skipTest("generate demo assets before validating their manifest")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema"], "eels-sim-lmto-demo-v1")
        self.assertEqual(metadata["scan"]["shape"], [16, 16])
        self.assertEqual(metadata["scan"]["beam_energy_keV"], 200.0)
        self.assertAlmostEqual(metadata["scan"]["field_of_view_A"], 8.3)
        self.assertEqual(metadata["scan"]["raw_frames_if_retained"], 768000)
        self.assertEqual(metadata["specimen"]["atom_count"], 128)
        self.assertEqual(set(metadata["final_correlations"]), {"Mn", "O", "Ti"})
        self.assertGreater(metadata["spectrum_example"]["fitted_edge_counts"]["Mn L₂,₃"], 100.0)
        for source in metadata["source_files"].values():
            if "dark_calibration" not in source:
                self.assertIn("200keV", source)
        required_assets = {
            "application_workflows.png",
            "structure_planes.png",
            "abtem_spatial_channels.png",
            "theoretical_spectrum_components.png",
            "raw_detector_frame.png",
            "analog_vs_counted_processing.png",
            "final_haadf.png",
            "final_mn.png",
            "final_o.png",
            "final_ti.png",
        }
        self.assertTrue(required_assets.issubset(set(metadata["assets"])))


if __name__ == "__main__":
    unittest.main()
