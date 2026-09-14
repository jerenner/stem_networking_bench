from __future__ import annotations

import ast
import unittest
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent


class CombinedDemoTests(unittest.TestCase):
    def test_full_demo_keeps_the_combined_story_order(self):
        tree = ast.parse((DEMO_ROOT / "scenes.py").read_text(encoding="utf-8"))
        full_demo = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullDemo"
        )
        construct = next(
            node
            for node in full_demo.body
            if isinstance(node, ast.FunctionDef) and node.name == "construct"
        )
        calls = [
            expression.value.func.attr
            for expression in construct.body
            if isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Call)
            and isinstance(expression.value.func, ast.Attribute)
        ]
        self.assertEqual(
            calls,
            [
                "intro_segment",
                "microscope_segment",
                "scan_segment",
                "packet_segment",
                "streaming_segment",
                "spectrum_segment",
                "live_maps_segment",
                "selection_segment",
                "outro_segment",
            ],
        )

    def test_render_targets_are_separate_from_existing_demos(self):
        script = (DEMO_ROOT / "render_demo.sh").read_text(encoding="utf-8")
        self.assertIn('MEDIA="$DEMO_ROOT/.manim"', script)
        self.assertIn('RENDERS="$DEMO_ROOT/renders"', script)
        self.assertIn("lmto_daqiri_combined", script)
        self.assertNotIn("lmto_doeels_workflow_200keV_FullDemo.mp4", script)
        self.assertNotIn("stem_daqiri_nio_FullDemo.mp4", script)

    def test_storyboard_records_native_tiles_and_scan_order(self):
        storyboard = (DEMO_ROOT / "STORYBOARD.md").read_text(encoding="utf-8")
        self.assertIn("192 `128 x 32`-pixel ZLP tiles", storyboard)
        self.assertIn("768 `32 x 128`-pixel", storyboard)
        self.assertIn("CoreLoss tiles; 960 equal-payload", storyboard)
        self.assertIn("unidirectional left-to-right fast scans", storyboard)


if __name__ == "__main__":
    unittest.main()
