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
        self.assertIn("RX0-3 under FPGA 0 and RX4-7 under FPGA 1", storyboard)
        self.assertIn("one ZLP read and one quarter of CoreLoss", storyboard)
        self.assertIn("frame stacks/buckets", storyboard)
        self.assertIn("count-preserving energy rebinning", storyboard)

    def test_sketch_tile_schedule_fills_edges_inward_with_complete_zlp_reads(self):
        """Test the pure mapping without importing the optional Manim runtime."""
        tree = ast.parse((DEMO_ROOT / "scenes.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "phase_tile_rows"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "scenes.py", "exec"), namespace)
        tile_rows = namespace["phase_tile_rows"]

        zlp_tiles = set()
        core_tiles = set()
        for phase in range(4):
            phase_zlp_tiles = set()
            phase_core_tiles = set()
            for source in range(8):
                zlp_row, core_row = tile_rows(source, phase)
                phase_zlp_tiles.update((zlp_row, phase * 6 + column) for column in range(6))
                phase_core_tiles.update((core_row, column) for column in range(24))
            self.assertEqual(len(phase_zlp_tiles), 48)
            self.assertEqual(len(phase_core_tiles), 192)
            self.assertEqual(
                {column for _, column in phase_zlp_tiles},
                set(range(phase * 6, phase * 6 + 6)),
            )
            self.assertEqual({row for row, _ in phase_zlp_tiles}, set(range(8)))
            self.assertEqual(
                {row for row, _ in phase_core_tiles},
                set(range(phase * 4, phase * 4 + 4))
                | set(range(28 - phase * 4, 32 - phase * 4)),
            )
            self.assertFalse(zlp_tiles & phase_zlp_tiles)
            self.assertFalse(core_tiles & phase_core_tiles)
            zlp_tiles.update(phase_zlp_tiles)
            core_tiles.update(phase_core_tiles)
        self.assertEqual(len(zlp_tiles), 192)
        self.assertEqual(len(core_tiles), 768)


if __name__ == "__main__":
    unittest.main()
