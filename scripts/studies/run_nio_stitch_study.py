#!/usr/bin/env python3
"""Run the manifest-driven NiO ZLP/CoreLoss stitch study."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-study-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--currents", nargs="*", default=None)
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--read-chunk-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source_manifest, args.output_root / "source_manifest.json")
    config = {
        "source_manifest": str(args.source_manifest.resolve()),
        "source_study_root": str(args.source_study_root.resolve()),
        "reader": args.reader,
        "tensor_frames": args.tensor_frames,
        "read_chunk_size": args.read_chunk_size,
        "frame_shape": [960, 3840],
        "edge_rows": 32,
        "blr_rows": 30,
        "zlp_width": 768,
        "zlp_period": 192,
        "zlp_group_columns": 4,
        "core_group_columns": 16,
        "modes": ["no_blr", "grouped_blr"],
        "dynamic_mask_applied": False,
        "static_valid_mask_applied": True,
        "fit_left_window": [160, 190],
        "fit_right_window": [194, 223],
        "excluded_boundary_columns": [191, 192, 193],
        "fit_model": "Huber robust quadratic in log intensity with a fitted ZLP step",
    }
    (args.output_root / "study_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    selected = set(args.currents) if args.currents else None
    completed = []
    groups = sorted(
        manifest["currents"].items(), key=lambda item: item[1]["current_pa"]
    )
    for key, source_group in groups:
        if selected is not None and key not in selected and source_group["current_label"] not in selected:
            continue
        spectrum_files = source_group.get("spectrum_files", [])
        if not spectrum_files:
            print(f"Skipping {key}: no spectrum files", flush=True)
            continue
        dark_frame = (
            args.source_study_root / "currents" / key / "dark" / "dark_frame.h5"
        )
        if not dark_frame.exists():
            raise FileNotFoundError(f"missing matching dark frame: {dark_frame}")
        current_dir = args.output_root / "currents" / key
        current_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"current_key": key, **source_group}
        (current_dir / "current_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        output_h5 = current_dir / "stitch_spectra.h5"
        if args.force or not output_h5.exists():
            run_command([
                sys.executable,
                script_dir / "analyze_stitch_dm4.py",
                *[item["path"] for item in spectrum_files],
                "--dark-frame", dark_frame,
                "--output", output_h5,
                "--reader", args.reader,
                "--tensor-frames", args.tensor_frames,
                "--read-chunk-size", args.read_chunk_size,
            ])
        else:
            print(f"Reusing {output_h5}", flush=True)
        completed.append(key)

    if not args.skip_comparison:
        run_command([
            sys.executable,
            script_dir / "compare_nio_stitch_study.py",
            "--study-root", args.output_root,
        ])
    summary = {
        "completed_currents": completed,
        "comparison_generated": not args.skip_comparison,
        "force": args.force,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
