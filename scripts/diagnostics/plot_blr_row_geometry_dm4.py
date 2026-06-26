#!/usr/bin/env python3
"""Compare detector-edge row geometry across raw spectrum DM4 files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack
from plot_nio_processing_analysis import configure_matplotlib_cache


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def label_for(path: Path):
    match = re.search(r"NiO\s+(.+?)\s+Spectrum", path.stem, re.IGNORECASE)
    return match.group(1).replace(" ", "") if match else path.stem


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio"
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--read-chunk-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--frames-axis", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if args.frames <= 0 or args.read_chunk_size <= 0:
        raise ValueError("frame and chunk counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    summaries = []

    for input_path in args.inputs:
        data, info = load_dm4(input_path, args.reader)
        stack = normalize_to_frame_stack(
            data, args.frames_axis, args.height, args.width
        )
        frame_count = min(args.frames, stack.shape[0])
        row_sum = np.zeros(args.height, dtype=np.float64)
        values_per_row = 0
        for start in range(0, frame_count, args.read_chunk_size):
            end = min(frame_count, start + args.read_chunk_size)
            block = materialize(stack[start:end], np)
            row_sum += block.sum(axis=(0, 2), dtype=np.float64)
            values_per_row += block.shape[0] * block.shape[2]
        profile = row_sum / values_per_row
        profiles.append(profile)
        top_transition = int(np.argmax(np.abs(np.diff(profile[:64]))) + 1)
        bottom_transition = int(
            np.argmax(np.abs(np.diff(profile[896:]))) + 897
        )
        summaries.append({
            "label": label_for(input_path),
            "path": str(input_path),
            "reader": info["reader"],
            "frames": frame_count,
            "shape": list(stack.shape),
            "strongest_top_edge_transition_row": top_transition,
            "strongest_bottom_edge_transition_row": bottom_transition,
            "rows_27_through_35": profile[27:36].tolist(),
            "rows_924_through_932": profile[924:933].tolist(),
        })
        print(
            f"{input_path}: frames={frame_count}, top transition={top_transition}, "
            f"bottom transition={bottom_transition}",
            flush=True,
        )
        del stack, data

    profiles = np.stack(profiles)
    labels = [item["label"] for item in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    regions = (
        (axes[0, 0], slice(0, 64), "Top detector edge"),
        (axes[0, 1], slice(896, 960), "Bottom detector edge"),
    )
    for axis, rows, title in regions:
        x = np.arange(args.height)[rows]
        for profile, label in zip(profiles, labels):
            axis.plot(x, profile[rows], linewidth=0.9, label=label)
        axis.set_title(f"{title}: raw row mean")
        axis.set_xlabel("Detector row")
        axis.set_ylabel("Raw detector value")
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)

    difference_regions = (
        (axes[1, 0], slice(1, 64), "Top detector edge"),
        (axes[1, 1], slice(897, 960), "Bottom detector edge"),
    )
    row_differences = np.abs(np.diff(profiles, axis=1))
    for axis, rows, title in difference_regions:
        x = np.arange(args.height)[rows]
        for differences, label in zip(row_differences, labels):
            axis.semilogy(x, differences[rows.start - 1:rows.stop - 1], linewidth=0.9, label=label)
        axis.set_title(f"{title}: absolute adjacent-row change")
        axis.set_xlabel("Right-hand detector row")
        axis.set_ylabel("Absolute row-mean change")
        axis.grid(alpha=0.2, which="both")
        axis.legend(ncol=2, fontsize=8)

    for axis in (axes[0, 0], axes[1, 0]):
        axis.axvspan(0, 29, color="tab:blue", alpha=0.08, label="BLR rows")
        axis.axvline(30, color="tab:orange", linestyle="--", linewidth=0.8)
        axis.axvline(32, color="black", linestyle="--", linewidth=0.8)
    for axis in (axes[0, 1], axes[1, 1]):
        axis.axvline(928, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(930, color="tab:orange", linestyle="--", linewidth=0.8)
        axis.axvspan(930, 959, color="tab:blue", alpha=0.08, label="BLR rows")

    fig.savefig(args.output_dir / "blr_row_geometry_across_currents.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "blr_row_geometry_across_currents.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
