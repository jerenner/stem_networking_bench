#!/usr/bin/env python3
"""Plot single-frame BLR baseline samples from DM4 spectrum data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_blr_grouping_dm4 import read_image
from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack
from plot_blr_baseline_samples import phase_statistics
from plot_nio_processing_analysis import configure_matplotlib_cache, subtract_imagej_blr


def parse_indices(text: str) -> list[int]:
    indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not indices:
        raise ValueError("at least one frame index is required")
    if any(index < 0 for index in indices):
        raise ValueError(f"frame indices must be non-negative: {indices}")
    return indices


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def folded_zlp(profile, np):
    return profile[:768].reshape(4, 192).mean(axis=0)


def half_baseline_profiles(no_blr, grouped_blr, columnwise_blr, np):
    height = no_blr.shape[0]
    half = height // 2
    grouped_baseline = no_blr - grouped_blr
    columnwise_baseline = no_blr - columnwise_blr
    rows = (half // 2, half + half // 2)
    return tuple(
        {
            "grouped": grouped_baseline[row].astype(np.float64, copy=False),
            "columnwise": columnwise_baseline[row].astype(np.float64, copy=False),
        }
        for row in rows
    )


def plot_one_frame(output_path,
                   label,
                   source_name,
                   frame_index,
                   profiles,
                   plt,
                   np):
    half_names = ("Top detector half", "Bottom detector half")
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), constrained_layout=True)

    for column, (half_name, profile) in enumerate(zip(half_names, profiles)):
        zlp_columnwise = folded_zlp(profile["columnwise"], np)
        zlp_grouped = folded_zlp(profile["grouped"], np)
        core_columnwise = profile["columnwise"][768:]
        core_grouped = profile["grouped"][768:]

        zlp_phase, zlp_phase_std, zlp_residuals = phase_statistics(
            zlp_columnwise, 4, np
        )
        core_phase, core_phase_std, core_residuals = phase_statistics(
            core_columnwise, 16, np
        )

        axes[0, column].plot(
            zlp_columnwise, linewidth=0.9, label="per-column edge-row mean"
        )
        axes[0, column].plot(
            zlp_grouped, linewidth=1.1, label="4-column grouped estimate"
        )
        axes[0, column].set_title(f"{half_name}: folded-ZLP BLR baseline")
        axes[0, column].set_xlabel("Physical ZLP detector column modulo 192")
        axes[0, column].set_ylabel("Baseline removed")

        axes[1, column].errorbar(
            np.arange(4), zlp_phase, yerr=zlp_phase_std, marker="o", capsize=3
        )
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].set_title(
            f"ZLP residual by phase modulo 4, RMS={np.sqrt(np.mean(zlp_residuals**2)):.2f}"
        )
        axes[1, column].set_xlabel("Column position inside each 4-column group")
        axes[1, column].set_ylabel("Columnwise minus group mean")
        axes[1, column].set_xticks(np.arange(4))

        zoom = min(512, core_columnwise.size)
        axes[2, column].plot(
            core_columnwise[:zoom], linewidth=0.8, label="per-column edge-row mean"
        )
        axes[2, column].plot(
            core_grouped[:zoom], linewidth=1.0, label="16-column grouped estimate"
        )
        for boundary in range(16, zoom, 16):
            axes[2, column].axvline(boundary, color="gray", alpha=0.12, linewidth=0.6)
        axes[2, column].set_title(f"{half_name}: CoreLoss BLR baseline, first {zoom} columns")
        axes[2, column].set_xlabel("CoreLoss-relative detector column")
        axes[2, column].set_ylabel("Baseline removed")

        axes[3, column].errorbar(
            np.arange(16), core_phase, yerr=core_phase_std, marker="o", capsize=3
        )
        axes[3, column].axhline(0.0, color="black", linewidth=0.8)
        axes[3, column].set_title(
            f"CoreLoss residual by phase modulo 16, RMS={np.sqrt(np.mean(core_residuals**2)):.2f}"
        )
        axes[3, column].set_xlabel("Column position inside each 16-column group")
        axes[3, column].set_ylabel("Columnwise minus group mean")
        axes[3, column].set_xticks(np.arange(16))

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[2, 0].legend()
    fig.suptitle(f"{label}: single-frame BLR samples, {source_name}, frame {frame_index}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_profiles(profiles, np):
    result = []
    for profile in profiles:
        zlp_columnwise = folded_zlp(profile["columnwise"], np)
        core_columnwise = profile["columnwise"][768:]
        zlp_phase, zlp_phase_std, zlp_residuals = phase_statistics(
            zlp_columnwise, 4, np
        )
        core_phase, core_phase_std, core_residuals = phase_statistics(
            core_columnwise, 16, np
        )
        result.append(
            {
                "zlp_phase_modulo_4_mean": zlp_phase.tolist(),
                "zlp_phase_modulo_4_stddev": zlp_phase_std.tolist(),
                "zlp_within_group_residual_rms": float(np.sqrt(np.mean(zlp_residuals**2))),
                "core_phase_modulo_16_mean": core_phase.tolist(),
                "core_phase_modulo_16_stddev": core_phase_std.tolist(),
                "core_within_group_residual_rms": float(np.sqrt(np.mean(core_residuals**2))),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--dark-frame", type=Path, required=True)
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio"
    )
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--frames-axis", type=int, default=None)
    parser.add_argument("--frame-indices", default="0,128,256,384")
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument("--label", default="Dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frame_indices = parse_indices(args.frame_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dark_frame, "r") as dark_h5:
        dark = read_image(dark_h5, args.dark_dataset, np)
    if dark.shape != (args.height, args.width):
        raise ValueError(f"dark frame shape {dark.shape} does not match requested frame shape")

    data, info = load_dm4(args.input, args.reader)
    stack = normalize_to_frame_stack(data, args.frames_axis, args.height, args.width)
    if any(index >= stack.shape[0] for index in frame_indices):
        raise ValueError(
            f"requested frame indices {frame_indices} outside stack with {stack.shape[0]} frames"
        )

    summary = {
        "source_file": str(args.input),
        "reader": info["reader"],
        "stack_shape": [int(value) for value in stack.shape],
        "dark_frame": str(args.dark_frame),
        "dataset_label": args.label,
        "frame_indices": frame_indices,
        "plots": [],
        "frames": [],
    }

    for frame_index in frame_indices:
        raw = materialize(stack[frame_index], np).astype(np.float32, copy=False)
        dark_subtracted = raw - dark
        grouped = subtract_imagej_blr(
            dark_subtracted[None],
            np,
            args.blr_rows,
            args.zlp_width,
            args.zlp_group_columns,
            args.core_group_columns,
        )[0]
        columnwise = subtract_imagej_blr(
            dark_subtracted[None],
            np,
            args.blr_rows,
            args.zlp_width,
            1,
            1,
        )[0]
        profiles = half_baseline_profiles(dark_subtracted, grouped, columnwise, np)
        plot_name = f"{args.label.lower()}_single_frame_blr_samples_{frame_index:04d}.png"
        plot_path = args.output_dir / plot_name
        plot_one_frame(
            plot_path,
            args.label,
            args.input.name,
            frame_index,
            profiles,
            plt,
            np,
        )
        summary["plots"].append(plot_name)
        summary["frames"].append(
            {
                "frame_index": frame_index,
                "half_order": ["top", "bottom"],
                "half_metrics": summarize_profiles(profiles, np),
            }
        )
        print(f"Wrote {plot_path}", flush=True)

    summary_path = args.output_dir / f"{args.label.lower()}_single_frame_blr_samples.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
