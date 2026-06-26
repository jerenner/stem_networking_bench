#!/usr/bin/env python3
"""Compare disabled, grouped, and columnwise BLR on streamed DM4 spectra."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack
from plot_nio_processing_analysis import configure_matplotlib_cache, subtract_imagej_blr


MODE_NAMES = ("no_blr", "grouped_blr", "columnwise_blr")
MODE_LABELS = ("No BLR", "Grouped BLR (ZLP=4, CoreLoss=16)", "Columnwise BLR")


def read_image(h5_file, dataset: str, np):
    data = h5_file[dataset][...]
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"expected one image at {dataset}, got {data.shape}")
    return data.astype(np.float32, copy=False)


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def fold_zlp(sums, counts, np):
    folded_sums = sums[..., :768].reshape(*sums.shape[:-1], 4, 192).sum(axis=-2)
    folded_counts = counts[..., :768].reshape(*counts.shape[:-1], 4, 192).sum(axis=-2)
    return safe_divide(folded_sums, folded_counts, np)


def jump_metrics(profile, group_origin, group_width, first_column, last_column, np):
    differences = np.abs(np.diff(profile))
    right_columns = np.arange(1, profile.size)
    selected = (right_columns >= first_column) & (right_columns < last_column)
    boundaries = selected & ((right_columns - group_origin) % group_width == 0)
    inside = selected & ~boundaries
    boundary_values = differences[boundaries]
    inside_values = differences[inside]
    boundary_mean = float(np.mean(boundary_values))
    inside_mean = float(np.mean(inside_values))
    return {
        "boundary_jump_mean": boundary_mean,
        "boundary_jump_median": float(np.median(boundary_values)),
        "inside_jump_mean": inside_mean,
        "inside_jump_median": float(np.median(inside_values)),
        "boundary_to_inside_mean_ratio": boundary_mean / inside_mean,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dark-frame", type=Path, required=True)
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio"
    )
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--frames-axis", type=int, default=None)
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--read-chunk-size", type=int, default=8)
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument(
        "--label",
        default="Dataset",
        help="Dataset label used in plot titles (for example, '500pA').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if args.tensor_frames <= 0 or args.read_chunk_size <= 0:
        raise ValueError("tensor and read chunk sizes must be positive")
    if args.height % 2:
        raise ValueError("detector height must be even")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dark_frame, "r") as dark_h5:
        dark = read_image(dark_h5, args.dark_dataset, np)
        valid = read_image(dark_h5, args.valid_mask_dataset, np) != 0
    if dark.shape != (args.height, args.width) or valid.shape != dark.shape:
        raise ValueError("dark frame or valid mask shape does not match requested frame shape")

    half = args.height // 2
    regions = (slice(args.edge_rows, half), slice(half, args.height - args.edge_rows))
    file_count = len(args.inputs)
    sums = np.zeros((3, file_count, 2, args.width), dtype=np.float64)
    counts = np.zeros((file_count, 2, args.width), dtype=np.float64)
    frame_counts = np.zeros(file_count, dtype=np.int64)

    for file_index, input_path in enumerate(args.inputs):
        data, info = load_dm4(input_path, args.reader)
        stack = normalize_to_frame_stack(data, args.frames_axis, args.height, args.width)
        print(
            f"{input_path}: reader={info['reader']} stack_shape={tuple(stack.shape)}",
            flush=True,
        )
        for tensor_start in range(0, stack.shape[0], args.tensor_frames):
            tensor_end = min(stack.shape[0], tensor_start + args.tensor_frames)
            frame_count = tensor_end - tensor_start
            raw_sum = np.zeros((args.height, args.width), dtype=np.float64)
            for read_start in range(tensor_start, tensor_end, args.read_chunk_size):
                read_end = min(tensor_end, read_start + args.read_chunk_size)
                raw_sum += materialize(stack[read_start:read_end], np).sum(
                    axis=0, dtype=np.float64
                )

            dark_subtracted = (raw_sum / frame_count).astype(np.float32) - dark
            variants = (
                dark_subtracted,
                subtract_imagej_blr(
                    dark_subtracted[None],
                    np,
                    args.blr_rows,
                    768,
                    args.zlp_group_columns,
                    args.core_group_columns,
                )[0],
                subtract_imagej_blr(
                    dark_subtracted[None], np, args.blr_rows, 768, 1, 1
                )[0],
            )
            for half_index, rows in enumerate(regions):
                region_valid = valid[rows]
                counts[file_index, half_index] += (
                    frame_count * np.count_nonzero(region_valid, axis=0)
                )
                for mode_index, corrected in enumerate(variants):
                    masked = np.where(region_valid, corrected[rows], 0.0)
                    sums[mode_index, file_index, half_index] += (
                        masked.sum(axis=0, dtype=np.float64) * frame_count
                    )
            frame_counts[file_index] += frame_count
            print(f"  batch {tensor_start}..{tensor_end - 1}", flush=True)
        del stack, data

    combined_sums = sums.sum(axis=(1, 2))
    combined_counts = counts.sum(axis=(0, 1))
    spectra = safe_divide(combined_sums, combined_counts[None], np)
    folded = np.stack([
        fold_zlp(sums[index].sum(axis=(0, 1)), counts.sum(axis=(0, 1)), np)
        for index in range(3)
    ])
    per_file_spectra = safe_divide(sums.sum(axis=2), counts.sum(axis=1)[None], np)
    per_file_folded = np.stack([
        np.stack([
            fold_zlp(sums[mode, file].sum(axis=0), counts[file].sum(axis=0), np)
            for file in range(file_count)
        ])
        for mode in range(3)
    ])

    rows = []
    for mode_index, (mode_name, mode_label) in enumerate(zip(MODE_NAMES, MODE_LABELS)):
        for region_name, profile, origin, width, first, last in (
            ("zlp_folded_tail", folded[mode_index], 0, args.zlp_group_columns, 76, 192),
            ("coreloss", spectra[mode_index], 768, args.core_group_columns, 784, 3840),
        ):
            metric = jump_metrics(profile, origin, width, first, last, np)
            rows.append({"mode": mode_name, "label": mode_label, "region": region_name, **metric})

    with h5py.File(args.output_dir / "blr_comparison.h5", "w") as output:
        output.create_dataset("spectra", data=spectra)
        output.create_dataset("folded_zlp", data=folded)
        output.create_dataset("per_file_spectra", data=per_file_spectra)
        output.create_dataset("per_file_folded_zlp", data=per_file_folded)
        output.create_dataset("source_frame_count", data=frame_counts)
        output.attrs["mode_names"] = json.dumps(MODE_NAMES)
        output.attrs["source_files"] = json.dumps([str(path) for path in args.inputs])

    with (args.output_dir / "boundary_jump_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = ("black", "tab:red", "tab:blue")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for index, (label, color) in enumerate(zip(MODE_LABELS, colors)):
        axes[0].plot(folded[index], color=color, linewidth=1.0, label=label)
        axes[1].plot(
            folded[index, 72:192], color=color, linewidth=1.0, label=label
        )
    axes[0].set_title(f"{args.label} folded ZLP under three BLR strategies")
    axes[1].set_title("Folded-ZLP plateau region (physical columns 72..191)")
    axes[1].set_xticks(np.arange(0, 120, 8), labels=np.arange(72, 192, 8))
    for axis in axes:
        axis.set_xlabel("Physical ZLP detector column modulo 192")
        axis.set_ylabel("Mean dark-subtracted detector value")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(args.output_dir / "zlp_blr_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    for index, (label, color) in enumerate(zip(MODE_LABELS, colors)):
        axes[0].plot(spectra[index, 768:], color=color, linewidth=0.75, label=label)
    axes[1].plot(
        spectra[1, 768:] - spectra[0, 768:],
        color="tab:red",
        linewidth=0.75,
        label="Grouped BLR - no BLR",
    )
    axes[1].plot(
        spectra[2, 768:] - spectra[0, 768:],
        color="tab:blue",
        linewidth=0.75,
        label="Columnwise BLR - no BLR",
    )
    for index, (label, color) in enumerate(zip(MODE_LABELS, colors)):
        axes[2].plot(
            np.abs(np.diff(spectra[index, 768:1280])),
            color=color,
            linewidth=0.75,
            label=label,
        )
    axes[0].set_title(f"{args.label} CoreLoss spectrum under three BLR strategies")
    axes[1].set_title("BLR-induced CoreLoss correction relative to no BLR")
    axes[2].set_title("Absolute adjacent-column jump, early CoreLoss")
    for axis in axes:
        axis.set_xlabel("CoreLoss-relative detector column")
        axis.set_ylabel("Detector value")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(args.output_dir / "coreloss_blr_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, region in zip(axes, ("zlp_folded_tail", "coreloss")):
        selected = [row for row in rows if row["region"] == region]
        axis.bar(
            np.arange(3),
            [row["boundary_to_inside_mean_ratio"] for row in selected],
            color=colors,
        )
        axis.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        axis.set_xticks(np.arange(3), labels=MODE_LABELS, rotation=15, ha="right")
        axis.set_title("ZLP columns 76..191" if region.startswith("zlp") else "CoreLoss")
        axis.set_ylabel("Boundary / within-group mean jump")
        axis.grid(alpha=0.2, axis="y")
    fig.savefig(args.output_dir / "blr_boundary_jump_ratios.png", dpi=180)
    plt.close(fig)

    summary = {
        "source_files": [str(path) for path in args.inputs],
        "dataset_label": args.label,
        "source_frame_counts": frame_counts.tolist(),
        "dark_frame": str(args.dark_frame),
        "dynamic_mask_applied": False,
        "static_valid_mask_applied": True,
        "modes": list(MODE_NAMES),
        "metrics": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
