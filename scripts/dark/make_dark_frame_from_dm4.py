#!/usr/bin/env python3
"""Build a blinker-aware dark calibration directly from DM4 frame stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack
from make_dark_frame import repair_blinker_pixels, write_dark_frame
from plot_nio_processing_analysis import (
    configure_matplotlib_cache,
    detector_regions,
    robust_limits,
    subtract_imagej_blr,
    symmetric_limit,
)


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def finalize_statistics(total, sumsq, count: int, np):
    mean = total / count
    variance = np.maximum(sumsq / count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def save_dark_plots(output_dir: Path,
                    raw_mean,
                    repaired_mean,
                    stddev,
                    blinker_mask,
                    source_means,
                    source_labels,
                    edge_rows: int,
                    blr_rows: int,
                    zlp_width: int,
                    zlp_period: int,
                    zlp_group_columns: int,
                    core_group_columns: int,
                    plt,
                    TwoSlopeNorm,
                    np):
    mean_vmin, mean_vmax = robust_limits(repaired_mean, np, 0.5, 99.5)
    _, std_vmax = robust_limits(stddev, np, 0.0, 99.5)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    images = [
        axes[0, 0].imshow(raw_mean, cmap="magma", vmin=mean_vmin, vmax=mean_vmax, aspect="auto"),
        axes[0, 1].imshow(repaired_mean, cmap="magma", vmin=mean_vmin, vmax=mean_vmax, aspect="auto"),
        axes[1, 0].imshow(stddev, cmap="viridis", vmin=0.0, vmax=std_vmax, aspect="auto"),
        axes[1, 1].imshow(blinker_mask, cmap="gray_r", vmin=0, vmax=1, aspect="auto"),
    ]
    titles = (
        "Raw all-frame dark mean",
        "Blinker-repaired dark mean",
        "Temporal dark standard deviation",
        f"Blinker mask ({int(blinker_mask.sum())} pixels)",
    )
    for axis, image, title in zip(axes.ravel(), images, titles):
        axis.set_title(title)
        axis.set_xlabel("Detector column")
        axis.set_ylabel("Detector row")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    fig.savefig(output_dir / "dark_overview.png", dpi=180)
    plt.close(fig)

    source_residuals = []
    for source_mean in source_means:
        delta = (source_mean - raw_mean)[None, :, :]
        source_residuals.append(
            subtract_imagej_blr(
                delta,
                np,
                blr_rows,
                zlp_width,
                zlp_group_columns,
                core_group_columns,
            )[0]
        )
    source_residuals = np.stack(source_residuals)
    top_rows, bottom_rows = detector_regions(raw_mean.shape[0], edge_rows)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    for residual, label in zip(source_residuals, source_labels):
        axes[0].plot(residual[top_rows].mean(axis=0), linewidth=0.75, label=label)
        axes[1].plot(residual[bottom_rows].mean(axis=0), linewidth=0.75, label=label)
        top_folded = residual[top_rows, :zlp_width].mean(axis=0).reshape(
            zlp_width // zlp_period, zlp_period
        ).mean(axis=0)
        bottom_folded = residual[bottom_rows, :zlp_width].mean(axis=0).reshape(
            zlp_width // zlp_period, zlp_period
        ).mean(axis=0)
        axes[2].plot(top_folded, linewidth=0.8, alpha=0.8, label=f"{label} top")
        axes[2].plot(bottom_folded, linewidth=0.8, linestyle="--", alpha=0.8, label=f"{label} bottom")
    for axis, title in zip(
        axes,
        (
            "Dark source-file residual column means: top imaging half",
            "Dark source-file residual column means: bottom imaging half",
            "Folded ZLP dark residuals by source file and detector half",
        ),
    ):
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("Detector column" if axis is not axes[2] else "Physical ZLP column modulo period")
        axis.set_ylabel("BLR-corrected residual")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    axes[0].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[1].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    fig.savefig(output_dir / "dark_source_residual_profiles.png", dpi=180)
    plt.close(fig)

    limit = symmetric_limit(source_residuals, np, 99.5)
    columns = min(2, len(source_residuals))
    rows = int(np.ceil(len(source_residuals) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 4.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, residual, label in zip(axes, source_residuals, source_labels):
        image = axis.imshow(
            residual,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
            aspect="auto",
        )
        axis.set_title(label)
        axis.set_xlabel("Detector column")
        axis.set_ylabel("Detector row")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    for axis in axes[len(source_residuals):]:
        axis.axis("off")
    fig.suptitle("BLR-corrected dark source means relative to all-frame dark")
    fig.savefig(output_dir / "dark_source_residual_maps.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio")
    parser.add_argument("--frames-axis", type=int, default=None)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--read-chunk-size", type=int, default=8)
    parser.add_argument("--blinker-std-threshold", type=float, default=500.0)
    parser.add_argument("--repair-neighbors", type=int, default=10)
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-period", type=int, default=192)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be positive")

    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = np.zeros((args.height, args.width), dtype=np.float64)
    sumsq = np.zeros_like(total)
    source_means = []
    source_stddevs = []
    source_counts = []
    source_labels = []
    total_count = 0

    for path in args.inputs:
        data, info = load_dm4(path, args.reader)
        stack = normalize_to_frame_stack(data, args.frames_axis, args.height, args.width)
        if stack.shape[1:] != (args.height, args.width):
            raise ValueError(f"unexpected frame shape {stack.shape[1:]} in {path}")
        source_total = np.zeros_like(total)
        source_sumsq = np.zeros_like(total)
        for start in range(0, stack.shape[0], args.read_chunk_size):
            end = min(stack.shape[0], start + args.read_chunk_size)
            block = materialize(stack[start:end], np).astype(np.float64, copy=False)
            source_total += block.sum(axis=0)
            source_sumsq += np.square(block).sum(axis=0)
        source_mean, source_stddev = finalize_statistics(
            source_total, source_sumsq, int(stack.shape[0]), np
        )
        source_means.append(source_mean)
        source_stddevs.append(source_stddev)
        source_counts.append(int(stack.shape[0]))
        source_labels.append(path.stem)
        total += source_total
        sumsq += source_sumsq
        total_count += int(stack.shape[0])
        print(
            f"{path}: reader={info['reader']} frames={stack.shape[0]} "
            f"shape={tuple(stack.shape[1:])}",
            flush=True,
        )
        del stack, data, source_total, source_sumsq

    raw_mean, stddev = finalize_statistics(total, sumsq, total_count, np)
    repaired, blinker_mask, repaired_count, unrepaired_count = repair_blinker_pixels(
        raw_mean,
        stddev,
        args.blinker_std_threshold,
        args.repair_neighbors,
        args.edge_rows,
    )
    valid_mask = ~blinker_mask
    write_dark_frame(
        args.output,
        "/processed",
        repaired,
        raw_mean,
        stddev,
        blinker_mask,
        valid_mask,
        total_count,
        Path(";".join(str(path) for path in args.inputs)),
        "/DM4",
        0,
        (total_count, args.height, args.width),
        args.blinker_std_threshold,
        args.repair_neighbors,
        args.edge_rows,
        repaired_count,
        unrepaired_count,
        "/dark_stddev",
        "/blinker_mask",
        "/valid_pixel_mask",
        "/raw_dark_mean",
        None,
    )

    source_means = np.stack(source_means)
    source_stddevs = np.stack(source_stddevs)
    with h5py.File(args.output, "a") as output:
        output.create_dataset("source_file_mean", data=source_means)
        output.create_dataset("source_file_stddev", data=source_stddevs)
        output.create_dataset("source_file_frame_count", data=np.asarray(source_counts, dtype=np.int64))
        output.attrs["source_files"] = np.array(
            [str(path) for path in args.inputs],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    save_dark_plots(
        args.output_dir,
        raw_mean,
        repaired,
        stddev,
        blinker_mask,
        source_means,
        source_labels,
        args.edge_rows,
        args.blr_rows,
        args.zlp_width,
        args.zlp_period,
        args.zlp_group_columns,
        args.core_group_columns,
        plt,
        TwoSlopeNorm,
        np,
    )
    summary = {
        "source_files": [str(path) for path in args.inputs],
        "source_frame_counts": source_counts,
        "total_frames": total_count,
        "frame_shape": [args.height, args.width],
        "blinker_std_threshold": args.blinker_std_threshold,
        "blinker_pixels": int(blinker_mask.sum()),
        "repaired_blinker_pixels": repaired_count,
        "unrepaired_blinker_pixels": unrepaired_count,
        "output_hdf5": str(args.output),
        "plots": [
            "dark_overview.png",
            "dark_source_residual_profiles.png",
            "dark_source_residual_maps.png",
        ],
    }
    (args.output_dir / "dark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
