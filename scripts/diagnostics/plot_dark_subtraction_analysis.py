#!/usr/bin/env python3
"""Create diagnostic plots for dark-subtracted STEM networking outputs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def configure_matplotlib_cache():
    mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
    xdg_cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def get_modules():
    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    return h5py, plt, TwoSlopeNorm, np


def read_first_frame(h5_file, dataset_name):
    dataset = h5_file[dataset_name]
    data = dataset[...]
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    if data.ndim == 2:
        return data
    raise ValueError(f"{dataset_name} must have shape [rows, cols] or [1, rows, cols]")


def robust_limits(frames, np, low=0.1, high=99.9):
    values = np.concatenate([frame[np.isfinite(frame)].ravel() for frame in frames])
    if values.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(values, [low, high])
    if vmin == vmax:
        vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin == vmax:
        vmin, vmax = -1.0, 1.0
    if vmin >= 0:
        vmin = -max(1.0, vmax * 0.05)
    if vmax <= 0:
        vmax = max(1.0, abs(vmin) * 0.05)
    return float(vmin), float(vmax)


def compute_stats(subtracted_dataset, invalid_mask):
    _, _, _, np = get_modules()
    num_frames = subtracted_dataset.shape[0]
    mean_projection = np.zeros(subtracted_dataset.shape[1:], dtype=np.float64)
    max_projection = np.full(subtracted_dataset.shape[1:], -np.inf, dtype=np.float32)
    min_projection = np.full(subtracted_dataset.shape[1:], np.inf, dtype=np.float32)
    stats = []

    for frame_index in range(num_frames):
        frame = subtracted_dataset[frame_index]
        finite = frame[np.isfinite(frame)]
        percentiles = np.percentile(finite, [0.1, 1, 50, 99, 99.9])
        negative_pixels = int(np.count_nonzero(frame < 0))
        positive_pixels = int(np.count_nonzero(frame > 0))

        row_max, col_max = np.unravel_index(np.argmax(frame), frame.shape)
        row_min, col_min = np.unravel_index(np.argmin(frame), frame.shape)

        invalid_nonzero = 0
        invalid_mean_abs = 0.0
        invalid_max_abs = 0.0
        if invalid_mask is not None:
            invalid_values = frame[invalid_mask]
            invalid_nonzero = int(np.count_nonzero(invalid_values))
            if invalid_values.size:
                invalid_abs = np.abs(invalid_values)
                invalid_mean_abs = float(np.mean(invalid_abs))
                invalid_max_abs = float(np.max(invalid_abs))

        stats.append(
            {
                "frame": frame_index,
                "min": float(np.min(finite)),
                "p0_1": float(percentiles[0]),
                "p1": float(percentiles[1]),
                "median": float(percentiles[2]),
                "p99": float(percentiles[3]),
                "p99_9": float(percentiles[4]),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "sum": float(np.sum(finite)),
                "negative_fraction": negative_pixels / frame.size,
                "positive_fraction": positive_pixels / frame.size,
                "max_row": int(row_max),
                "max_col": int(col_max),
                "min_row": int(row_min),
                "min_col": int(col_min),
                "invalid_nonzero": invalid_nonzero,
                "invalid_mean_abs": invalid_mean_abs,
                "invalid_max_abs": invalid_max_abs,
            }
        )

        mean_projection += frame / num_frames
        max_projection = np.maximum(max_projection, frame)
        min_projection = np.minimum(min_projection, frame)

    return stats, mean_projection.astype(np.float32), max_projection, min_projection


def write_stats_csv(stats, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)


def plot_all_frames(subtracted_dataset, stats, output_path, dpi):
    _, plt, TwoSlopeNorm, np = get_modules()
    num_frames = subtracted_dataset.shape[0]
    cols = 4
    rows = int(np.ceil(num_frames / cols))
    selected_frames = [subtracted_dataset[i] for i in range(num_frames)]
    vmin, vmax = robust_limits(selected_frames, np)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 4.2 * rows), squeeze=False)
    last_image = None
    for frame_index, frame in enumerate(selected_frames):
        ax = axes[frame_index // cols][frame_index % cols]
        last_image = ax.imshow(frame, cmap="coolwarm", norm=norm, aspect="auto", interpolation="none")
        ax.set_title(
            f"Output {frame_index}: mean={stats[frame_index]['mean']:.1f}, "
            f"max={stats[frame_index]['max']:.0f}"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    for idx in range(num_frames, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.75, label="dark-subtracted sum")
    fig.suptitle("All Dark-Subtracted Summed Outputs", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_stats(stats, output_path, dpi):
    _, plt, _, np = get_modules()
    frames = np.array([row["frame"] for row in stats])
    sums = np.array([row["sum"] for row in stats])
    means = np.array([row["mean"] for row in stats])
    mins = np.array([row["min"] for row in stats])
    maxes = np.array([row["max"] for row in stats])
    neg = 100.0 * np.array([row["negative_fraction"] for row in stats])
    pos = 100.0 * np.array([row["positive_fraction"] for row in stats])
    invalid_nonzero = np.array([row["invalid_nonzero"] for row in stats])
    invalid_mean_abs = np.array([row["invalid_mean_abs"] for row in stats])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(frames, sums, marker="o", label="sum")
    axes[0, 0].set_title("Total Signal Per Output")
    axes[0, 0].set_xlabel("Output index")
    axes[0, 0].set_ylabel("sum")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(frames, means, marker="o", color="tab:green")
    axes[0, 1].set_title("Mean Pixel Value")
    axes[0, 1].set_xlabel("Output index")
    axes[0, 1].set_ylabel("mean")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(frames, maxes, marker="o", label="max")
    axes[1, 0].plot(frames, mins, marker="o", label="min")
    axes[1, 0].set_title("Signed Range Confirms Float Dark Subtraction")
    axes[1, 0].set_xlabel("Output index")
    axes[1, 0].set_ylabel("value")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(frames, pos, marker="o", label="positive pixels (%)")
    axes[1, 1].plot(frames, neg, marker="o", label="negative pixels (%)")
    axes[1, 1].plot(frames, invalid_nonzero / max(1, invalid_nonzero.max()) * max(pos.max(), 1),
                    marker="x", linestyle="--", label="invalid-mask nonzero (scaled)")
    axes[1, 1].set_title("Sparse Signed Output And Mask Diagnostic")
    axes[1, 1].set_xlabel("Output index")
    axes[1, 1].set_ylabel("percent / scaled count")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.25)

    twin = axes[1, 1].twinx()
    twin.plot(frames, invalid_mean_abs, color="tab:red", alpha=0.35, label="invalid mean |value|")
    twin.set_ylabel("invalid-mask mean |value|")

    fig.suptitle("Dark-Subtracted Output Statistics", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_projections(mean_projection, max_projection, min_projection, output_path, dpi):
    _, plt, TwoSlopeNorm, np = get_modules()
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    projection_items = [
        ("Mean Across Outputs", mean_projection, "coolwarm"),
        ("Max Projection", max_projection, "magma"),
        ("Min Projection", min_projection, "viridis"),
    ]

    for ax, (title, image, cmap) in zip(axes, projection_items):
        if title == "Mean Across Outputs":
            vmin, vmax = robust_limits([image], np, low=1, high=99)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            im = ax.imshow(image, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
        else:
            vmin, vmax = robust_limits([image], np, low=1, high=99.9)
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Dark-Subtracted Output Projections", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(stats, dark_file, subtracted_file, output_path, dark_shape, subtracted_shape, invalid_pixels):
    nonzero_invalid_frames = sum(1 for row in stats if row["invalid_nonzero"] > 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"dark_file: {dark_file}\n")
        out.write(f"subtracted_file: {subtracted_file}\n")
        out.write(f"dark_shape: {dark_shape}\n")
        out.write(f"subtracted_shape: {subtracted_shape}\n")
        out.write(f"invalid_pixels_from_blinker_mask: {invalid_pixels}\n")
        out.write(f"outputs_with_nonzero_invalid_pixels: {nonzero_invalid_frames}\n")
        out.write("\n")
        out.write("Interpretation hints:\n")
        out.write("- subtracted output dtype is float32 and contains negative values, which confirms the subtraction path emitted signed corrected data.\n")
        out.write("- 20 output frames matches two receivers x ten 128-frame reduced batches for a 1280-frame-per-receiver run.\n")
        out.write("- nonzero values at invalid-mask positions indicate the valid-pixel mask was not applied at runtime for this file.\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze dark-subtracted HDF5 output.")
    parser.add_argument("--dark-file", type=Path, default=Path("walking_dot_dark_frame.h5"))
    parser.add_argument("--subtracted-file", type=Path, default=Path("subtracted_frames_out_net.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("dark_subtraction_analysis"))
    parser.add_argument("--dataset", default="/processed")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    h5py, _, _, np = get_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dark_file, "r") as dark_h5, h5py.File(args.subtracted_file, "r") as sub_h5:
        dark_frame = read_first_frame(dark_h5, args.dataset)
        blinker_mask = read_first_frame(dark_h5, "/blinker_mask").astype(bool) if "/blinker_mask" in dark_h5 else None
        invalid_mask = blinker_mask
        invalid_pixels = int(np.count_nonzero(invalid_mask)) if invalid_mask is not None else 0

        subtracted = sub_h5[args.dataset]
        stats, mean_projection, max_projection, min_projection = compute_stats(subtracted, invalid_mask)

        write_stats_csv(stats, args.output_dir / "subtracted_frame_stats.csv")
        plot_all_frames(subtracted, stats, args.output_dir / "subtracted_frames_all.png", args.dpi)
        plot_stats(stats, args.output_dir / "subtracted_frame_stats.png", args.dpi)
        plot_projections(
            mean_projection,
            max_projection,
            min_projection,
            args.output_dir / "subtracted_output_projections.png",
            args.dpi,
        )
        write_summary(
            stats,
            args.dark_file,
            args.subtracted_file,
            args.output_dir / "analysis_summary.txt",
            dark_frame.shape,
            subtracted.shape,
            invalid_pixels,
        )

    print(f"Wrote analysis plots and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
