#!/usr/bin/env python3
"""Plot a dark-frame HDF5 file created by make_dark_frame.py."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from textwrap import wrap


def normalize_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return "/processed"
    return dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"


def get_plot_modules():
    mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
    xdg_cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return h5py, plt, np


def read_single_frame(h5_file, dataset_path: str, required: bool = True):
    dataset_path = normalize_dataset_path(dataset_path)
    if dataset_path not in h5_file:
        if required:
            raise KeyError(f"dataset {dataset_path} not found")
        return None, None

    dataset = h5_file[dataset_path]
    data = dataset[...]
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    elif data.ndim != 2:
        raise ValueError(f"dataset {dataset_path} must have shape [rows, cols] or [1, rows, cols]")

    return data, dataset


def robust_limits(array, low_percentile=1.0, high_percentile=99.0):
    _, _, np = get_plot_modules()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [low_percentile, high_percentile])
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def format_scalar(value):
    try:
        if hasattr(value, "item"):
            value = value.item()
    except ValueError:
        pass
    return str(value)


def summarize_frame(name, array):
    _, _, np = get_plot_modules()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [f"{name}: no finite values"]
    return [
        f"{name} min: {float(np.min(finite)):.3f}",
        f"{name} mean: {float(np.mean(finite)):.3f}",
        f"{name} max: {float(np.max(finite)):.3f}",
    ]


def metadata_lines(file_path, dark_dataset, dark_frame, dark_stddev, blinker_mask, valid_mask):
    _, _, np = get_plot_modules()

    attrs = dark_dataset.attrs
    lines = [
        f"file: {file_path.name}",
        f"dark shape: {tuple(dark_frame.shape)}",
        f"dark dtype: {dark_frame.dtype}",
    ]
    lines.extend(summarize_frame("dark", dark_frame))

    for key in (
        "frames_averaged",
        "start_frame",
        "source_dataset",
        "source_shape",
        "blinker_std_threshold",
        "repair_neighbors",
        "edge_rows",
        "blinker_pixels",
        "repaired_blinker_pixels",
        "unrepaired_blinker_pixels",
    ):
        if key in attrs:
            lines.append(f"{key}: {format_scalar(attrs[key])}")

    total_pixels = dark_frame.size
    if blinker_mask is not None:
        blinker_pixels = int(np.count_nonzero(blinker_mask))
        lines.append(f"blinker fraction: {100.0 * blinker_pixels / total_pixels:.3f}%")
    if valid_mask is not None:
        valid_pixels = int(np.count_nonzero(valid_mask))
        lines.append(f"valid pixel fraction: {100.0 * valid_pixels / total_pixels:.3f}%")
    if dark_stddev is not None:
        lines.extend(summarize_frame("stddev", dark_stddev))

    if "source_file" in attrs:
        source = format_scalar(attrs["source_file"])
        wrapped = wrap(f"source_file: {source}", width=58)
        lines.extend(wrapped)

    return lines


def show_image(ax, array, title, cmap, low_percentile=1.0, high_percentile=99.0, binary=False):
    if array is None:
        ax.axis("off")
        ax.set_title(f"{title} not present")
        return None

    if binary:
        image = ax.imshow(array, cmap="gray", aspect="auto", interpolation="none", vmin=0, vmax=1)
    else:
        vmin, vmax = robust_limits(array, low_percentile, high_percentile)
        image = ax.imshow(array, cmap=cmap, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return image


def plot_dark_frame(
    file_path,
    output_path,
    dark_dataset="/processed",
    stddev_dataset="/dark_stddev",
    blinker_mask_dataset="/blinker_mask",
    valid_mask_dataset="/valid_pixel_mask",
    cmap="magma",
    dpi=150,
):
    h5py, plt, np = get_plot_modules()
    file_path = Path(file_path)
    output_path = Path(output_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    with h5py.File(file_path, "r") as h5_file:
        dark_frame, dark_h5_dataset = read_single_frame(h5_file, dark_dataset, required=True)
        dark_stddev, _ = read_single_frame(h5_file, stddev_dataset, required=False)
        blinker_mask, _ = read_single_frame(h5_file, blinker_mask_dataset, required=False)
        valid_mask, _ = read_single_frame(h5_file, valid_mask_dataset, required=False)

        if blinker_mask is not None:
            blinker_mask = blinker_mask.astype(bool)
        if valid_mask is not None:
            valid_mask = valid_mask.astype(bool)

        fig = plt.figure(figsize=(18, 12))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
        ]

        dark_image = show_image(axes[0], dark_frame, "Repaired Dark Frame (/processed)", cmap)
        if dark_image is not None:
            fig.colorbar(dark_image, ax=axes[0], shrink=0.8)

        stddev_image = show_image(axes[1], dark_stddev, "Dark Temporal StdDev (/dark_stddev)", "viridis")
        if stddev_image is not None:
            fig.colorbar(stddev_image, ax=axes[1], shrink=0.8)

        mask_to_plot = blinker_mask
        mask_title = "Blinker Mask (/blinker_mask, white=flagged)"
        if mask_to_plot is None and valid_mask is not None:
            mask_to_plot = np.logical_not(valid_mask)
            mask_title = "Invalid Pixels from /valid_pixel_mask"
        mask_image = show_image(axes[2], mask_to_plot, mask_title, "gray", binary=True)
        if mask_image is not None:
            fig.colorbar(mask_image, ax=axes[2], shrink=0.8, ticks=[0, 1])

        axes[3].axis("off")
        lines = metadata_lines(file_path, dark_h5_dataset, dark_frame, dark_stddev, blinker_mask, valid_mask)
        axes[3].set_title("Dark-Frame Build Metadata", loc="left")
        axes[3].text(
            0.0,
            0.98,
            "\n".join(lines),
            transform=axes[3].transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        fig.suptitle(f"Dark Frame Overview: {file_path.name}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved dark-frame overview to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot dark-frame HDF5 output from make_dark_frame.py.")
    parser.add_argument("file", type=Path, help="Dark-frame HDF5 file.")
    parser.add_argument("--output", type=Path, default=Path("dark_frame_overview.png"), help="Output image path.")
    parser.add_argument("--dark-dataset", default="/processed", help="Dark frame dataset path.")
    parser.add_argument("--stddev-dataset", default="/dark_stddev", help="Dark stddev dataset path.")
    parser.add_argument("--blinker-mask-dataset", default="/blinker_mask", help="Blinker mask dataset path.")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask", help="Valid mask dataset path.")
    parser.add_argument("--cmap", default="magma", help="Colormap for dark-frame image.")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    args = parser.parse_args()

    plot_dark_frame(
        args.file,
        args.output,
        dark_dataset=args.dark_dataset,
        stddev_dataset=args.stddev_dataset,
        blinker_mask_dataset=args.blinker_mask_dataset,
        valid_mask_dataset=args.valid_mask_dataset,
        cmap=args.cmap,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
