#!/usr/bin/env python3
"""Plot raw single-frame QC figures with BLR-row diagnostics.

This is intended for detector-effect triage: inspect individual raw frames and,
optionally, the per-frame BLR baseline estimates that grouped BLR would subtract.
The figures are written as TIFF files with nearest-neighbor image rendering so
individual detector pixels are not visually interpolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stem_analysis.dm4 import load_dm4, normalize_to_frame_stack
from stem_analysis.hdf5 import read_single_image
from stem_analysis.plotting import configure_matplotlib_cache, robust_limits
from stem_analysis.processing import compute_blr_baseline


def parse_indices(text: str | None, frame_count: int) -> list[int]:
    if text is None or text.lower() in {"auto", "auto3"}:
        if frame_count == 1:
            return [0]
        return sorted({0, frame_count // 2, frame_count - 1})

    indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not indices:
        raise ValueError("at least one frame index is required")
    invalid = [index for index in indices if index < 0 or index >= frame_count]
    if invalid:
        raise ValueError(f"frame indices outside [0, {frame_count - 1}]: {invalid}")
    return indices


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def read_hdf5_stack(path: Path, dataset_name: str):
    import h5py

    h5_file = h5py.File(path, "r")
    try:
        dataset = h5_file[dataset_name]
    except Exception:
        h5_file.close()
        raise
    if dataset.ndim == 2:
        frame_count = 1
        shape = (1, int(dataset.shape[0]), int(dataset.shape[1]))
    elif dataset.ndim == 3:
        frame_count = int(dataset.shape[0])
        shape = tuple(int(value) for value in dataset.shape)
    else:
        h5_file.close()
        raise ValueError(f"{path}:{dataset_name} must be 2D or 3D, got {dataset.shape}")
    return h5_file, dataset, frame_count, shape


def read_frame_from_hdf5(dataset, frame_index: int, np):
    if dataset.ndim == 2:
        return dataset[...].astype(np.float32, copy=False)
    return dataset[frame_index, :, :].astype(np.float32, copy=False)


def load_input_stack(args):
    suffix = args.input.suffix.lower()
    if args.input_format == "hdf5" or (
        args.input_format == "auto" and suffix in {".h5", ".hdf5", ".hdf"}
    ):
        return ("hdf5",) + read_hdf5_stack(args.input, args.dataset_name)

    data, info = load_dm4(args.input, args.reader)
    stack = normalize_to_frame_stack(data, args.frames_axis, args.height, args.width)
    shape = tuple(int(value) for value in stack.shape)
    return "dm4", info, stack, int(shape[0]), shape


def expand_baseline(baseline, width: int, zlp_width: int, zlp_group: int, core_group: int, np):
    """Expand one frame/half grouped baseline bins into per-column values."""
    zlp_bins = zlp_width // zlp_group
    parts = []
    if zlp_width:
        parts.append(np.repeat(baseline[:zlp_bins], zlp_group))
    if zlp_width < width:
        parts.append(np.repeat(baseline[zlp_bins:], core_group))
    return np.concatenate(parts).astype(np.float32, copy=False)


def compute_baseline_profiles(frame, dark, args, np):
    if dark is not None:
        corrected_input = frame.astype(np.float32, copy=False) - dark
    else:
        corrected_input = frame.astype(np.float32, copy=False)

    block = corrected_input[None, :, :]
    grouped = compute_blr_baseline(
        block,
        args.blr_rows,
        args.zlp_width,
        args.zlp_group_columns,
        args.core_group_columns,
        np,
    )[0]
    columnwise = compute_blr_baseline(
        block,
        args.blr_rows,
        args.zlp_width,
        1,
        1,
        np,
    )[0]

    height, width = frame.shape
    return {
        "top_grouped": expand_baseline(
            grouped[0], width, args.zlp_width, args.zlp_group_columns, args.core_group_columns, np
        ),
        "bottom_grouped": expand_baseline(
            grouped[1], width, args.zlp_width, args.zlp_group_columns, args.core_group_columns, np
        ),
        "top_columnwise": columnwise[0].astype(np.float32, copy=False),
        "bottom_columnwise": columnwise[1].astype(np.float32, copy=False),
        "source": "dark-subtracted" if dark is not None else "raw",
    }


def folded_zlp(profile, zlp_width: int, zlp_repeat_width: int):
    return profile[:zlp_width].reshape(zlp_width // zlp_repeat_width, zlp_repeat_width).mean(axis=0)


def plot_frame_qc(output_path, frame, profiles, title, args, plt, np):
    image = frame.astype(np.float32, copy=False)
    vmin, vmax = robust_limits(image, np, args.image_percentile_low, args.image_percentile_high)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.8, 1.0, 1.0])
    ax_image = fig.add_subplot(gs[0, :])
    ax_zlp = fig.add_subplot(gs[1, 0])
    ax_core = fig.add_subplot(gs[1, 1])
    ax_zlp_phase = fig.add_subplot(gs[2, 0])
    ax_core_phase = fig.add_subplot(gs[2, 1])

    im = ax_image.imshow(
        image,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    half_height = image.shape[0] // 2
    for y in (args.blr_rows - 0.5, half_height - 0.5, image.shape[0] - args.blr_rows - 0.5):
        ax_image.axhline(y, color="cyan", linewidth=0.8, alpha=0.7)
    ax_image.axvline(args.zlp_width - 0.5, color="white", linewidth=0.8, alpha=0.7)
    ax_image.set_title(f"{title}\nraw frame, display percentiles {args.image_percentile_low:g}-{args.image_percentile_high:g}")
    ax_image.set_xlabel("Column")
    ax_image.set_ylabel("Row")
    fig.colorbar(im, ax=ax_image, shrink=0.75, label="Raw pixel value")

    zlp_x = np.arange(args.zlp_repeat_width)
    for half_name, color_prefix in (("top", "tab:blue"), ("bottom", "tab:orange")):
        col = folded_zlp(profiles[f"{half_name}_columnwise"], args.zlp_width, args.zlp_repeat_width)
        grp = folded_zlp(profiles[f"{half_name}_grouped"], args.zlp_width, args.zlp_repeat_width)
        ax_zlp.plot(zlp_x, col, linewidth=0.8, color=color_prefix, alpha=0.55, label=f"{half_name} per-column")
        ax_zlp.plot(zlp_x, grp, linewidth=1.2, color=color_prefix, linestyle="--", label=f"{half_name} grouped")
        residual = col - grp
        phase = np.array([residual[i::args.zlp_group_columns].mean() for i in range(args.zlp_group_columns)])
        ax_zlp_phase.plot(np.arange(args.zlp_group_columns), phase, marker="o", color=color_prefix, label=half_name)

    core_limit = min(args.core_plot_columns, image.shape[1] - args.zlp_width)
    core_x = np.arange(core_limit)
    for half_name, color_prefix in (("top", "tab:blue"), ("bottom", "tab:orange")):
        col = profiles[f"{half_name}_columnwise"][args.zlp_width:args.zlp_width + core_limit]
        grp = profiles[f"{half_name}_grouped"][args.zlp_width:args.zlp_width + core_limit]
        ax_core.plot(core_x, col, linewidth=0.8, color=color_prefix, alpha=0.55, label=f"{half_name} per-column")
        ax_core.plot(core_x, grp, linewidth=1.2, color=color_prefix, linestyle="--", label=f"{half_name} grouped")
        residual = col - grp
        phase = np.array([residual[i::args.core_group_columns].mean() for i in range(args.core_group_columns)])
        ax_core_phase.plot(np.arange(args.core_group_columns), phase, marker="o", color=color_prefix, label=half_name)

    for boundary in range(args.core_group_columns, core_limit, args.core_group_columns):
        ax_core.axvline(boundary, color="gray", linewidth=0.5, alpha=0.18)

    ax_zlp.set_title(f"Folded ZLP BLR estimate from {profiles['source']} BLR rows")
    ax_zlp.set_xlabel(f"ZLP physical column modulo {args.zlp_repeat_width}")
    ax_zlp.set_ylabel("Baseline estimate")
    ax_zlp.grid(alpha=0.2)
    ax_zlp.legend(ncol=2, fontsize=8)

    ax_core.set_title(f"CoreLoss BLR estimate, first {core_limit} columns")
    ax_core.set_xlabel("CoreLoss-relative column")
    ax_core.set_ylabel("Baseline estimate")
    ax_core.grid(alpha=0.2)
    ax_core.legend(ncol=2, fontsize=8)

    ax_zlp_phase.axhline(0.0, color="black", linewidth=0.8)
    ax_zlp_phase.set_title("Mean ZLP residual by position inside 4-column group")
    ax_zlp_phase.set_xlabel("Phase")
    ax_zlp_phase.set_ylabel("Per-column minus grouped")
    ax_zlp_phase.set_xticks(np.arange(args.zlp_group_columns))
    ax_zlp_phase.grid(alpha=0.2)
    ax_zlp_phase.legend(fontsize=8)

    ax_core_phase.axhline(0.0, color="black", linewidth=0.8)
    ax_core_phase.set_title("Mean CoreLoss residual by position inside 16-column group")
    ax_core_phase.set_xlabel("Phase")
    ax_core_phase.set_ylabel("Per-column minus grouped")
    ax_core_phase.set_xticks(np.arange(args.core_group_columns))
    ax_core_phase.grid(alpha=0.2)
    ax_core_phase.legend(fontsize=8)

    fig.savefig(output_path, dpi=args.dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def write_raw_tiff(output_path, frame, np):
    from PIL import Image

    if np.issubdtype(frame.dtype, np.integer) and frame.min() >= 0 and frame.max() <= 65535:
        image = Image.fromarray(frame.astype(np.uint16, copy=False))
    else:
        image = Image.fromarray(frame.astype(np.float32, copy=False), mode="F")
    image.save(output_path, compression="tiff_lzw")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="DM4 or HDF5 frame-stack files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-format", choices=("auto", "dm4", "hdf5"), default="auto")
    parser.add_argument("--dataset-name", default="/frames", help="HDF5 input dataset.")
    parser.add_argument("--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--frames-axis", type=int, default=None)
    parser.add_argument("--frame-indices", default=None, help="Comma-separated indices, or auto/auto3.")
    parser.add_argument("--dark-frame", type=Path, default=None)
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-repeat-width", type=int, default=192)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument("--core-plot-columns", type=int, default=512)
    parser.add_argument("--image-percentile-low", type=float, default=0.5)
    parser.add_argument("--image-percentile-high", type=float, default=99.95)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--save-raw-tiff", action="store_true")
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib_cache()

    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dark = None
    if args.dark_frame is not None:
        with h5py.File(args.dark_frame, "r") as dark_h5:
            dark = read_single_image(dark_h5, args.dark_dataset, np)
        if dark.shape != (args.height, args.width):
            raise ValueError(f"dark frame shape {dark.shape} does not match {(args.height, args.width)}")

    summary = {
        "inputs": [],
        "output_dir": str(args.output_dir),
        "dark_frame": str(args.dark_frame) if args.dark_frame else None,
        "dark_dataset": args.dark_dataset,
        "blr_rows": args.blr_rows,
        "zlp_width": args.zlp_width,
        "zlp_group_columns": args.zlp_group_columns,
        "core_group_columns": args.core_group_columns,
    }

    for input_path in args.inputs:
        loaded = load_input_stack(argparse.Namespace(**{**vars(args), "input": input_path}))
        input_summary = {"path": str(input_path), "plots": []}
        close_handle = None

        if loaded[0] == "hdf5":
            _, h5_file, dataset, frame_count, shape = loaded
            close_handle = h5_file

            def read_frame(index):
                return read_frame_from_hdf5(dataset, index, np)

            input_summary["reader"] = "hdf5"
        else:
            _, info, stack, frame_count, shape = loaded

            def read_frame(index):
                return materialize(stack[index], np).astype(np.float32, copy=False)

            input_summary["reader"] = info["reader"]

        input_summary["shape"] = [int(value) for value in shape]
        frame_indices = parse_indices(args.frame_indices, frame_count)
        input_summary["frame_indices"] = frame_indices
        stem = input_path.stem.replace(" ", "_").replace("/", "_")

        try:
            for frame_index in frame_indices:
                frame = read_frame(frame_index)
                profiles = compute_baseline_profiles(frame, dark, args, np)
                label = args.label or stem
                title = f"{label}: {input_path.name}, frame {frame_index}"
                figure_name = f"{stem}_frame_{frame_index:04d}_qc.tiff"
                figure_path = args.output_dir / figure_name
                plot_frame_qc(figure_path, frame, profiles, title, args, plt, np)
                entry = {
                    "frame_index": int(frame_index),
                    "qc_tiff": figure_name,
                    "raw_min": float(np.nanmin(frame)),
                    "raw_max": float(np.nanmax(frame)),
                    "raw_mean": float(np.nanmean(frame)),
                }
                if args.save_raw_tiff:
                    raw_name = f"{stem}_frame_{frame_index:04d}_raw_pixels.tiff"
                    write_raw_tiff(args.output_dir / raw_name, frame, np)
                    entry["raw_pixel_tiff"] = raw_name
                input_summary["plots"].append(entry)
                print(f"Wrote {figure_path}", flush=True)
        finally:
            if close_handle is not None:
                close_handle.close()

        summary["inputs"].append(input_summary)

    summary_path = args.output_dir / "raw_frame_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
