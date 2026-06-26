#!/usr/bin/env python3
"""Stream DM4 spectrum stacks through the offline correction chain.

The script computes final column spectra without materializing corrected frames.
For each tensor-sized frame batch it forms the raw mean, then applies static dark
subtraction, ImageJ BLR, valid-pixel masking, and the dynamic half-column mask.
This produces the same final mean as correcting every frame because dark
subtraction and BLR are linear and the runtime dynamic mask is shared by every
frame in a tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack
from offline_process_hdf5 import apply_dynamic_and_valid_mask
from plot_nio_processing_analysis import configure_matplotlib_cache, subtract_imagej_blr


def normalize_dataset_path(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def read_single_image(h5_file, dataset_path: str, np):
    data = h5_file[normalize_dataset_path(dataset_path)][...]
    if data.ndim == 2:
        return data.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32, copy=False)
    raise ValueError(f"expected a single image at {dataset_path}, got {data.shape}")


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def detector_regions(height: int, edge_rows: int):
    if height % 2 != 0:
        raise ValueError("detector height must be even")
    half = height // 2
    if edge_rows < 0 or edge_rows >= half:
        raise ValueError("edge_rows leaves no imaging pixels")
    return (slice(edge_rows, half), slice(half, height - edge_rows))


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def fold_zlp(sums, counts, zlp_width: int, zlp_period: int, np):
    if zlp_width % zlp_period != 0:
        raise ValueError("zlp_width must be divisible by zlp_period")
    repeats = zlp_width // zlp_period
    folded_sums = sums[..., :zlp_width].reshape(
        *sums.shape[:-1], repeats, zlp_period
    ).sum(axis=-2)
    folded_counts = counts[..., :zlp_width].reshape(
        *counts.shape[:-1], repeats, zlp_period
    ).sum(axis=-2)
    return safe_divide(folded_sums, folded_counts, np), folded_sums, folded_counts


def write_full_csv(path: Path,
                   valid_mean,
                   pipeline_mean,
                   static_valid_mean,
                   valid_counts,
                   pipeline_counts,
                   np):
    combined_valid = safe_divide(
        (valid_mean * valid_counts).sum(axis=0), valid_counts.sum(axis=0), np
    )
    combined_pipeline = safe_divide(
        (pipeline_mean * pipeline_counts).sum(axis=0),
        pipeline_counts.sum(axis=0),
        np,
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "output_column",
            "top_valid_mean",
            "bottom_valid_mean",
            "combined_valid_mean",
            "top_pipeline_zeroed_mean",
            "bottom_pipeline_zeroed_mean",
            "combined_pipeline_zeroed_mean",
            "top_static_mask_only_mean",
            "bottom_static_mask_only_mean",
            "top_valid_samples",
            "bottom_valid_samples",
        ])
        for column in range(valid_mean.shape[-1]):
            writer.writerow([
                column,
                valid_mean[0, column],
                valid_mean[1, column],
                combined_valid[column],
                pipeline_mean[0, column],
                pipeline_mean[1, column],
                combined_pipeline[column],
                static_valid_mean[0, column],
                static_valid_mean[1, column],
                int(valid_counts[0, column]),
                int(valid_counts[1, column]),
            ])


def save_plots(output_dir: Path,
               valid_mean,
               pipeline_mean,
               valid_sums,
               valid_counts,
               per_file_sums,
               per_file_counts,
               static_valid_sums,
               static_valid_counts,
               source_labels,
               zlp_width: int,
               zlp_period: int,
               mask_fraction,
               plt,
               np):
    combined_valid = safe_divide(
        valid_sums.sum(axis=0), valid_counts.sum(axis=0), np
    )
    columns = np.arange(valid_mean.shape[-1])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    axes[0].plot(columns, valid_mean[0], linewidth=0.8, label="top imaging half")
    axes[0].plot(columns, valid_mean[1], linewidth=0.8, label="bottom imaging half")
    axes[0].plot(columns, combined_valid, color="black", linewidth=1.1, label="combined")
    axes[0].axvline(zlp_width, color="tab:red", linestyle="--", linewidth=0.9)
    axes[0].set_title("Final corrected spectrum over all source frames")
    axes[0].set_xlabel("Detector output column (not yet calibrated to energy loss)")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(columns, pipeline_mean[0], linewidth=0.8, label="top imaging half")
    axes[1].plot(columns, pipeline_mean[1], linewidth=0.8, label="bottom imaging half")
    axes[1].axvline(zlp_width, color="tab:red", linestyle="--", linewidth=0.9)
    axes[1].set_title("Pipeline-equivalent mean with masked pixels retained as zeros")
    axes[1].set_xlabel("Detector output column")
    axes[1].set_ylabel("Mean corrected detector value")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(output_dir / "final_spectrum_full_columns.png", dpi=180)
    plt.close(fig)

    folded, _, _ = fold_zlp(
        valid_sums, valid_counts, zlp_width, zlp_period, np
    )
    folded_combined = safe_divide(
        fold_zlp(valid_sums.sum(axis=0), valid_counts.sum(axis=0), zlp_width, zlp_period, np)[1],
        fold_zlp(valid_sums.sum(axis=0), valid_counts.sum(axis=0), zlp_width, zlp_period, np)[2],
        np,
    )
    fig, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    axis.plot(folded[0], linewidth=1.0, label="top imaging half")
    axis.plot(folded[1], linewidth=1.0, label="bottom imaging half")
    axis.plot(folded_combined, color="black", linewidth=1.3, label="combined")
    axis.set_title("ZLP folded across four repeated 192-column readouts")
    axis.set_xlabel("Physical ZLP detector column modulo 192")
    axis.set_ylabel("Mean corrected detector value")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "final_spectrum_zlp_folded.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    core_columns = columns[zlp_width:]
    axis.plot(core_columns, valid_mean[0, zlp_width:], linewidth=0.8, label="top imaging half")
    axis.plot(core_columns, valid_mean[1, zlp_width:], linewidth=0.8, label="bottom imaging half")
    axis.plot(core_columns, combined_valid[zlp_width:], color="black", linewidth=1.1, label="combined")
    axis.set_title("CoreLoss detector-column spectrum")
    axis.set_xlabel("Detector output column (energy calibration unavailable)")
    axis.set_ylabel("Mean corrected detector value")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "final_spectrum_coreloss.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    for file_index, label in enumerate(source_labels):
        file_combined = safe_divide(
            per_file_sums[file_index].sum(axis=0),
            per_file_counts[file_index].sum(axis=0),
            np,
        )
        axes[0].plot(file_combined, linewidth=0.75, label=label)
        file_folded, _, _ = fold_zlp(
            per_file_sums[file_index].sum(axis=0),
            per_file_counts[file_index].sum(axis=0),
            zlp_width,
            zlp_period,
            np,
        )
        axes[1].plot(file_folded, linewidth=0.9, label=label)
    axes[0].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Corrected spectrum by source file")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[1].set_title("Folded ZLP spectrum by source file")
    axes[1].set_xlabel("Physical ZLP detector column modulo 192")
    axes[1].set_ylabel("Mean corrected detector value")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output_dir / "spectrum_by_source_file.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, 4.8), constrained_layout=True)
    axis.plot(mask_fraction[0], linewidth=0.8, label="top imaging half")
    axis.plot(mask_fraction[1], linewidth=0.8, label="bottom imaging half")
    axis.axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axis.set_title("Fraction of frame-row samples excluded by valid/dynamic masks")
    axis.set_xlabel("Detector output column")
    axis.set_ylabel("Excluded fraction")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "spectrum_mask_fraction.png", dpi=180)
    plt.close(fig)

    static_combined = safe_divide(
        static_valid_sums.sum(axis=0), static_valid_counts.sum(axis=0), np
    )
    dynamic_difference = combined_valid - static_combined
    static_folded, _, _ = fold_zlp(
        static_valid_sums.sum(axis=0),
        static_valid_counts.sum(axis=0),
        zlp_width,
        zlp_period,
        np,
    )
    dynamic_folded, _, _ = fold_zlp(
        valid_sums.sum(axis=0),
        valid_counts.sum(axis=0),
        zlp_width,
        zlp_period,
        np,
    )
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    axes[0].plot(static_combined, linewidth=0.9, label="static valid mask only")
    axes[0].plot(combined_valid, linewidth=0.9, label="static + dynamic masks")
    axes[0].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Effect of dynamic masking on the combined spectrum")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[0].legend()
    axes[1].plot(dynamic_difference, color="tab:red", linewidth=0.8)
    axes[1].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Dynamic-masked minus static-mask-only spectrum")
    axes[1].set_ylabel("Difference")
    axes[2].plot(static_folded, linewidth=1.0, label="static valid mask only")
    axes[2].plot(dynamic_folded, linewidth=1.0, label="static + dynamic masks")
    axes[2].set_title("Folded ZLP comparison")
    axes[2].set_xlabel("Physical ZLP detector column modulo 192")
    axes[2].set_ylabel("Mean corrected detector value")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "dynamic_mask_spectrum_comparison.png", dpi=180)
    plt.close(fig)


def save_batch_plots(output_dir: Path,
                     batch_sums,
                     batch_counts,
                     batch_source_indices,
                     source_labels,
                     zlp_width: int,
                     zlp_period: int,
                     batch_mask_pixels,
                     plt,
                     np):
    batch_combined = safe_divide(
        batch_sums.sum(axis=1), batch_counts.sum(axis=1), np
    )
    batch_count = batch_combined.shape[0]
    folded = []
    for index in range(batch_count):
        folded_mean, _, _ = fold_zlp(
            batch_sums[index].sum(axis=0),
            batch_counts[index].sum(axis=0),
            zlp_width,
            zlp_period,
            np,
        )
        folded.append(folded_mean)
    folded = np.stack(folded)
    peak_position = np.argmax(folded, axis=1)
    peak_height = np.max(folded, axis=1)
    zlp_area = np.nansum(np.maximum(folded, 0.0), axis=1)
    core_mean = np.nanmean(batch_combined[:, zlp_width:min(zlp_width + 768, batch_combined.shape[1])], axis=1)
    batch_axis = np.arange(batch_count)
    boundaries = np.flatnonzero(np.diff(batch_source_indices) != 0) + 0.5

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].plot(batch_axis, peak_height, marker="o", markersize=3, linewidth=0.8)
    axes[0].set_ylabel("Folded ZLP peak")
    axes[0].set_title("Per-128-frame batch spectrum diagnostics")
    axes[1].plot(batch_axis, peak_position, marker="o", markersize=3, linewidth=0.8)
    axes[1].set_ylabel("ZLP peak column")
    axes[2].plot(batch_axis, zlp_area, marker="o", markersize=3, linewidth=0.8, label="folded ZLP positive area")
    axes[2].plot(batch_axis, core_mean, marker="o", markersize=3, linewidth=0.8, label="early CoreLoss mean")
    axes[2].set_ylabel("Detector value")
    axes[2].legend()
    axes[3].plot(batch_axis, batch_mask_pixels, marker="o", markersize=3, linewidth=0.8)
    axes[3].set_ylabel("Masked pixels")
    axes[3].set_xlabel("Sequential 128-frame batch")
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(boundary, color="black", linestyle="--", linewidth=0.7)
        axis.grid(alpha=0.2)
    centers = []
    labels = []
    for source_index, label in enumerate(source_labels):
        positions = np.flatnonzero(batch_source_indices == source_index)
        if positions.size:
            centers.append(positions.mean())
            labels.append(label)
    secondary = axes[0].secondary_xaxis("top")
    secondary.set_xticks(centers)
    secondary.set_xticklabels(labels, rotation=15, ha="left", fontsize=8)
    fig.savefig(output_dir / "spectrum_batch_metrics.png", dpi=180)
    plt.close(fig)

    row_center = np.nanmedian(batch_combined, axis=1, keepdims=True)
    row_scale = np.nanpercentile(np.abs(batch_combined - row_center), 99, axis=1, keepdims=True)
    normalized = (batch_combined - row_center) / np.maximum(row_scale, 1.0)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    image = axes[0].imshow(
        normalized,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[0].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Per-batch full-spectrum heatmap, independently scaled")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Sequential batch")
    fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02)
    folded_center = np.nanmedian(folded, axis=1, keepdims=True)
    folded_scale = np.nanpercentile(np.abs(folded - folded_center), 99, axis=1, keepdims=True)
    folded_normalized = (folded - folded_center) / np.maximum(folded_scale, 1.0)
    image = axes[1].imshow(
        folded_normalized,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[1].set_title("Per-batch folded-ZLP heatmap, independently scaled")
    axes[1].set_xlabel("Physical ZLP detector column modulo period")
    axes[1].set_ylabel("Sequential batch")
    fig.colorbar(image, ax=axes[1], fraction=0.025, pad=0.02)
    fig.savefig(output_dir / "spectrum_batch_heatmaps.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input spectrum DM4 files.")
    parser.add_argument("--dark-frame", type=Path, required=True)
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio")
    parser.add_argument("--frames-axis", type=int, default=None)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames-per-file", type=int, default=None)
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--read-chunk-size", type=int, default=8)
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-period", type=int, default=192)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument("--median-window-pixels", type=int, default=31)
    parser.add_argument("--dynamic-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--dynamic-threshold-offset", type=float, default=500.0)
    parser.add_argument("--disable-valid-pixel-mask", action="store_true")
    parser.add_argument("--disable-dynamic-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tensor_frames <= 0 or args.read_chunk_size <= 0:
        raise ValueError("tensor and read chunk sizes must be positive")
    if args.max_frames_per_file is not None and args.max_frames_per_file <= 0:
        raise ValueError("max_frames_per_file must be positive")

    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dark_frame, "r") as dark_h5:
        dark_frame = read_single_image(dark_h5, args.dark_dataset, np)
        valid_mask = read_single_image(dark_h5, args.valid_mask_dataset, np)

    if dark_frame.shape != (args.height, args.width):
        raise ValueError(
            f"dark frame shape {dark_frame.shape} does not match {(args.height, args.width)}"
        )

    regions = detector_regions(args.height, args.edge_rows)
    file_count = len(args.inputs)
    per_file_sums = np.zeros((file_count, 2, args.width), dtype=np.float64)
    per_file_valid_counts = np.zeros_like(per_file_sums)
    per_file_pipeline_counts = np.zeros_like(per_file_sums)
    per_file_static_sums = np.zeros_like(per_file_sums)
    per_file_static_valid_counts = np.zeros_like(per_file_sums)
    per_file_frames = np.zeros(file_count, dtype=np.int64)
    per_file_tensor_count = np.zeros(file_count, dtype=np.int64)
    batch_sums = []
    batch_valid_counts = []
    batch_static_sums = []
    batch_static_valid_counts = []
    batch_source_indices = []
    batch_start_frames = []
    batch_end_frames = []
    batch_mask_pixels = []
    source_labels = [path.stem for path in args.inputs]

    for file_index, input_path in enumerate(args.inputs):
        data, info = load_dm4(input_path, args.reader)
        stack = normalize_to_frame_stack(
            data, args.frames_axis, args.height, args.width
        )
        end_frame = stack.shape[0]
        if args.max_frames_per_file is not None:
            end_frame = min(end_frame, args.start_frame + args.max_frames_per_file)
        if args.start_frame < 0 or args.start_frame >= end_frame:
            raise ValueError(f"invalid selected range for {input_path}")
        print(
            f"{input_path}: reader={info['reader']} stack_shape={tuple(stack.shape)} "
            f"processing frames {args.start_frame}..{end_frame - 1}",
            flush=True,
        )

        for tensor_start in range(args.start_frame, end_frame, args.tensor_frames):
            tensor_end = min(end_frame, tensor_start + args.tensor_frames)
            frame_count = tensor_end - tensor_start
            raw_sum = np.zeros((args.height, args.width), dtype=np.float64)
            for read_start in range(tensor_start, tensor_end, args.read_chunk_size):
                read_end = min(tensor_end, read_start + args.read_chunk_size)
                block = materialize(stack[read_start:read_end], np)
                raw_sum += block.sum(axis=0, dtype=np.float64)

            corrected_mean = (raw_sum / frame_count).astype(np.float32)
            corrected_mean -= dark_frame
            corrected_mean = subtract_imagej_blr(
                corrected_mean[None, :, :],
                np,
                args.blr_rows,
                args.zlp_width,
                args.zlp_group_columns,
                args.core_group_columns,
            )
            static_zero_mask = (
                valid_mask == 0.0
                if not args.disable_valid_pixel_mask
                else np.zeros_like(valid_mask, dtype=bool)
            )
            static_corrected_mean = corrected_mean[0].copy()
            static_corrected_mean[static_zero_mask] = 0.0
            zero_mask = apply_dynamic_and_valid_mask(
                corrected_mean,
                valid_mask,
                not args.disable_valid_pixel_mask,
                not args.disable_dynamic_mask,
                args.median_window_pixels,
                args.dynamic_threshold_ratio,
                args.dynamic_threshold_offset,
                args.edge_rows,
                True,
                np,
            )
            corrected_mean = corrected_mean[0]

            for half, rows in enumerate(regions):
                per_file_static_sums[file_index, half] += (
                    static_corrected_mean[rows].sum(axis=0, dtype=np.float64)
                    * frame_count
                )
                per_file_static_valid_counts[file_index, half] += (
                    frame_count * np.count_nonzero(~static_zero_mask[rows], axis=0)
                )
                per_file_sums[file_index, half] += (
                    corrected_mean[rows].sum(axis=0, dtype=np.float64) * frame_count
                )
                per_file_pipeline_counts[file_index, half] += (
                    frame_count * (rows.stop - rows.start)
                )
                per_file_valid_counts[file_index, half] += (
                    frame_count * np.count_nonzero(~zero_mask[rows], axis=0)
                )

            batch_sums.append(np.stack([
                corrected_mean[rows].sum(axis=0, dtype=np.float64) * frame_count
                for rows in regions
            ]))
            batch_valid_counts.append(np.stack([
                frame_count * np.count_nonzero(~zero_mask[rows], axis=0)
                for rows in regions
            ]))
            batch_static_sums.append(np.stack([
                static_corrected_mean[rows].sum(axis=0, dtype=np.float64) * frame_count
                for rows in regions
            ]))
            batch_static_valid_counts.append(np.stack([
                frame_count * np.count_nonzero(~static_zero_mask[rows], axis=0)
                for rows in regions
            ]))
            batch_source_indices.append(file_index)
            batch_start_frames.append(tensor_start)
            batch_end_frames.append(tensor_end - 1)
            batch_mask_pixels.append(int(zero_mask.sum()))

            per_file_frames[file_index] += frame_count
            per_file_tensor_count[file_index] += 1
            print(
                f"  batch {tensor_start}..{tensor_end - 1}: "
                f"masked {int(zero_mask.sum())} detector pixels",
                flush=True,
            )

        del stack, data

    valid_sums = per_file_sums.sum(axis=0)
    valid_counts = per_file_valid_counts.sum(axis=0)
    pipeline_counts = per_file_pipeline_counts.sum(axis=0)
    static_valid_sums = per_file_static_sums.sum(axis=0)
    static_valid_counts = per_file_static_valid_counts.sum(axis=0)
    valid_mean = safe_divide(valid_sums, valid_counts, np)
    pipeline_mean = safe_divide(valid_sums, pipeline_counts, np)
    static_valid_mean = safe_divide(static_valid_sums, static_valid_counts, np)
    mask_fraction = 1.0 - safe_divide(valid_counts, pipeline_counts, np)
    folded_valid, folded_sums, folded_counts = fold_zlp(
        valid_sums, valid_counts, args.zlp_width, args.zlp_period, np
    )
    batch_sums = np.stack(batch_sums)
    batch_valid_counts = np.stack(batch_valid_counts)
    batch_static_sums = np.stack(batch_static_sums)
    batch_static_valid_counts = np.stack(batch_static_valid_counts)
    batch_source_indices = np.asarray(batch_source_indices, dtype=np.int32)
    batch_start_frames = np.asarray(batch_start_frames, dtype=np.int64)
    batch_end_frames = np.asarray(batch_end_frames, dtype=np.int64)
    batch_mask_pixels = np.asarray(batch_mask_pixels, dtype=np.int64)

    output_h5 = args.output_dir / "final_spectrum.h5"
    with h5py.File(output_h5, "w") as output:
        output.create_dataset("full_columns_valid_mean", data=valid_mean)
        output.create_dataset("full_columns_pipeline_zeroed_mean", data=pipeline_mean)
        output.create_dataset("full_columns_sum", data=valid_sums)
        output.create_dataset("full_columns_valid_count", data=valid_counts)
        output.create_dataset("full_columns_pipeline_count", data=pipeline_counts)
        output.create_dataset("full_columns_static_mask_only_mean", data=static_valid_mean)
        output.create_dataset("full_columns_static_mask_only_sum", data=static_valid_sums)
        output.create_dataset(
            "full_columns_static_mask_only_valid_count", data=static_valid_counts
        )
        output.create_dataset("zlp_folded_valid_mean", data=folded_valid)
        output.create_dataset("zlp_folded_sum", data=folded_sums)
        output.create_dataset("zlp_folded_valid_count", data=folded_counts)
        output.create_dataset("per_file_full_columns_sum", data=per_file_sums)
        output.create_dataset("per_file_full_columns_valid_count", data=per_file_valid_counts)
        output.create_dataset("per_batch_full_columns_sum", data=batch_sums)
        output.create_dataset("per_batch_full_columns_valid_count", data=batch_valid_counts)
        output.create_dataset("per_batch_static_mask_only_sum", data=batch_static_sums)
        output.create_dataset(
            "per_batch_static_mask_only_valid_count", data=batch_static_valid_counts
        )
        output.create_dataset("per_batch_source_index", data=batch_source_indices)
        output.create_dataset("per_batch_start_frame", data=batch_start_frames)
        output.create_dataset("per_batch_end_frame", data=batch_end_frames)
        output.create_dataset("per_batch_masked_pixel_count", data=batch_mask_pixels)
        output.attrs["source_files"] = np.array(
            [str(path) for path in args.inputs],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        output.attrs["dark_frame"] = str(args.dark_frame)
        output.attrs["processing_order"] = json.dumps([
            "batch_mean",
            "dark_subtraction",
            "ImageJ_BLR",
            "valid_and_dynamic_half_column_mask",
            "column_accumulation",
        ])

    write_full_csv(
        args.output_dir / "final_spectrum_full_columns.csv",
        valid_mean,
        pipeline_mean,
        static_valid_mean,
        valid_counts,
        pipeline_counts,
        np,
    )
    save_plots(
        args.output_dir,
        valid_mean,
        pipeline_mean,
        valid_sums,
        valid_counts,
        per_file_sums,
        per_file_valid_counts,
        static_valid_sums,
        static_valid_counts,
        source_labels,
        args.zlp_width,
        args.zlp_period,
        mask_fraction,
        plt,
        np,
    )
    save_batch_plots(
        args.output_dir,
        batch_sums,
        batch_valid_counts,
        batch_source_indices,
        source_labels,
        args.zlp_width,
        args.zlp_period,
        batch_mask_pixels,
        plt,
        np,
    )

    summary = {
        "source_files": [str(path) for path in args.inputs],
        "frames_per_file": per_file_frames.tolist(),
        "tensor_batches_per_file": per_file_tensor_count.tolist(),
        "total_frames": int(per_file_frames.sum()),
        "frame_shape": [args.height, args.width],
        "dark_frame": str(args.dark_frame),
        "apply_valid_pixel_mask": not args.disable_valid_pixel_mask,
        "apply_dynamic_mask": not args.disable_dynamic_mask,
        "dynamic_mask_median_window_pixels": args.median_window_pixels,
        "dynamic_mask_threshold_offset": args.dynamic_threshold_offset,
        "zlp_width": args.zlp_width,
        "zlp_period": args.zlp_period,
        "output_hdf5": str(output_h5),
        "plots": [
            "final_spectrum_full_columns.png",
            "final_spectrum_zlp_folded.png",
            "final_spectrum_coreloss.png",
            "spectrum_by_source_file.png",
            "spectrum_mask_fraction.png",
            "dynamic_mask_spectrum_comparison.png",
            "spectrum_batch_metrics.png",
            "spectrum_batch_heatmaps.png",
        ],
    }
    (args.output_dir / "spectrum_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
