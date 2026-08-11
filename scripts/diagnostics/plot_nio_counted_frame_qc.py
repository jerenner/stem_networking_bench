#!/usr/bin/env python3
"""Plot representative corrected frames and their counted-event coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stem_analysis import process_tensor_block
from stem_analysis.dm4 import load_dm4, normalize_to_frame_stack


RAW_ZLP_WIDTH = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counting-dir", type=Path, required=True)
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def compute_array(array, np):
    return np.asarray(array.compute() if hasattr(array, "compute") else array)


def neighboring_maximum(frame, np):
    maximum = np.zeros_like(frame)
    maximum[1:, :] = np.maximum(maximum[1:, :], frame[:-1, :])
    maximum[:-1, :] = np.maximum(maximum[:-1, :], frame[1:, :])
    maximum[:, 1:] = np.maximum(maximum[:, 1:], frame[:, :-1])
    maximum[:, :-1] = np.maximum(maximum[:, :-1], frame[:, 1:])
    maximum[1:, 1:] = np.maximum(maximum[1:, 1:], frame[:-1, :-1])
    maximum[1:, :-1] = np.maximum(maximum[1:, :-1], frame[:-1, 1:])
    maximum[:-1, 1:] = np.maximum(maximum[:-1, 1:], frame[1:, :-1])
    maximum[:-1, :-1] = np.maximum(maximum[:-1, :-1], frame[1:, 1:])
    return maximum


def selected_frames(per_frame_counts, tail_core, np):
    totals = per_frame_counts.sum(axis=1)
    tails = per_frame_counts[:, tail_core:].sum(axis=1)
    peaks = per_frame_counts.max(axis=1)
    center = (
        np.abs(totals - np.median(totals)) / max(float(np.std(totals)), 1.0)
        + np.abs(tails - np.median(tails)) / max(float(np.std(tails)), 1.0)
    )
    candidates = [
        ("typical", int(np.argmin(center))),
        ("maximum_tail", int(np.argmax(tails))),
        ("busiest_channel", int(np.argmax(peaks))),
    ]
    unique = []
    seen = set()
    for label, index in candidates:
        if index not in seen:
            unique.append((label, index))
            seen.add(index)
    return unique, totals, tails, peaks


def densest_crop(event_mask, size, np, column_start=0):
    height, width = event_mask.shape
    search = event_mask[:, column_start:]
    width = search.shape[1]
    rows = height // size
    columns = width // size
    trimmed = search[:rows * size, :columns * size]
    block_counts = trimmed.reshape(rows, size, columns, size).sum(axis=(1, 3))
    block_row, block_column = np.unravel_index(np.argmax(block_counts), block_counts.shape)
    row_start = int(block_row * size)
    crop_column = int(column_start + block_column * size)
    return row_start, crop_column, int(block_counts[block_row, block_column])


def plot_frame(output_path,
               label,
               global_index,
               source_index,
               source_frame,
               corrected,
               event_mask,
               stored_counts,
               tail_core,
               threshold,
               np,
               plt,
               colors):
    frame_counts = event_mask.sum(axis=0)
    total = int(frame_counts.sum())
    tail = int(frame_counts[tail_core:].sum())
    if not np.array_equal(frame_counts.astype(stored_counts.dtype), stored_counts):
        raise RuntimeError(f"reconstructed event coordinates differ for global frame {global_index}")

    lower, upper = np.percentile(corrected, (0.5, 99.9))
    magnitude = max(abs(float(lower)), abs(float(upper)), 1.0)
    norm = colors.SymLogNorm(linthresh=500.0, vmin=-magnitude, vmax=magnitude)
    binned = event_mask.reshape(224, 4, 768, 4).sum(axis=(1, 3))
    crop_row, crop_column, crop_events = densest_crop(event_mask, 128, np)
    crop = corrected[crop_row:crop_row + 128, crop_column:crop_column + 128]
    crop_mask = event_mask[crop_row:crop_row + 128, crop_column:crop_column + 128]
    event_rows, event_columns = np.nonzero(crop_mask)

    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), constrained_layout=True)
    image = axes[0, 0].imshow(
        corrected, aspect="auto", interpolation="nearest", cmap="coolwarm", norm=norm,
        extent=(RAW_ZLP_WIDTH, 3840, corrected.shape[0], 0),
    )
    axes[0, 0].set(
        xlabel="raw detector column", ylabel="imaging-row index",
        title=f"Corrected CoreLoss frame ({label})",
    )
    fig.colorbar(image, ax=axes[0, 0], label="corrected detector value")

    density = axes[0, 1].imshow(
        binned, aspect="auto", interpolation="nearest", cmap="magma",
        extent=(RAW_ZLP_WIDTH, 3840, corrected.shape[0], 0), vmin=0,
    )
    axes[0, 1].set(
        xlabel="raw detector column", ylabel="imaging-row index",
        title=f"Counted events in 4x4 display bins ({total:,} events)",
    )
    fig.colorbar(density, ax=axes[0, 1], label="events per 4x4 detector block")

    raw_columns = np.arange(RAW_ZLP_WIDTH, 3840)
    axes[1, 0].plot(raw_columns, frame_counts, linewidth=0.8, color="#0F766E")
    axes[1, 0].axvline(RAW_ZLP_WIDTH + tail_core, color="#B45309", linestyle="--",
                       label="provisional 350 eV tail")
    axes[1, 0].set(
        xlabel="raw detector column", ylabel="events in this frame",
        title=f"Frame spectrum: CoreLoss={total:,}, tail={tail:,}",
    )
    axes[1, 0].legend()

    crop_image = axes[1, 1].imshow(
        crop, interpolation="nearest", cmap="gray", norm=norm,
        extent=(RAW_ZLP_WIDTH + crop_column, RAW_ZLP_WIDTH + crop_column + 128,
                crop_row + 128, crop_row),
    )
    axes[1, 1].scatter(
        RAW_ZLP_WIDTH + crop_column + event_columns + 0.5,
        crop_row + event_rows + 0.5,
        s=18, facecolors="none", edgecolors="#EF4444", linewidths=0.8,
    )
    axes[1, 1].set(
        xlabel="raw detector column", ylabel="imaging-row index",
        title=f"Densest 128x128 crop: {crop_events} local maxima",
    )
    fig.colorbar(crop_image, ax=axes[1, 1], label="corrected detector value")
    fig.suptitle(
        f"Global frame {global_index}; source {source_index}, frame {source_frame}; "
        f"threshold={threshold:.1f} corrected units",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    event_values = corrected[event_mask]
    linear_maximum = max(
        threshold * 2.0,
        float(np.percentile(event_values, 99.0)) if event_values.size else threshold * 2.0,
    )
    linear_norm = colors.TwoSlopeNorm(
        vmin=-threshold, vcenter=0.0, vmax=linear_maximum
    )
    block_events = event_mask.reshape(224, 4, 768, 4).any(axis=(1, 3))
    block_amplitudes = np.where(event_mask, corrected, -np.inf).reshape(
        224, 4, 768, 4
    ).max(axis=(1, 3))
    block_amplitudes = np.ma.masked_where(~block_events, block_amplitudes)
    counted_cmap = plt.get_cmap("inferno").copy()
    counted_cmap.set_bad("black")

    tail_row, tail_column, tail_crop_events = densest_crop(
        event_mask, 256, np, column_start=tail_core
    )
    tail_crop = corrected[tail_row:tail_row + 256, tail_column:tail_column + 256]
    tail_crop_mask = event_mask[
        tail_row:tail_row + 256, tail_column:tail_column + 256
    ]
    tail_rows, tail_columns = np.nonzero(tail_crop_mask)

    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), constrained_layout=True)
    image = axes[0, 0].imshow(
        corrected, aspect="auto", interpolation="nearest", cmap="coolwarm",
        norm=linear_norm, extent=(RAW_ZLP_WIDTH, 3840, corrected.shape[0], 0),
    )
    axes[0, 0].set(
        xlabel="raw detector column", ylabel="imaging-row index",
        title="Corrected frame, linear threshold-centered scale",
    )
    fig.colorbar(image, ax=axes[0, 0], label="corrected detector value")

    counted = axes[0, 1].imshow(
        block_amplitudes, aspect="auto", interpolation="nearest", cmap=counted_cmap,
        vmin=threshold, vmax=linear_maximum,
        extent=(RAW_ZLP_WIDTH, 3840, corrected.shape[0], 0),
    )
    axes[0, 1].set(
        xlabel="raw detector column", ylabel="imaging-row index",
        title="Counted-only amplitudes in 4x4 display bins",
    )
    fig.colorbar(counted, ax=axes[0, 1], label="counted-pixel amplitude")

    tail_image = axes[1, 0].imshow(
        tail_crop, interpolation="nearest", cmap="gray", norm=linear_norm,
        extent=(RAW_ZLP_WIDTH + tail_column, RAW_ZLP_WIDTH + tail_column + 256,
                tail_row + 256, tail_row),
    )
    axes[1, 0].scatter(
        RAW_ZLP_WIDTH + tail_column + tail_columns + 0.5,
        tail_row + tail_rows + 0.5,
        s=28, facecolors="none", edgecolors="#FACC15", linewidths=1.0,
    )
    axes[1, 0].set(
        xlabel="raw detector column", ylabel="imaging-row index",
        title=f"Densest 256x256 tail crop: {tail_crop_events} counted events",
    )
    fig.colorbar(tail_image, ax=axes[1, 0], label="corrected detector value")

    histogram_stop = max(linear_maximum * 1.2, threshold * 2.5)
    axes[1, 1].hist(
        corrected.ravel(), bins=300, range=(-threshold, histogram_stop),
        histtype="step", color="#0F766E", linewidth=1.0,
    )
    axes[1, 1].axvline(
        threshold, color="#DC2626", linestyle="--",
        label=f"counting threshold {threshold:.0f}",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        xlabel="corrected detector value", ylabel="pixels",
        title=f"Frame-value distribution ({total:,} local maxima retained)",
    )
    axes[1, 1].legend()
    fig.suptitle(
        f"Linear counted-frame QC: global {global_index}; source {source_index}, "
        f"frame {source_frame}; tail={tail:,}",
        fontsize=14,
    )
    linear_path = output_path.with_name(f"{output_path.stem}_linear.png")
    fig.savefig(linear_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from scripts.studies.run_nio_counting_study import (
        load_dark_products,
        processor_config,
    )

    counting_dir = args.counting_dir.resolve()
    output_dir = (args.output_dir or counting_dir / "counted_frame_qc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((counting_dir / "counting_summary.json").read_text())
    with h5py.File(counting_dir / "counted_spectrum.h5", "r") as source:
        per_frame_counts = source["per_frame_core_counts"][:]
        source_indices = source["per_frame_source_index"][:]
        source_frames = source["per_frame_source_frame"][:]

    tail_core = summary["energy_calibration"]["tail_raw_channel"] - RAW_ZLP_WIDTH
    selections, totals, tails, peaks = selected_frames(per_frame_counts, tail_core, np)
    dark_frame, valid_mask = load_dark_products(Path(summary["dark_frame"]), np)
    config = processor_config(
        summary["blr_mode"], summary["processing"]["dynamic_mask_median_window_pixels"]
    )
    threshold_quantized = summary["selected_threshold"]["threshold_quantized"]
    xray_quantized = summary["selected_threshold"]["xray_threshold_quantized"]
    scale = summary["quantization"]["scale"]
    offset = summary["quantization"]["offset"]
    threshold_corrected = (threshold_quantized - offset) / scale

    rows = []
    loaded_batches = {}
    for label, global_index in selections:
        source_index = int(source_indices[global_index])
        source_frame = int(source_frames[global_index])
        batch_start = source_frame // 128 * 128
        key = (source_index, batch_start)
        if key not in loaded_batches:
            path = Path(summary["spectrum_files"][source_index])
            data, _ = load_dm4(path, args.reader)
            stack = normalize_to_frame_stack(data, height=960, width=3840)
            raw = compute_array(stack[batch_start:batch_start + 128], np)
            corrected, _ = process_tensor_block(raw, dark_frame, valid_mask, config, np)
            imaging = np.concatenate(
                (corrected[:, 32:480, :], corrected[:, 480:928, :]), axis=1
            )
            loaded_batches[key] = np.ascontiguousarray(imaging[:, :, RAW_ZLP_WIDTH:])
            del data, stack, raw, corrected, imaging

        corrected_frame = loaded_batches[key][source_frame - batch_start]
        quantized = np.rint(np.clip(corrected_frame * scale + offset, 0, 65535)).astype(
            np.uint16
        )
        thresholded = quantized.copy()
        thresholded[
            (thresholded <= threshold_quantized) | (thresholded >= xray_quantized)
        ] = 0
        event_mask = thresholded > neighboring_maximum(thresholded, np)
        stem = f"{label}_global_{global_index:04d}_source_{source_index}_frame_{source_frame:04d}"
        plot_frame(
            output_dir / f"{stem}.png", label, global_index, source_index, source_frame,
            corrected_frame, event_mask, per_frame_counts[global_index], tail_core,
            threshold_corrected, np, plt, colors,
        )
        Image.fromarray(event_mask.astype(np.uint8) * 255).save(
            output_dir / f"{stem}_event_mask.tiff", compression="tiff_deflate"
        )
        rows.append({
            "selection": label,
            "global_frame": global_index,
            "source_index": source_index,
            "source_frame": source_frame,
            "coreloss_events": int(totals[global_index]),
            "tail_events": int(tails[global_index]),
            "maximum_events_in_one_channel": int(peaks[global_index]),
        })

    with (output_dir / "selected_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} counted-frame QC sets to {output_dir}")


if __name__ == "__main__":
    main()
