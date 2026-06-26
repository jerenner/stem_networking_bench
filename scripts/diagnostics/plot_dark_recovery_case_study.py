#!/usr/bin/env python3
"""Study temporal dark-frame recovery/drift across frame chunks.

This diagnostic treats a dark-frame stack as a time series. It averages the stack
into fixed-size temporal chunks, subtracts one chunk mean from the others, applies
the same ImageJ-style BLR baseline correction used by the offline/Holoscan
processing path, then plots residual maps and ZLP folded-column trends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_nio_processing_analysis import (
    configure_matplotlib_cache,
    detector_regions,
    robust_limits,
    subtract_imagej_blr,
    symmetric_limit,
)


def compute_chunk_statistics(dataset, np, start_frame: int, end_frame: int, read_chunk_size: int):
    accumulator = None
    sumsq = None
    frame_count = end_frame - start_frame
    if frame_count <= 0:
        raise ValueError("chunk has no frames")

    for start in range(start_frame, end_frame, read_chunk_size):
        end = min(end_frame, start + read_chunk_size)
        block = dataset[start:end].astype(np.float64, copy=False)
        if accumulator is None:
            accumulator = np.zeros(block.shape[1:], dtype=np.float64)
            sumsq = np.zeros(block.shape[1:], dtype=np.float64)
        accumulator += block.sum(axis=0)
        sumsq += np.square(block).sum(axis=0)

    mean = accumulator / frame_count
    variance = (sumsq / frame_count) - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def chunk_label(start: int, end: int) -> str:
    return f"{start}..{end - 1}"


def zlp_mod_mask(width: int,
                 zlp_width: int,
                 zlp_period: int,
                 mod_start: int,
                 mod_end: int,
                 np):
    if zlp_width > width:
        raise ValueError(f"zlp_width={zlp_width} exceeds image width={width}")
    if zlp_width % zlp_period != 0:
        raise ValueError("zlp_width must be divisible by zlp_period")
    if not (0 <= mod_start <= mod_end < zlp_period):
        raise ValueError("ZLP mod range must lie within one ZLP period")

    columns = np.arange(width)
    return (columns < zlp_width) & (columns % zlp_period >= mod_start) & (
        columns % zlp_period <= mod_end
    )


def masked_mean(image, valid_mask, axis, np):
    return np.nanmean(np.where(valid_mask, image, np.nan), axis=axis)


def column_profiles(image, top_rows: slice, bottom_rows: slice, valid_pixels, np):
    return (
        masked_mean(image[top_rows], valid_pixels[top_rows], axis=0, np=np),
        masked_mean(image[bottom_rows], valid_pixels[bottom_rows], axis=0, np=np),
    )


def folded_zlp_profile(column_profile, zlp_width: int, zlp_period: int, np):
    return column_profile[:zlp_width].reshape(zlp_width // zlp_period, zlp_period).mean(axis=0)


def save_raw_chunk_profiles(chunk_means,
                            chunk_ranges,
                            top_rows: slice,
                            bottom_rows: slice,
                            valid_pixels,
                            zlp_width: int,
                            output_path: Path,
                            plt,
                            np):
    height = chunk_means.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    for index, chunk_mean in enumerate(chunk_means):
        label = f"chunk {index}: {chunk_label(*chunk_ranges[index])}"
        axes[0].plot(
            masked_mean(chunk_mean, valid_pixels, axis=1, np=np),
            label=label,
            linewidth=0.9,
        )
        top_profile, bottom_profile = column_profiles(
            chunk_mean, top_rows, bottom_rows, valid_pixels, np
        )
        axes[1].plot(top_profile, label=label, linewidth=0.8)
        axes[2].plot(bottom_profile, label=label, linewidth=0.8)

    axes[0].axvline(height // 2, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Raw dark chunk row means")
    axes[0].set_xlabel("Row")
    axes[0].set_ylabel("Mean raw value")

    for axis, title in (
        (axes[1], "Raw dark chunk column means, top imaging half"),
        (axes[2], "Raw dark chunk column means, bottom imaging half"),
    ):
        axis.axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Column")
        axis.set_ylabel("Mean raw value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_residual_maps(residuals,
                       chunk_ranges,
                       reference_chunk: int,
                       valid_pixels,
                       output_path: Path,
                       plt,
                       TwoSlopeNorm,
                       np):
    cols = min(2, residuals.shape[0])
    rows = int(np.ceil(residuals.shape[0] / cols))
    plotted_residuals = np.where(valid_pixels[None, :, :], residuals, np.nan)
    limit = symmetric_limit(plotted_residuals, np, 99.5)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for index, (axis, residual) in enumerate(zip(axes, plotted_residuals)):
        image = axis.imshow(
            residual,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
            aspect="auto",
        )
        title = f"chunk {index}: {chunk_label(*chunk_ranges[index])}"
        if index == reference_chunk:
            title += " (reference)"
        axis.set_title(title)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)

    for axis in axes[residuals.shape[0]:]:
        axis.axis("off")
    fig.suptitle(
        f"BLR-corrected chunk residual maps relative to chunk {reference_chunk}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_residual_profiles(residuals,
                           chunk_ranges,
                           reference_chunk: int,
                           top_rows: slice,
                           bottom_rows: slice,
                           valid_pixels,
                           zlp_width: int,
                           output_path: Path,
                           plt,
                           np):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    for index, residual in enumerate(residuals):
        label = f"chunk {index}: {chunk_label(*chunk_ranges[index])}"
        if index == reference_chunk:
            label += " (reference)"
        top_profile, bottom_profile = column_profiles(
            residual, top_rows, bottom_rows, valid_pixels, np
        )
        axes[0].plot(top_profile, label=label, linewidth=0.8)
        axes[1].plot(bottom_profile, label=label, linewidth=0.8)

    for axis, title in (
        (axes[0], "BLR-corrected residual column mean, top imaging half"),
        (axes[1], "BLR-corrected residual column mean, bottom imaging half"),
    ):
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Column")
        axis.set_ylabel("Mean residual")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_folded_zlp_profiles(residuals,
                             chunk_ranges,
                             reference_chunk: int,
                             top_rows: slice,
                             bottom_rows: slice,
                             valid_pixels,
                             zlp_width: int,
                             zlp_period: int,
                             mod_start: int,
                             mod_end: int,
                             output_path: Path,
                             plt,
                             np):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    for index, residual in enumerate(residuals):
        label = f"chunk {index}: {chunk_label(*chunk_ranges[index])}"
        if index == reference_chunk:
            label += " (reference)"
        top_profile, bottom_profile = column_profiles(
            residual, top_rows, bottom_rows, valid_pixels, np
        )
        axes[0].plot(
            folded_zlp_profile(top_profile, zlp_width, zlp_period, np),
            label=label,
            linewidth=0.9,
        )
        axes[1].plot(
            folded_zlp_profile(bottom_profile, zlp_width, zlp_period, np),
            label=label,
            linewidth=0.9,
        )

    for axis, title in (
        (axes[0], "Folded ZLP residual profile, top imaging half"),
        (axes[1], "Folded ZLP residual profile, bottom imaging half"),
    ):
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvspan(mod_start, mod_end, color="tab:red", alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel(f"Physical ZLP column modulo {zlp_period}")
        axis.set_ylabel("Mean residual")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_chunk_stddev_maps(chunk_stddevs,
                           chunk_ranges,
                           blinker_std_threshold: float,
                           output_path: Path,
                           plt,
                           np):
    cols = min(2, chunk_stddevs.shape[0])
    rows = int(np.ceil(chunk_stddevs.shape[0] / cols))
    _, vmax = robust_limits(chunk_stddevs, np, 0.0, 99.5)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for index, (axis, stddev) in enumerate(zip(axes, chunk_stddevs)):
        image = axis.imshow(stddev, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        flagged = int(np.count_nonzero(stddev > blinker_std_threshold))
        axis.set_title(
            f"chunk {index}: {chunk_label(*chunk_ranges[index])}; "
            f"{flagged} pixels > {blinker_std_threshold:g}"
        )
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)

    for axis in axes[chunk_stddevs.shape[0]:]:
        axis.axis("off")
    fig.suptitle("Per-pixel temporal standard deviation by chunk")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_folded_zlp_heatmaps(folded_top,
                             folded_bottom,
                             chunk_ranges,
                             reference_chunk: int,
                             mod_start: int,
                             mod_end: int,
                             output_path: Path,
                             plt,
                             TwoSlopeNorm,
                             np):
    limit = symmetric_limit(np.concatenate([folded_top.ravel(), folded_bottom.ravel()]), np, 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    y_labels = [f"{index}: {chunk_label(*frame_range)}" for index, frame_range in enumerate(chunk_ranges)]
    for axis, values, title in (
        (axes[0], folded_top, "Top imaging half"),
        (axes[1], folded_bottom, "Bottom imaging half"),
    ):
        image = axis.imshow(
            values,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
            aspect="auto",
        )
        axis.axvspan(mod_start - 0.5, mod_end + 0.5, color="tab:red", alpha=0.15)
        axis.axhline(reference_chunk, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Physical ZLP column modulo period")
        axis.set_yticks(range(len(y_labels)))
        axis.set_yticklabels(y_labels)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    fig.suptitle("Folded ZLP residual heatmaps relative to reference chunk")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_suspicious_mod_trends(folded_top,
                               folded_bottom,
                               chunk_ranges,
                               reference_chunk: int,
                               mod_start: int,
                               mod_end: int,
                               output_path: Path,
                               plt,
                               np):
    frame_midpoints = np.array([(start + end - 1) / 2 for start, end in chunk_ranges])
    mods = np.arange(mod_start, mod_end + 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for axis, values, title in (
        (axes[0], folded_top, "Top imaging half"),
        (axes[1], folded_bottom, "Bottom imaging half"),
    ):
        band = values[:, mod_start:mod_end + 1]
        for offset, mod in enumerate(mods):
            axis.plot(frame_midpoints, band[:, offset], color="tab:blue", alpha=0.25, linewidth=0.9)
        axis.plot(
            frame_midpoints,
            band.mean(axis=1),
            color="tab:red",
            marker="o",
            linewidth=2.0,
            label=f"mean mods {mod_start}..{mod_end}",
        )
        axis.fill_between(
            frame_midpoints,
            band.min(axis=1),
            band.max(axis=1),
            color="tab:red",
            alpha=0.12,
            label="mod min..max",
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(frame_midpoints[reference_chunk], color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Frame midpoint")
        axis.set_ylabel("Folded ZLP residual")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Suspicious ZLP physical-column residual trends")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pairwise_heatmaps(pairwise,
                           chunk_ranges,
                           output_path: Path,
                           plt,
                           np):
    labels = [f"{index}: {chunk_label(*frame_range)}" for index, frame_range in enumerate(chunk_ranges)]
    fields = [
        ("imaging_rms", "Imaging RMS"),
        ("zlp_band_rms", "ZLP suspicious-band RMS"),
        ("zlp_band_mean", "ZLP suspicious-band signed mean"),
        ("core_rms", "CoreLoss RMS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (field, title) in zip(axes.ravel(), fields):
        values = pairwise[field]
        if field.endswith("_mean"):
            limit = symmetric_limit(values, np, 100.0)
            image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        else:
            _, vmax = robust_limits(values, np, 0.0, 100.0)
            image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Reference dark chunk")
        axis.set_ylabel("Target frame chunk")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        for target in range(values.shape[0]):
            for reference in range(values.shape[1]):
                axis.text(
                    reference,
                    target,
                    f"{values[target, reference]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[target, reference] > values.max() * 0.5 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    fig.suptitle("Pairwise chunk mismatch metrics after BLR correction")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input HDF5 file containing dark frames.")
    parser.add_argument("--dataset", default="/frames", help="Input frame-stack dataset path.")
    parser.add_argument("--output-dir", type=Path, default=Path("dark_recovery_case_study"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--chunk-frames", type=int, default=128)
    parser.add_argument("--read-chunk-size", type=int, default=16)
    parser.add_argument("--reference-chunk", type=int, default=1)
    parser.add_argument(
        "--blinker-std-threshold",
        type=float,
        default=500.0,
        help="Exclude pixels exceeding this temporal stddev in any chunk; <=0 disables.",
    )
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-period", type=int, default=192)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument("--suspicious-zlp-mod-start", type=int, default=48)
    parser.add_argument("--suspicious-zlp-mod-end", type=int, default=55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.read_chunk_size <= 0:
        raise ValueError("--read-chunk-size must be positive")
    if args.frames % args.chunk_frames != 0:
        raise ValueError("--frames must be divisible by --chunk-frames")

    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5_file:
        dataset = h5_file[args.dataset]
        if dataset.ndim != 3:
            raise ValueError(f"expected [frames, rows, cols], got {dataset.shape}")
        if args.start_frame < 0 or args.start_frame + args.frames > dataset.shape[0]:
            raise ValueError("selected frame range is outside the input dataset")

        _, height, width = dataset.shape
        top_rows, bottom_rows = detector_regions(height, args.edge_rows)
        chunk_count = args.frames // args.chunk_frames
        if not (0 <= args.reference_chunk < chunk_count):
            raise ValueError(f"--reference-chunk must be in [0, {chunk_count - 1}]")

        chunk_means = []
        chunk_stddevs = []
        chunk_ranges = []
        for chunk_index in range(chunk_count):
            chunk_start = args.start_frame + chunk_index * args.chunk_frames
            chunk_end = chunk_start + args.chunk_frames
            mean, stddev = compute_chunk_statistics(
                dataset, np, chunk_start, chunk_end, args.read_chunk_size
            )
            chunk_means.append(mean)
            chunk_stddevs.append(stddev)
            chunk_ranges.append((chunk_start, chunk_end))

    chunk_means = np.stack(chunk_means)
    chunk_stddevs = np.stack(chunk_stddevs)
    reference_mean = chunk_means[args.reference_chunk]
    if args.blinker_std_threshold > 0:
        chunk_blinker_masks = chunk_stddevs > args.blinker_std_threshold
        valid_pixels = ~np.any(chunk_blinker_masks, axis=0)
    else:
        chunk_blinker_masks = np.zeros_like(chunk_stddevs, dtype=bool)
        valid_pixels = np.ones(chunk_stddevs.shape[1:], dtype=bool)

    residuals = []
    for chunk_mean in chunk_means:
        delta = (chunk_mean - reference_mean).astype(np.float32)
        residuals.append(
            subtract_imagej_blr(
                delta[None, :, :],
                np,
                args.blr_rows,
                args.zlp_width,
                args.zlp_group_columns,
                args.core_group_columns,
            )[0]
        )
    residuals = np.stack(residuals)

    top_folded = []
    bottom_folded = []
    for residual in residuals:
        top_profile, bottom_profile = column_profiles(
            residual, top_rows, bottom_rows, valid_pixels, np
        )
        top_folded.append(folded_zlp_profile(top_profile, args.zlp_width, args.zlp_period, np))
        bottom_folded.append(
            folded_zlp_profile(bottom_profile, args.zlp_width, args.zlp_period, np)
        )
    top_folded = np.stack(top_folded)
    bottom_folded = np.stack(bottom_folded)

    zlp_band_columns = zlp_mod_mask(
        width,
        args.zlp_width,
        args.zlp_period,
        args.suspicious_zlp_mod_start,
        args.suspicious_zlp_mod_end,
        np,
    )
    imaging_mask = np.zeros((height, width), dtype=bool)
    imaging_mask[top_rows, :] = True
    imaging_mask[bottom_rows, :] = True
    imaging_mask &= valid_pixels
    zlp_band_mask = imaging_mask & zlp_band_columns[None, :]
    core_columns = np.arange(width) >= args.zlp_width
    core_mask = imaging_mask & core_columns[None, :]

    pairwise = {
        "imaging_rms": np.zeros((chunk_count, chunk_count), dtype=np.float32),
        "zlp_band_rms": np.zeros((chunk_count, chunk_count), dtype=np.float32),
        "zlp_band_mean": np.zeros((chunk_count, chunk_count), dtype=np.float32),
        "core_rms": np.zeros((chunk_count, chunk_count), dtype=np.float32),
    }
    for target in range(chunk_count):
        for reference in range(chunk_count):
            delta = (chunk_means[target] - chunk_means[reference]).astype(np.float32)
            corrected = subtract_imagej_blr(
                delta[None, :, :],
                np,
                args.blr_rows,
                args.zlp_width,
                args.zlp_group_columns,
                args.core_group_columns,
            )[0]
            pairwise["imaging_rms"][target, reference] = np.sqrt(
                np.mean(np.square(corrected[imaging_mask]))
            )
            pairwise["zlp_band_rms"][target, reference] = np.sqrt(
                np.mean(np.square(corrected[zlp_band_mask]))
            )
            pairwise["zlp_band_mean"][target, reference] = np.mean(corrected[zlp_band_mask])
            pairwise["core_rms"][target, reference] = np.sqrt(
                np.mean(np.square(corrected[core_mask]))
            )

    save_raw_chunk_profiles(
        chunk_means,
        chunk_ranges,
        top_rows,
        bottom_rows,
        valid_pixels,
        args.zlp_width,
        args.output_dir / "raw_chunk_mean_profiles.png",
        plt,
        np,
    )
    save_residual_maps(
        residuals,
        chunk_ranges,
        args.reference_chunk,
        valid_pixels,
        args.output_dir / f"blr_residual_maps_ref_chunk_{args.reference_chunk:03d}.png",
        plt,
        TwoSlopeNorm,
        np,
    )
    save_residual_profiles(
        residuals,
        chunk_ranges,
        args.reference_chunk,
        top_rows,
        bottom_rows,
        valid_pixels,
        args.zlp_width,
        args.output_dir / f"blr_residual_profiles_ref_chunk_{args.reference_chunk:03d}.png",
        plt,
        np,
    )
    save_folded_zlp_profiles(
        residuals,
        chunk_ranges,
        args.reference_chunk,
        top_rows,
        bottom_rows,
        valid_pixels,
        args.zlp_width,
        args.zlp_period,
        args.suspicious_zlp_mod_start,
        args.suspicious_zlp_mod_end,
        args.output_dir / f"zlp_folded_profiles_ref_chunk_{args.reference_chunk:03d}.png",
        plt,
        np,
    )
    save_chunk_stddev_maps(
        chunk_stddevs,
        chunk_ranges,
        args.blinker_std_threshold,
        args.output_dir / "chunk_temporal_stddev_maps.png",
        plt,
        np,
    )
    save_folded_zlp_heatmaps(
        top_folded,
        bottom_folded,
        chunk_ranges,
        args.reference_chunk,
        args.suspicious_zlp_mod_start,
        args.suspicious_zlp_mod_end,
        args.output_dir / f"zlp_folded_heatmaps_ref_chunk_{args.reference_chunk:03d}.png",
        plt,
        TwoSlopeNorm,
        np,
    )
    save_suspicious_mod_trends(
        top_folded,
        bottom_folded,
        chunk_ranges,
        args.reference_chunk,
        args.suspicious_zlp_mod_start,
        args.suspicious_zlp_mod_end,
        args.output_dir / f"suspicious_zlp_mod_trends_ref_chunk_{args.reference_chunk:03d}.png",
        plt,
        np,
    )
    save_pairwise_heatmaps(
        pairwise,
        chunk_ranges,
        args.output_dir / "pairwise_chunk_mismatch_heatmaps.png",
        plt,
        np,
    )

    summary = {
        "input": str(args.input),
        "dataset": args.dataset,
        "frame_shape": [int(height), int(width)],
        "chunk_frames": args.chunk_frames,
        "reference_chunk": args.reference_chunk,
        "blinker_std_threshold": args.blinker_std_threshold,
        "valid_pixels_after_blinker_exclusion": int(np.count_nonzero(valid_pixels)),
        "excluded_pixels_after_blinker_exclusion": int(np.count_nonzero(~valid_pixels)),
        "chunks": [
            {
                "index": index,
                "start_frame": start,
                "end_frame_inclusive": end - 1,
                "raw_mean_imaging": float(chunk_means[index][imaging_mask].mean()),
                "raw_stddev_imaging_mean": float(chunk_stddevs[index][imaging_mask].mean()),
                "blinker_pixels": int(np.count_nonzero(chunk_blinker_masks[index])),
                "zlp_band_residual_mean_vs_reference": float(
                    residuals[index][zlp_band_mask].mean()
                ),
                "zlp_band_residual_rms_vs_reference": float(
                    np.sqrt(np.mean(np.square(residuals[index][zlp_band_mask])))
                ),
                "core_residual_rms_vs_reference": float(
                    np.sqrt(np.mean(np.square(residuals[index][core_mask])))
                ),
            }
            for index, (start, end) in enumerate(chunk_ranges)
        ],
        "pairwise": {
            key: value.tolist()
            for key, value in pairwise.items()
        },
        "plots": [
            "raw_chunk_mean_profiles.png",
            f"blr_residual_maps_ref_chunk_{args.reference_chunk:03d}.png",
            f"blr_residual_profiles_ref_chunk_{args.reference_chunk:03d}.png",
            f"zlp_folded_profiles_ref_chunk_{args.reference_chunk:03d}.png",
            f"zlp_folded_heatmaps_ref_chunk_{args.reference_chunk:03d}.png",
            f"suspicious_zlp_mod_trends_ref_chunk_{args.reference_chunk:03d}.png",
            "chunk_temporal_stddev_maps.png",
            "pairwise_chunk_mismatch_heatmaps.png",
        ],
    }
    (args.output_dir / "dark_recovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
