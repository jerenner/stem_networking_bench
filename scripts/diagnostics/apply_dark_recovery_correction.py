#!/usr/bin/env python3
"""Fit and evaluate a time-dependent ZLP dark-recovery correction.

The correction is fitted from temporal averages of a dark-frame acquisition. It
models the BLR-corrected residual at each selected detector pixel either with a
single slope anchored to an existing static dark frame or with an offset and
slope in each acquisition segment. The fitted model is nonzero only in a
configurable physical-column range repeated across the ZLP readouts.

The intended processing order is:

    static dark subtraction -> ImageJ BLR -> recovery slope correction

This script writes the correction model and diagnostic plots. Applying the
model to temporal mean images is exactly equivalent to applying it to every
frame and then averaging because all three operations above are linear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_nio_processing_analysis import (
    configure_matplotlib_cache,
    detector_regions,
    subtract_imagej_blr,
    symmetric_limit,
)


def normalize_dataset_path(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def read_single_image(h5_file, dataset_path: str, np):
    data = h5_file[normalize_dataset_path(dataset_path)][...]
    if data.ndim == 2:
        return data.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32, copy=False)
    raise ValueError(
        f"{dataset_path} must have shape [rows, cols] or [1, rows, cols], "
        f"got {data.shape}"
    )


def compute_mean_and_stddev(dataset, start: int, end: int, read_size: int, np):
    count = end - start
    total = np.zeros(dataset.shape[1:], dtype=np.float64)
    sumsq = np.zeros(dataset.shape[1:], dtype=np.float64)
    for block_start in range(start, end, read_size):
        block_end = min(end, block_start + read_size)
        block = dataset[block_start:block_end].astype(np.float64, copy=False)
        total += block.sum(axis=0)
        sumsq += np.square(block).sum(axis=0)
    mean = total / count
    variance = np.maximum(sumsq / count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def selected_zlp_columns(width: int,
                         zlp_width: int,
                         zlp_period: int,
                         mod_start: int,
                         mod_end: int,
                         np):
    if zlp_width > width or zlp_width % zlp_period != 0:
        raise ValueError("ZLP width must fit the image and be divisible by its period")
    if not (0 <= mod_start <= mod_end < zlp_period):
        raise ValueError("selected modulo-column range is outside one ZLP period")
    columns = np.arange(width)
    modulo = columns % zlp_period
    return (columns < zlp_width) & (modulo >= mod_start) & (modulo <= mod_end)


def masked_column_profile(image, rows: slice, valid_pixels, np):
    return np.nanmean(np.where(valid_pixels[rows], image[rows], np.nan), axis=0)


def chunk_residual_profiles(residuals,
                            chunk_bins: int,
                            rows: slice,
                            valid_pixels,
                            np):
    profiles = np.stack(
        [masked_column_profile(image, rows, valid_pixels, np) for image in residuals]
    )
    chunk_count = profiles.shape[0] // chunk_bins
    return profiles.reshape(chunk_count, chunk_bins, profiles.shape[1]).mean(axis=1)


def folded_profiles(profiles, zlp_width: int, zlp_period: int, np):
    return profiles[:, :zlp_width].reshape(
        profiles.shape[0], zlp_width // zlp_period, zlp_period
    ).mean(axis=1)


def save_column_comparison(pre_profiles,
                           post_profiles,
                           labels,
                           zlp_width: int,
                           selected_columns,
                           output_path: Path,
                           plt,
                           np):
    selected_indices = np.flatnonzero(selected_columns)
    padded_selection = np.r_[False, selected_columns[:zlp_width], False].astype(np.int8)
    selection_edges = np.diff(padded_selection)
    selection_starts = np.flatnonzero(selection_edges == 1)
    selection_ends = np.flatnonzero(selection_edges == -1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for row, half_name in enumerate(("Top imaging half", "Bottom imaging half")):
        for profiles, axis, stage in (
            (pre_profiles[row], axes[row, 0], "Before recovery correction"),
            (post_profiles[row], axes[row, 1], "After recovery correction"),
        ):
            for profile, label in zip(profiles, labels):
                axis.plot(profile[:zlp_width], linewidth=0.8, label=label)
            for start, end in zip(selection_starts, selection_ends):
                axis.axvspan(start, end - 1, color="tab:red", alpha=0.07)
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_title(f"{half_name}: {stage}")
            axis.set_xlabel("ZLP output column")
            axis.set_ylabel("Mean dark residual")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Dark-subtracted + BLR-corrected column means by 128-frame temporal chunk\n"
        f"Shaded bands are corrected ({selected_indices.size} output columns)"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summed_column_comparison(pre_profiles,
                                  post_profiles,
                                  zlp_width: int,
                                  selected_columns,
                                  mod_start: int,
                                  mod_end: int,
                                  output_path: Path,
                                  plt,
                                  np):
    summed = np.stack([
        pre_profiles[0].sum(axis=0),
        post_profiles[0].sum(axis=0),
        pre_profiles[1].sum(axis=0),
        post_profiles[1].sum(axis=0),
    ])
    limit = float(np.max(np.abs(summed[:, :zlp_width])))
    if limit == 0.0:
        limit = 1.0

    padded_selection = np.r_[False, selected_columns[:zlp_width], False].astype(np.int8)
    selection_edges = np.diff(padded_selection)
    selection_starts = np.flatnonzero(selection_edges == 1)
    selection_ends = np.flatnonzero(selection_edges == -1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = (
        (axes[0, 0], summed[0], "Top imaging half: before correction", "tab:blue"),
        (axes[0, 1], summed[1], "Top imaging half: after correction", "tab:orange"),
        (axes[1, 0], summed[2], "Bottom imaging half: before correction", "tab:blue"),
        (axes[1, 1], summed[3], "Bottom imaging half: after correction", "tab:orange"),
    )
    for axis, profile, title, color in panels:
        axis.plot(profile[:zlp_width], color=color, linewidth=0.9)
        for start, end in zip(selection_starts, selection_ends):
            axis.axvspan(start, end - 1, color="tab:red", alpha=0.08)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylim(-limit * 1.03, limit * 1.03)
        axis.set_title(title)
        axis.set_xlabel("ZLP output column")
        axis.set_ylabel("Sum of four 128-frame mean residuals")
        axis.grid(alpha=0.2)

    fig.suptitle(
        "Temporally summed ZLP column-mean profiles\n"
        f"Shaded bands are physical columns {mod_start}..{mod_end} "
        "in each repeated ZLP read"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_chunk_column_comparisons(pre_profiles,
                                      post_profiles,
                                      chunk_ranges,
                                      zlp_width: int,
                                      selected_columns,
                                      mod_start: int,
                                      mod_end: int,
                                      output_dir: Path,
                                      plt,
                                      np):
    all_profiles = np.concatenate([
        pre_profiles[0][:, :zlp_width],
        post_profiles[0][:, :zlp_width],
        pre_profiles[1][:, :zlp_width],
        post_profiles[1][:, :zlp_width],
    ])
    limit = float(np.max(np.abs(all_profiles)))
    if limit == 0.0:
        limit = 1.0

    padded_selection = np.r_[False, selected_columns[:zlp_width], False].astype(np.int8)
    selection_edges = np.diff(padded_selection)
    selection_starts = np.flatnonzero(selection_edges == 1)
    selection_ends = np.flatnonzero(selection_edges == -1)
    output_names = []

    for chunk, (start_frame, end_frame) in enumerate(chunk_ranges):
        fig, axes = plt.subplots(
            2, 2, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True
        )
        panels = (
            (axes[0, 0], pre_profiles[0][chunk], "Top half: before correction", "tab:blue"),
            (axes[0, 1], post_profiles[0][chunk], "Top half: after correction", "tab:orange"),
            (axes[1, 0], pre_profiles[1][chunk], "Bottom half: before correction", "tab:blue"),
            (axes[1, 1], post_profiles[1][chunk], "Bottom half: after correction", "tab:orange"),
        )
        for axis, profile, title, color in panels:
            axis.plot(profile[:zlp_width], color=color, linewidth=0.9)
            for selection_start, selection_end in zip(selection_starts, selection_ends):
                axis.axvspan(
                    selection_start, selection_end - 1, color="tab:red", alpha=0.08
                )
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_ylim(-limit * 1.03, limit * 1.03)
            axis.set_title(title)
            axis.set_xlabel("ZLP output column")
            axis.set_ylabel("Mean dark residual")
            axis.grid(alpha=0.2)

        fig.suptitle(
            f"ZLP column-mean profile, frames {start_frame}..{end_frame - 1}\n"
            f"Shaded bands are physical columns {mod_start}..{mod_end} "
            "in each repeated ZLP read"
        )
        output_name = (
            f"zlp_column_means_frames_{start_frame:03d}_{end_frame - 1:03d}_before_after.png"
        )
        fig.savefig(output_dir / output_name, dpi=180)
        plt.close(fig)
        output_names.append(output_name)

    return output_names


def save_folded_comparison(pre_folded,
                           post_folded,
                           labels,
                           mod_start: int,
                           mod_end: int,
                           output_path: Path,
                           plt):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for row, half_name in enumerate(("Top imaging half", "Bottom imaging half")):
        for profiles, axis, stage in (
            (pre_folded[row], axes[row, 0], "Before recovery correction"),
            (post_folded[row], axes[row, 1], "After recovery correction"),
        ):
            for profile, label in zip(profiles, labels):
                axis.plot(profile, linewidth=0.9, label=label)
            axis.axvspan(mod_start, mod_end, color="tab:red", alpha=0.12)
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_title(f"{half_name}: {stage}")
            axis.set_xlabel("Physical ZLP column modulo period")
            axis.set_ylabel("Mean dark residual")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, ncol=2)
    fig.suptitle("Folded ZLP column means before and after time-dependent correction")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_temporal_band_trend(pre_bin_profiles,
                             post_bin_profiles,
                             frame_midpoints,
                             zlp_width: int,
                             zlp_period: int,
                             mod_start: int,
                             mod_end: int,
                             reference_frame: float,
                             output_path: Path,
                             plt,
                             np):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for row, half_name in enumerate(("Top imaging half", "Bottom imaging half")):
        pre_folded = folded_profiles(
            pre_bin_profiles[row], zlp_width, zlp_period, np
        )
        post_folded = folded_profiles(
            post_bin_profiles[row], zlp_width, zlp_period, np
        )
        pre_band = pre_folded[:, mod_start:mod_end + 1].mean(axis=1)
        post_band = post_folded[:, mod_start:mod_end + 1].mean(axis=1)
        axes[row].plot(
            frame_midpoints, pre_band, marker="o", linewidth=1.7,
            label="before correction",
        )
        axes[row].plot(
            frame_midpoints, post_band, marker="o", linewidth=1.7,
            label="after correction",
        )
        axes[row].axhline(0.0, color="black", linewidth=0.7)
        axes[row].axvline(reference_frame, color="black", linestyle="--", linewidth=0.8)
        axes[row].set_title(half_name)
        axes[row].set_xlabel("Frame midpoint")
        axes[row].set_ylabel(f"Mean residual, physical columns {mod_start}..{mod_end}")
        axes[row].grid(alpha=0.2)
        axes[row].legend()
    fig.suptitle("Temporal evolution of the affected ZLP band")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_half_band_linear_fit(pre_bin_profiles,
                              frame_midpoints,
                              compact_intercepts,
                              compact_slopes,
                              reference_frame: float,
                              zlp_width: int,
                              zlp_period: int,
                              mod_start: int,
                              mod_end: int,
                              output_path: Path,
                              plt,
                              np):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    fit_times = np.linspace(frame_midpoints.min(), frame_midpoints.max(), 300)
    selected = slice(mod_start, mod_end + 1)

    for half, (axis, half_name) in enumerate(zip(axes, ("Top imaging half", "Bottom imaging half"))):
        folded = folded_profiles(
            pre_bin_profiles[half], zlp_width, zlp_period, np
        )
        observed = folded[:, selected].mean(axis=1)
        intercept = float(compact_intercepts[half, selected].mean())
        slope = float(compact_slopes[half, selected].mean())
        fitted_at_points = intercept + slope * (frame_midpoints - reference_frame)
        fitted_line = intercept + slope * (fit_times - reference_frame)
        residual_sum = float(np.sum(np.square(observed - fitted_at_points)))
        total_sum = float(np.sum(np.square(observed - observed.mean())))
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0

        axis.scatter(
            frame_midpoints,
            observed,
            color="tab:blue",
            s=36,
            label="32-frame residual means used by fit",
            zorder=3,
        )
        axis.plot(
            fit_times,
            fitted_line,
            color="tab:red",
            linewidth=2.0,
            label="least-squares linear fit",
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.axvline(reference_frame, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{half_name}: a={intercept:.4f}, b={slope:.6f} counts/frame, "
            f"R²={r_squared:.3f}"
        )
        axis.set_xlabel("Frame midpoint t")
        axis.set_ylabel(f"Mean residual, physical columns {mod_start}..{mod_end}")
        axis.grid(alpha=0.2)
        axis.legend()

    fig.suptitle(
        f"Half-band linear recovery model: correction(t) = a + b (t - {reference_frame:g})"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_slope_diagnostics(slope,
                           top_rows: slice,
                           bottom_rows: slice,
                           valid_pixels,
                           zlp_width: int,
                           zlp_period: int,
                           mod_start: int,
                           mod_end: int,
                           output_path: Path,
                           plt,
                           TwoSlopeNorm,
                           np):
    plotted = np.where(valid_pixels, slope, np.nan)
    limit = symmetric_limit(plotted[:, :zlp_width], np, 99.5)
    top_profile = masked_column_profile(slope, top_rows, valid_pixels, np)
    bottom_profile = masked_column_profile(slope, bottom_rows, valid_pixels, np)
    folded = folded_profiles(
        np.stack([top_profile, bottom_profile]), zlp_width, zlp_period, np
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    image = axes[0].imshow(
        plotted[:, :zlp_width],
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
        aspect="auto",
    )
    axes[0].set_title("Fitted recovery slope map in ZLP")
    axes[0].set_xlabel("ZLP output column")
    axes[0].set_ylabel("Row")
    fig.colorbar(image, ax=axes[0], label="Residual counts per frame")
    axes[1].plot(folded[0], label="top imaging half")
    axes[1].plot(folded[1], label="bottom imaging half")
    axes[1].axvspan(mod_start, mod_end, color="tab:red", alpha=0.12)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_title("Folded mean recovery slope")
    axes[1].set_xlabel("Physical ZLP column modulo period")
    axes[1].set_ylabel("Residual counts per frame")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input dark-frame HDF5 stack.")
    parser.add_argument("dark_frame", type=Path, help="Static dark-frame HDF5 file.")
    parser.add_argument("--input-dataset", default="/frames")
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--output-dir", type=Path, default=Path("dark_recovery_correction"))
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--fit-bin-frames", type=int, default=32)
    parser.add_argument("--plot-chunk-frames", type=int, default=128)
    parser.add_argument(
        "--fit-model",
        choices=(
            "global-linear",
            "half-band-linear",
            "folded-column-linear",
            "segment-linear",
        ),
        default="half-band-linear",
        help=(
            "global-linear fits one slope around the static-dark reference; "
            "half-band-linear fits one intercept/slope per detector half for "
            "the mean selected ZLP band; "
            "folded-column-linear fits one intercept/slope per detector half "
            "and physical ZLP column; "
            "segment-linear fits an offset and slope within each plot chunk"
        ),
    )
    parser.add_argument("--read-chunk-size", type=int, default=16)
    parser.add_argument("--reference-start-frame", type=int, default=128)
    parser.add_argument("--reference-frames", type=int, default=128)
    parser.add_argument("--blinker-std-threshold", type=float, default=500.0)
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-period", type=int, default=192)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument("--correct-mod-start", type=int, default=42)
    parser.add_argument("--correct-mod-end", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0 or args.fit_bin_frames <= 0 or args.plot_chunk_frames <= 0:
        raise ValueError("frame counts must be positive")
    if args.frames % args.fit_bin_frames != 0:
        raise ValueError("--frames must be divisible by --fit-bin-frames")
    if args.plot_chunk_frames % args.fit_bin_frames != 0:
        raise ValueError("--plot-chunk-frames must be divisible by --fit-bin-frames")
    if args.frames % args.plot_chunk_frames != 0:
        raise ValueError("--frames must be divisible by --plot-chunk-frames")

    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_output = args.model_output or args.output_dir / "dark_recovery_model.h5"
    model_output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dark_frame, "r") as dark_h5:
        dark_frame = read_single_image(dark_h5, args.dark_dataset, np)
        valid_pixels = read_single_image(dark_h5, args.valid_mask_dataset, np) != 0

    with h5py.File(args.input, "r") as input_h5:
        dataset = input_h5[normalize_dataset_path(args.input_dataset)]
        if dataset.ndim != 3:
            raise ValueError(f"input must have shape [frames, rows, cols], got {dataset.shape}")
        total_frames, height, width = dataset.shape
        end_frame = args.start_frame + args.frames
        if args.start_frame < 0 or end_frame > total_frames:
            raise ValueError("selected input frame range is outside the dataset")
        if dark_frame.shape != (height, width) or valid_pixels.shape != (height, width):
            raise ValueError("dark frame or valid-pixel mask shape does not match input")

        bin_means = []
        bin_stddevs = []
        bin_ranges = []
        for start in range(args.start_frame, end_frame, args.fit_bin_frames):
            end = start + args.fit_bin_frames
            mean, stddev = compute_mean_and_stddev(
                dataset, start, end, args.read_chunk_size, np
            )
            bin_means.append(mean)
            bin_stddevs.append(stddev)
            bin_ranges.append((start, end))

    bin_means = np.stack(bin_means)
    bin_stddevs = np.stack(bin_stddevs)
    if args.blinker_std_threshold > 0:
        valid_pixels &= ~np.any(bin_stddevs > args.blinker_std_threshold, axis=0)

    pre_residuals = np.stack([
        subtract_imagej_blr(
            (mean - dark_frame)[None, :, :],
            np,
            args.blr_rows,
            args.zlp_width,
            args.zlp_group_columns,
            args.core_group_columns,
        )[0]
        for mean in bin_means
    ])

    frame_midpoints = np.array(
        [(start + end - 1) / 2.0 for start, end in bin_ranges], dtype=np.float64
    )
    reference_frame = args.reference_start_frame + (args.reference_frames - 1) / 2.0
    corrected_columns = selected_zlp_columns(
        width,
        args.zlp_width,
        args.zlp_period,
        args.correct_mod_start,
        args.correct_mod_end,
        np,
    )
    bins_per_plot_chunk = args.plot_chunk_frames // args.fit_bin_frames
    top_rows, bottom_rows = detector_regions(height, args.edge_rows)
    if args.fit_model == "global-linear":
        time_offsets = (frame_midpoints - reference_frame).astype(np.float32)
        denominator = float(np.dot(time_offsets, time_offsets))
        if denominator == 0.0:
            raise ValueError("temporal bins do not span the reference frame")
        recovery_slopes = (
            np.tensordot(time_offsets, pre_residuals, axes=(0, 0)) / denominator
        ).astype(np.float32)[None, :, :]
        recovery_intercepts = np.zeros_like(recovery_slopes)
        segment_midpoints = np.array([reference_frame], dtype=np.float32)
        segment_starts = np.array([args.start_frame], dtype=np.int64)
        post_residuals = (
            pre_residuals
            - time_offsets[:, None, None] * recovery_slopes[0][None, :, :]
        )
        compact_intercepts = None
        compact_slopes = None
        linear_fit_point_count = len(frame_midpoints)
    elif args.fit_model in ("half-band-linear", "folded-column-linear"):
        chunk_count = args.frames // args.plot_chunk_frames
        chunk_residuals = pre_residuals.reshape(
            chunk_count, bins_per_plot_chunk, height, width
        ).mean(axis=1, dtype=np.float32)
        chunk_midpoints = frame_midpoints.reshape(
            chunk_count, bins_per_plot_chunk
        ).mean(axis=1)
        if args.fit_model == "half-band-linear":
            fit_residuals = pre_residuals
            fit_midpoints = frame_midpoints
        else:
            fit_residuals = chunk_residuals
            fit_midpoints = chunk_midpoints
        fit_offsets = (fit_midpoints - reference_frame).astype(np.float32)
        linear_fit_point_count = len(fit_midpoints)

        folded_values = np.empty(
            (2, linear_fit_point_count, args.zlp_period), dtype=np.float32
        )
        for half, rows in enumerate((top_rows, bottom_rows)):
            profiles = np.stack([
                masked_column_profile(image, rows, valid_pixels, np)
                for image in fit_residuals
            ])
            folded_values[half] = folded_profiles(
                profiles, args.zlp_width, args.zlp_period, np
            )

        sample_count = float(linear_fit_point_count)
        sum_x = float(fit_offsets.sum())
        sum_x2 = float(np.dot(fit_offsets, fit_offsets))
        denominator = sample_count * sum_x2 - sum_x * sum_x
        if denominator == 0.0:
            raise ValueError("folded-column-linear fitting needs at least two chunk times")
        selected_mods = np.zeros(args.zlp_period, dtype=bool)
        selected_mods[args.correct_mod_start:args.correct_mod_end + 1] = True
        if args.fit_model == "half-band-linear":
            band_values = folded_values[:, :, selected_mods].mean(axis=2)
            sum_y = band_values.sum(axis=1)
            sum_xy = np.sum(band_values * fit_offsets[None, :], axis=1)
            band_slopes = (
                (sample_count * sum_xy - sum_x * sum_y) / denominator
            ).astype(np.float32)
            band_intercepts = (
                (sum_y - band_slopes * sum_x) / sample_count
            ).astype(np.float32)
            compact_intercepts = np.zeros(
                (2, args.zlp_period), dtype=np.float32
            )
            compact_slopes = np.zeros_like(compact_intercepts)
            compact_intercepts[:, selected_mods] = band_intercepts[:, None]
            compact_slopes[:, selected_mods] = band_slopes[:, None]
        else:
            sum_y = folded_values.sum(axis=1)
            sum_xy = np.sum(
                folded_values * fit_offsets[None, :, None], axis=1
            )
            compact_slopes = (
                (sample_count * sum_xy - sum_x * sum_y) / denominator
            ).astype(np.float32)
            compact_intercepts = (
                (sum_y - compact_slopes * sum_x) / sample_count
            ).astype(np.float32)
            compact_intercepts[:, ~selected_mods] = 0.0
            compact_slopes[:, ~selected_mods] = 0.0

        intercept_map = np.zeros((height, width), dtype=np.float32)
        slope_map = np.zeros_like(intercept_map)
        zlp_repeats = args.zlp_width // args.zlp_period
        for half, rows in enumerate((slice(0, height // 2), slice(height // 2, height))):
            intercept_map[rows, :args.zlp_width] = np.tile(
                compact_intercepts[half], zlp_repeats
            )[None, :]
            slope_map[rows, :args.zlp_width] = np.tile(
                compact_slopes[half], zlp_repeats
            )[None, :]
        intercept_map[~valid_pixels] = 0.0
        slope_map[~valid_pixels] = 0.0

        recovery_intercepts = intercept_map[None, :, :]
        recovery_slopes = slope_map[None, :, :]
        segment_midpoints = np.array([reference_frame], dtype=np.float32)
        segment_starts = np.array([args.start_frame], dtype=np.int64)
        time_offsets = (frame_midpoints - reference_frame).astype(np.float32)
        post_residuals = (
            pre_residuals
            - recovery_intercepts[0][None, :, :]
            - time_offsets[:, None, None] * recovery_slopes[0][None, :, :]
        )
    else:
        segment_count = args.frames // args.plot_chunk_frames
        recovery_intercepts = np.empty(
            (segment_count, height, width), dtype=np.float32
        )
        recovery_slopes = np.empty_like(recovery_intercepts)
        post_residuals = np.empty_like(pre_residuals)
        segment_midpoints = np.empty(segment_count, dtype=np.float32)
        segment_starts = np.empty(segment_count, dtype=np.int64)
        for segment in range(segment_count):
            first_bin = segment * bins_per_plot_chunk
            last_bin = first_bin + bins_per_plot_chunk
            segment_values = pre_residuals[first_bin:last_bin]
            segment_times = frame_midpoints[first_bin:last_bin].astype(np.float32)
            segment_midpoint = float(segment_times.mean())
            local_offsets = segment_times - segment_midpoint
            denominator = float(np.dot(local_offsets, local_offsets))
            if denominator == 0.0:
                raise ValueError("segment-linear fitting needs at least two fit bins per segment")
            intercept = segment_values.mean(axis=0, dtype=np.float32)
            slope = (
                np.tensordot(
                    local_offsets,
                    segment_values - intercept[None, :, :],
                    axes=(0, 0),
                )
                / denominator
            ).astype(np.float32)
            recovery_intercepts[segment] = intercept
            recovery_slopes[segment] = slope
            segment_midpoints[segment] = segment_midpoint
            segment_starts[segment] = args.start_frame + segment * args.plot_chunk_frames
            post_residuals[first_bin:last_bin] = (
                segment_values
                - intercept[None, :, :]
                - local_offsets[:, None, None] * slope[None, :, :]
            )
        compact_intercepts = None
        compact_slopes = None
        linear_fit_point_count = bins_per_plot_chunk

    if args.fit_model not in ("half-band-linear", "folded-column-linear"):
        recovery_intercepts[:, :, ~corrected_columns] = 0.0
        recovery_slopes[:, :, ~corrected_columns] = 0.0
        recovery_intercepts[:, ~valid_pixels] = 0.0
        recovery_slopes[:, ~valid_pixels] = 0.0
    # Recompute map-based models after masking invalid and unselected pixels.
    if args.fit_model == "global-linear":
        post_residuals = (
            pre_residuals
            - time_offsets[:, None, None] * recovery_slopes[0][None, :, :]
        )
    elif args.fit_model == "segment-linear":
        for segment in range(recovery_slopes.shape[0]):
            first_bin = segment * bins_per_plot_chunk
            last_bin = first_bin + bins_per_plot_chunk
            local_offsets = (
                frame_midpoints[first_bin:last_bin] - segment_midpoints[segment]
            ).astype(np.float32)
            post_residuals[first_bin:last_bin] = (
                pre_residuals[first_bin:last_bin]
                - recovery_intercepts[segment][None, :, :]
                - local_offsets[:, None, None] * recovery_slopes[segment][None, :, :]
            )
    del bin_means, bin_stddevs

    pre_bin_profiles = [
        np.stack([masked_column_profile(image, rows, valid_pixels, np) for image in pre_residuals])
        for rows in (top_rows, bottom_rows)
    ]
    post_bin_profiles = [
        np.stack([masked_column_profile(image, rows, valid_pixels, np) for image in post_residuals])
        for rows in (top_rows, bottom_rows)
    ]
    pre_chunk_profiles = [
        profiles.reshape(-1, bins_per_plot_chunk, width).mean(axis=1)
        for profiles in pre_bin_profiles
    ]
    post_chunk_profiles = [
        profiles.reshape(-1, bins_per_plot_chunk, width).mean(axis=1)
        for profiles in post_bin_profiles
    ]
    pre_folded = [
        folded_profiles(profiles, args.zlp_width, args.zlp_period, np)
        for profiles in pre_chunk_profiles
    ]
    post_folded = [
        folded_profiles(profiles, args.zlp_width, args.zlp_period, np)
        for profiles in post_chunk_profiles
    ]
    labels = [
        f"frames {args.start_frame + i * args.plot_chunk_frames}.."
        f"{args.start_frame + (i + 1) * args.plot_chunk_frames - 1}"
        for i in range(args.frames // args.plot_chunk_frames)
    ]
    plot_chunk_ranges = [
        (
            args.start_frame + i * args.plot_chunk_frames,
            args.start_frame + (i + 1) * args.plot_chunk_frames,
        )
        for i in range(args.frames // args.plot_chunk_frames)
    ]

    save_column_comparison(
        pre_chunk_profiles,
        post_chunk_profiles,
        labels,
        args.zlp_width,
        corrected_columns,
        args.output_dir / "zlp_column_means_before_after.png",
        plt,
        np,
    )
    save_summed_column_comparison(
        pre_chunk_profiles,
        post_chunk_profiles,
        args.zlp_width,
        corrected_columns,
        args.correct_mod_start,
        args.correct_mod_end,
        args.output_dir / "zlp_column_means_summed_before_after.png",
        plt,
        np,
    )
    per_chunk_plot_names = save_per_chunk_column_comparisons(
        pre_chunk_profiles,
        post_chunk_profiles,
        plot_chunk_ranges,
        args.zlp_width,
        corrected_columns,
        args.correct_mod_start,
        args.correct_mod_end,
        args.output_dir,
        plt,
        np,
    )
    save_folded_comparison(
        pre_folded,
        post_folded,
        labels,
        args.correct_mod_start,
        args.correct_mod_end,
        args.output_dir / "folded_zlp_column_means_before_after.png",
        plt,
    )
    save_temporal_band_trend(
        pre_bin_profiles,
        post_bin_profiles,
        frame_midpoints,
        args.zlp_width,
        args.zlp_period,
        args.correct_mod_start,
        args.correct_mod_end,
        reference_frame,
        args.output_dir / "affected_zlp_band_temporal_trend_before_after.png",
        plt,
        np,
    )
    if args.fit_model == "half-band-linear":
        save_half_band_linear_fit(
            pre_bin_profiles,
            frame_midpoints,
            compact_intercepts,
            compact_slopes,
            reference_frame,
            args.zlp_width,
            args.zlp_period,
            args.correct_mod_start,
            args.correct_mod_end,
            args.output_dir / "affected_zlp_band_linear_fit.png",
            plt,
            np,
        )
    save_slope_diagnostics(
        recovery_slopes.mean(axis=0),
        top_rows,
        bottom_rows,
        valid_pixels,
        args.zlp_width,
        args.zlp_period,
        args.correct_mod_start,
        args.correct_mod_end,
        args.output_dir / "recovery_slope_diagnostics.png",
        plt,
        TwoSlopeNorm,
        np,
    )

    with h5py.File(model_output, "w") as model_h5:
        model_h5.create_dataset("processed", data=dark_frame[None], dtype=np.float32)
        intercept_dataset = model_h5.create_dataset(
            "dark_recovery_intercept", data=recovery_intercepts, dtype=np.float32
        )
        slope_dataset = model_h5.create_dataset(
            "dark_recovery_slope", data=recovery_slopes, dtype=np.float32
        )
        model_h5.create_dataset("segment_start_frame", data=segment_starts)
        model_h5.create_dataset("segment_midpoint", data=segment_midpoints)
        if compact_intercepts is not None:
            model_h5.create_dataset(
                "folded_column_intercept", data=compact_intercepts, dtype=np.float32
            )
            model_h5.create_dataset(
                "folded_column_slope", data=compact_slopes, dtype=np.float32
            )
        model_h5.create_dataset(
            "valid_pixel_mask", data=valid_pixels[None].astype(np.uint8), dtype=np.uint8
        )
        intercept_dataset.attrs["description"] = (
            "Post-BLR dark residual offset for each temporal segment"
        )
        slope_dataset.attrs["description"] = (
            "Post-BLR dark residual slope in counts per frame for each temporal segment"
        )
        slope_dataset.attrs["fit_model"] = args.fit_model
        slope_dataset.attrs["reference_frame"] = reference_frame
        slope_dataset.attrs["correct_mod_start"] = args.correct_mod_start
        slope_dataset.attrs["correct_mod_end"] = args.correct_mod_end
        slope_dataset.attrs["zlp_width"] = args.zlp_width
        slope_dataset.attrs["zlp_period"] = args.zlp_period
        slope_dataset.attrs["processing_stage"] = "after_static_dark_and_blr"

    band = slice(args.correct_mod_start, args.correct_mod_end + 1)
    metrics = {}
    for index, half_name in enumerate(("top", "bottom")):
        pre_band = folded_profiles(
            pre_bin_profiles[index], args.zlp_width, args.zlp_period, np
        )[:, band].mean(axis=1)
        post_band = folded_profiles(
            post_bin_profiles[index], args.zlp_width, args.zlp_period, np
        )[:, band].mean(axis=1)
        metrics[half_name] = {
            "pre_temporal_span": float(np.ptp(pre_band)),
            "post_temporal_span": float(np.ptp(post_band)),
            "pre_temporal_rms": float(np.sqrt(np.mean(np.square(pre_band)))),
            "post_temporal_rms": float(np.sqrt(np.mean(np.square(post_band)))),
            "pre_bin_means": pre_band.tolist(),
            "post_bin_means": post_band.tolist(),
        }

    summary = {
        "input": str(args.input),
        "input_dataset": normalize_dataset_path(args.input_dataset),
        "static_dark_frame": str(args.dark_frame),
        "model_output": str(model_output),
        "fit_model": args.fit_model,
        "processing_order": [
            "static_dark_subtraction",
            "ImageJ_BLR",
            "recovery_intercept_and_slope",
        ],
        "fit_bin_frames": args.fit_bin_frames,
        "linear_fit_point_count": linear_fit_point_count,
        "segment_frames": args.plot_chunk_frames,
        "segment_start_frames": segment_starts.tolist(),
        "segment_midpoints": segment_midpoints.tolist(),
        "reference_frame": reference_frame,
        "corrected_physical_zlp_columns": [args.correct_mod_start, args.correct_mod_end],
        "corrected_output_columns": int(np.count_nonzero(corrected_columns)),
        "valid_pixels_used": int(np.count_nonzero(valid_pixels)),
        "validation_note": (
            "The model and diagnostics use the same dark acquisition. Validate "
            "transfer on an independent time-matched dark acquisition before "
            "applying this calibration to scientific data."
        ),
        "metrics": metrics,
        "plots": [
            "zlp_column_means_before_after.png",
            "zlp_column_means_summed_before_after.png",
            *per_chunk_plot_names,
            "folded_zlp_column_means_before_after.png",
            "affected_zlp_band_temporal_trend_before_after.png",
            *(
                ["affected_zlp_band_linear_fit.png"]
                if args.fit_model == "half-band-linear"
                else []
            ),
            "recovery_slope_diagnostics.png",
        ],
    }
    (args.output_dir / "dark_recovery_correction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
