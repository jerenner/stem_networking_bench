#!/usr/bin/env python3
"""Run a threshold-before-summing study for the NiO beam-current data.

The ImageJ analysis applied a positive threshold after dark subtraction, BLR,
and masking, but before summing. Thresholds were region-specific:

  ZLP threshold  = NUM_SIGMAS * stddev(dark-subtracted dark ZLP residuals)
  Core threshold = NUM_SIGMAS * stddev(dark-subtracted dark CoreLoss residuals)

This script reproduces that strategy while also sweeping multiple sigma
multipliers so the threshold sensitivity can be inspected across beam currents.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stem_analysis.config import ProcessorConfig
from stem_analysis.dm4 import load_dm4, normalize_to_frame_stack
from stem_analysis.hdf5 import read_single_image
from stem_analysis.plotting import configure_matplotlib_cache, detector_regions
from stem_analysis.processing import apply_dynamic_and_valid_mask, subtract_imagej_blr
from stem_analysis.spectra import fold_zlp, safe_divide


def parse_multipliers(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one threshold multiplier is required")
    if any(value < 0 for value in values):
        raise ValueError(f"threshold multipliers must be non-negative: {values}")
    return values


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def load_dark_assets(dark_path: Path, args, h5py, np):
    with h5py.File(dark_path, "r") as dark_h5:
        dark_frame = read_single_image(dark_h5, args.dark_dataset, np)
        valid_mask = read_single_image(dark_h5, args.valid_mask_dataset, np)
    if dark_frame.shape != (args.height, args.width):
        raise ValueError(
            f"{dark_path} dark frame shape {dark_frame.shape} does not match "
            f"{(args.height, args.width)}"
        )
    return dark_frame, valid_mask


def accumulate_clipped_stats(values, accumulator, np, clip_abs: float) -> None:
    finite = values[np.isfinite(values)]
    if clip_abs > 0:
        finite = finite[(finite >= -clip_abs) & (finite <= clip_abs)]
    if finite.size == 0:
        return
    as_float = finite.astype(np.float64, copy=False)
    accumulator["count"] += int(as_float.size)
    accumulator["sum"] += float(as_float.sum(dtype=np.float64))
    accumulator["sumsq"] += float(np.square(as_float, dtype=np.float64).sum(dtype=np.float64))


def finalize_stats(accumulator, np) -> dict[str, float | int]:
    count = accumulator["count"]
    if count <= 1:
        return {"count": int(count), "mean": float("nan"), "stddev": float("nan")}
    mean = accumulator["sum"] / count
    variance = max(accumulator["sumsq"] / count - mean * mean, 0.0)
    return {"count": int(count), "mean": float(mean), "stddev": float(np.sqrt(variance))}


def compute_imagej_dark_noise(dark_files: list[Path], dark_frame, args, np) -> dict[str, dict]:
    """Compute ImageJ-style clipped dark-subtracted noise for ZLP/CoreLoss."""
    accumulators = {
        "zlp": {"count": 0, "sum": 0.0, "sumsq": 0.0},
        "core": {"count": 0, "sum": 0.0, "sumsq": 0.0},
    }

    for path in dark_files:
        data, info = load_dm4(path, args.reader)
        stack = normalize_to_frame_stack(data, None, args.height, args.width)
        print(f"  dark noise {path.name}: reader={info['reader']} shape={tuple(stack.shape)}", flush=True)
        for start in range(0, stack.shape[0], args.read_chunk_size):
            end = min(stack.shape[0], start + args.read_chunk_size)
            block = materialize(stack[start:end], np).astype(np.float32, copy=False)
            residual = block - dark_frame[None, :, :]
            accumulate_clipped_stats(
                residual[:, :, :args.zlp_width],
                accumulators["zlp"],
                np,
                args.noise_clip_abs,
            )
            accumulate_clipped_stats(
                residual[:, :, args.zlp_width:],
                accumulators["core"],
                np,
                args.noise_clip_abs,
            )
        del stack, data

    return {region: finalize_stats(acc, np) for region, acc in accumulators.items()}


def add_valid_counts(counts, zero_mask, frame_count: int, regions, np):
    for half, rows in enumerate(regions):
        counts[half] += frame_count * np.count_nonzero(~zero_mask[rows], axis=0)


def accumulate_region_thresholds(region_values,
                                 region_start: int,
                                 half: int,
                                 sums,
                                 retained_counts,
                                 thresholds,
                                 np) -> None:
    for threshold_index, threshold in enumerate(thresholds):
        keep = region_values > threshold
        sums[threshold_index, half, region_start:region_start + region_values.shape[-1]] += np.where(
            keep, region_values, 0.0
        ).sum(axis=(0, 1), dtype=np.float64)
        retained_counts[threshold_index, half, region_start:region_start + region_values.shape[-1]] += (
            np.count_nonzero(keep, axis=(0, 1))
        )


def accumulate_thresholded_spectra(spectrum_files: list[Path],
                                   dark_frame,
                                   valid_mask,
                                   multipliers,
                                   zlp_stddev: float,
                                   core_stddev: float,
                                   args,
                                   np):
    regions = detector_regions(args.height, args.edge_rows)
    multiplier_count = len(multipliers)
    file_count = len(spectrum_files)

    thresholded_sums = np.zeros((multiplier_count, 2, args.width), dtype=np.float64)
    retained_counts = np.zeros((multiplier_count, 2, args.width), dtype=np.float64)
    per_file_sums = np.zeros((file_count, multiplier_count, 2, args.width), dtype=np.float64)
    per_file_retained_counts = np.zeros_like(per_file_sums)
    unthresholded_sums = np.zeros((2, args.width), dtype=np.float64)
    valid_counts = np.zeros((2, args.width), dtype=np.float64)
    per_file_unthresholded_sums = np.zeros((file_count, 2, args.width), dtype=np.float64)
    per_file_valid_counts = np.zeros_like(per_file_unthresholded_sums)
    batch_sums = []
    batch_counts = []
    batch_source_indices = []
    batch_start_frames = []
    batch_end_frames = []
    batch_mask_pixels = []
    frames_per_file = np.zeros(file_count, dtype=np.int64)

    zlp_thresholds = np.asarray(multipliers, dtype=np.float64) * zlp_stddev
    core_thresholds = np.asarray(multipliers, dtype=np.float64) * core_stddev
    mask_config = ProcessorConfig(
        noop=True,
        subtract_dark_frame=False,
        apply_valid_pixel_mask=not args.disable_valid_pixel_mask,
        apply_blr_correction=False,
        apply_dynamic_half_column_mask=not args.disable_dynamic_mask,
        dynamic_mask_median_window_pixels=args.median_window_pixels,
        dynamic_mask_threshold_ratio=args.dynamic_threshold_ratio,
        dynamic_mask_threshold_offset=args.dynamic_threshold_offset,
        dynamic_mask_excluded_edge_rows=args.edge_rows,
        dynamic_mask_two_sided=True,
    )

    for file_index, path in enumerate(spectrum_files):
        data, info = load_dm4(path, args.reader)
        stack = normalize_to_frame_stack(data, None, args.height, args.width)
        end_frame = stack.shape[0]
        if args.max_frames_per_file is not None:
            end_frame = min(end_frame, args.start_frame + args.max_frames_per_file)
        if args.start_frame < 0 or args.start_frame >= end_frame:
            raise ValueError(f"invalid selected range for {path}")

        print(
            f"  spectrum {path.name}: reader={info['reader']} shape={tuple(stack.shape)} "
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

            batch_mean = (raw_sum / frame_count).astype(np.float32)
            batch_mean -= dark_frame
            batch_mean = subtract_imagej_blr(
                batch_mean[None],
                np,
                args.blr_rows,
                args.zlp_width,
                args.zlp_group_columns,
                args.core_group_columns,
            )
            zero_mask = apply_dynamic_and_valid_mask(batch_mean, valid_mask, mask_config, np)
            add_valid_counts(valid_counts, zero_mask, frame_count, regions, np)
            add_valid_counts(per_file_valid_counts[file_index], zero_mask, frame_count, regions, np)

            batch_threshold_sums = np.zeros((multiplier_count, 2, args.width), dtype=np.float64)
            batch_threshold_retained = np.zeros_like(batch_threshold_sums)

            for read_start in range(tensor_start, tensor_end, args.read_chunk_size):
                read_end = min(tensor_end, read_start + args.read_chunk_size)
                corrected = materialize(stack[read_start:read_end], np).astype(np.float32, copy=True)
                corrected -= dark_frame[None, :, :]
                corrected = subtract_imagej_blr(
                    corrected,
                    np,
                    args.blr_rows,
                    args.zlp_width,
                    args.zlp_group_columns,
                    args.core_group_columns,
                )
                corrected[:, zero_mask] = 0.0

                for half, rows in enumerate(regions):
                    half_values = corrected[:, rows, :]
                    half_sum = half_values.sum(axis=(0, 1), dtype=np.float64)
                    unthresholded_sums[half] += half_sum
                    per_file_unthresholded_sums[file_index, half] += half_sum

                    zlp_values = half_values[:, :, :args.zlp_width]
                    core_values = half_values[:, :, args.zlp_width:]
                    accumulate_region_thresholds(
                        zlp_values,
                        0,
                        half,
                        thresholded_sums,
                        retained_counts,
                        zlp_thresholds,
                        np,
                    )
                    accumulate_region_thresholds(
                        zlp_values,
                        0,
                        half,
                        per_file_sums[file_index],
                        per_file_retained_counts[file_index],
                        zlp_thresholds,
                        np,
                    )
                    accumulate_region_thresholds(
                        zlp_values,
                        0,
                        half,
                        batch_threshold_sums,
                        batch_threshold_retained,
                        zlp_thresholds,
                        np,
                    )
                    accumulate_region_thresholds(
                        core_values,
                        args.zlp_width,
                        half,
                        thresholded_sums,
                        retained_counts,
                        core_thresholds,
                        np,
                    )
                    accumulate_region_thresholds(
                        core_values,
                        args.zlp_width,
                        half,
                        per_file_sums[file_index],
                        per_file_retained_counts[file_index],
                        core_thresholds,
                        np,
                    )
                    accumulate_region_thresholds(
                        core_values,
                        args.zlp_width,
                        half,
                        batch_threshold_sums,
                        batch_threshold_retained,
                        core_thresholds,
                        np,
                    )

            batch_sums.append(batch_threshold_sums)
            batch_counts.append(batch_threshold_retained)
            batch_source_indices.append(file_index)
            batch_start_frames.append(tensor_start)
            batch_end_frames.append(tensor_end - 1)
            batch_mask_pixels.append(int(zero_mask.sum()))
            frames_per_file[file_index] += frame_count
            print(
                f"    batch {tensor_start}..{tensor_end - 1}: masked {int(zero_mask.sum())} pixels",
                flush=True,
            )

        del stack, data

    return {
        "multipliers": np.asarray(multipliers, dtype=np.float64),
        "zlp_thresholds": zlp_thresholds,
        "core_thresholds": core_thresholds,
        "thresholded_sums": thresholded_sums,
        "retained_counts": retained_counts,
        "unthresholded_sums": unthresholded_sums,
        "valid_counts": valid_counts,
        "per_file_sums": per_file_sums,
        "per_file_retained_counts": per_file_retained_counts,
        "per_file_unthresholded_sums": per_file_unthresholded_sums,
        "per_file_valid_counts": per_file_valid_counts,
        "per_batch_sums": np.stack(batch_sums),
        "per_batch_retained_counts": np.stack(batch_counts),
        "per_batch_source_index": np.asarray(batch_source_indices, dtype=np.int32),
        "per_batch_start_frame": np.asarray(batch_start_frames, dtype=np.int64),
        "per_batch_end_frame": np.asarray(batch_end_frames, dtype=np.int64),
        "per_batch_masked_pixel_count": np.asarray(batch_mask_pixels, dtype=np.int64),
        "frames_per_file": frames_per_file,
    }


def combined_mean(sums, counts, np):
    return safe_divide(sums.sum(axis=0), counts.sum(axis=0), np)


def positive_area(profile, np):
    baseline = np.nanmedian(profile[-64:])
    return float(np.nansum(np.maximum(profile - baseline, 0.0)))


def threshold_metrics(result, np, zlp_width: int, zlp_period: int):
    unthresholded = combined_mean(result["unthresholded_sums"], result["valid_counts"], np)
    unthresholded_folded, _, _ = fold_zlp(
        result["unthresholded_sums"].sum(axis=0),
        result["valid_counts"].sum(axis=0),
        zlp_width,
        zlp_period,
        np,
    )
    rows = []
    for index, multiplier in enumerate(result["multipliers"]):
        spectrum = combined_mean(result["thresholded_sums"][index], result["valid_counts"], np)
        folded, _, _ = fold_zlp(
            result["thresholded_sums"][index].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            zlp_width,
            zlp_period,
            np,
        )
        retained = safe_divide(
            result["retained_counts"][index].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            np,
        )
        rows.append({
            "multiplier": float(multiplier),
            "zlp_threshold": float(result["zlp_thresholds"][index]),
            "core_threshold": float(result["core_thresholds"][index]),
            "zlp_peak_height": float(np.nanmax(folded)),
            "zlp_peak_retained_vs_unthresholded": float(np.nanmax(folded) / np.nanmax(unthresholded_folded)),
            "zlp_positive_area": positive_area(folded, np),
            "zlp_positive_area_retained_vs_unthresholded": float(
                positive_area(folded, np) / positive_area(unthresholded_folded, np)
            ),
            "core_early_mean": float(np.nanmean(spectrum[zlp_width:zlp_width + 256])),
            "core_early_retained_vs_unthresholded": float(
                np.nanmean(spectrum[zlp_width:zlp_width + 256])
                / np.nanmean(unthresholded[zlp_width:zlp_width + 256])
            ),
            "zlp_sample_retained_fraction": float(np.nanmean(retained[:zlp_width])),
            "core_sample_retained_fraction": float(np.nanmean(retained[zlp_width:])),
        })
    return rows


def save_current_outputs(output_dir: Path,
                         result,
                         noise,
                         source_files: list[Path],
                         dark_files: list[Path],
                         dark_path: Path,
                         args,
                         h5py,
                         np,
                         plt) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = threshold_metrics(result, np, args.zlp_width, args.zlp_period)

    with h5py.File(output_dir / "threshold_spectra.h5", "w") as h5:
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                h5.create_dataset(key, data=value)
        h5.attrs["source_files"] = np.array(
            [str(path) for path in source_files], dtype=h5py.string_dtype(encoding="utf-8")
        )
        h5.attrs["dark_files"] = np.array(
            [str(path) for path in dark_files], dtype=h5py.string_dtype(encoding="utf-8")
        )
        h5.attrs["dark_frame"] = str(dark_path)
        h5.attrs["threshold_strategy"] = "ImageJ-style region thresholds: multiplier * clipped dark-subtracted dark stddev"
        h5.attrs["processing_order"] = json.dumps([
            "dark_subtraction",
            "ImageJ_grouped_BLR",
            "valid_pixel_mask",
            "dynamic_half_column_mask",
            "positive_threshold",
            "column_accumulation",
        ])

    fields = list(metrics[0].keys())
    with (output_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)

    noise_summary = {
        "imagej_macro_reference": "Threshold_for_CountingSumming.ijm",
        "noise_clip_abs": args.noise_clip_abs,
        "num_sigmas_reference": args.recommended_multiplier,
        "zlp": noise["zlp"],
        "core": noise["core"],
    }
    (output_dir / "threshold_noise_summary.json").write_text(
        json.dumps(noise_summary, indent=2), encoding="utf-8"
    )

    columns = np.arange(args.width)
    unthresholded = combined_mean(result["unthresholded_sums"], result["valid_counts"], np)
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    axes[0].plot(columns, unthresholded, color="black", linewidth=1.1, label="no threshold")
    for index, multiplier in enumerate(result["multipliers"]):
        spectrum = combined_mean(result["thresholded_sums"][index], result["valid_counts"], np)
        axes[0].plot(columns, spectrum, linewidth=0.75, label=f"{multiplier:g} sigma")
        axes[1].plot(
            columns[args.zlp_width:],
            spectrum[args.zlp_width:],
            linewidth=0.75,
            label=f"{multiplier:g} sigma",
        )
    axes[0].axvline(args.zlp_width, color="tab:red", linestyle="--", linewidth=0.8)
    axes[0].set_title("Threshold sweep: full detector-column spectrum")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[1].set_title("Threshold sweep: CoreLoss region")
    axes[1].set_xlabel("Detector output column")
    axes[1].set_ylabel("Mean corrected detector value")
    for metric_name, label in (
        ("zlp_peak_retained_vs_unthresholded", "ZLP peak"),
        ("zlp_positive_area_retained_vs_unthresholded", "ZLP positive area"),
        ("core_early_retained_vs_unthresholded", "early CoreLoss mean"),
        ("zlp_sample_retained_fraction", "ZLP retained samples"),
        ("core_sample_retained_fraction", "Core retained samples"),
    ):
        axes[2].plot(
            [row["multiplier"] for row in metrics],
            [row[metric_name] for row in metrics],
            marker="o",
            label=label,
        )
    axes[2].axvline(args.recommended_multiplier, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("Threshold sensitivity metrics")
    axes[2].set_xlabel("Threshold multiplier (sigma)")
    axes[2].set_ylabel("Fraction / ratio")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=3)
    fig.savefig(output_dir / "threshold_sweep_spectra.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    folded_unthresholded, _, _ = fold_zlp(
        result["unthresholded_sums"].sum(axis=0),
        result["valid_counts"].sum(axis=0),
        args.zlp_width,
        args.zlp_period,
        np,
    )
    axis.plot(folded_unthresholded, color="black", linewidth=1.1, label="no threshold")
    for index, multiplier in enumerate(result["multipliers"]):
        folded, _, _ = fold_zlp(
            result["thresholded_sums"][index].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            args.zlp_width,
            args.zlp_period,
            np,
        )
        axis.plot(folded, linewidth=0.9, label=f"{multiplier:g} sigma")
    axis.set_title("Threshold sweep: folded ZLP")
    axis.set_xlabel(f"Physical ZLP detector column modulo {args.zlp_period}")
    axis.set_ylabel("Mean corrected detector value")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(output_dir / "threshold_sweep_zlp_folded.png", dpi=180)
    plt.close(fig)

    return metrics


def save_comparison_outputs(output_root: Path, current_results, args, np, plt) -> None:
    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in current_results:
        for metric in item["metrics"]:
            rows.append({
                "current_key": item["current_key"],
                "current_label": item["current_label"],
                "current_pa": item["current_pa"],
                **metric,
            })
    fields = list(rows[0].keys())
    with (comparison_dir / "threshold_metrics_all_currents.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    recommended = min(
        range(len(current_results[0]["result"]["multipliers"])),
        key=lambda index: abs(
            current_results[0]["result"]["multipliers"][index] - args.recommended_multiplier
        ),
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(current_results)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for item, color in zip(current_results, colors):
        result = item["result"]
        spectrum = combined_mean(result["thresholded_sums"][recommended], result["valid_counts"], np)
        folded, _, _ = fold_zlp(
            result["thresholded_sums"][recommended].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            args.zlp_width,
            args.zlp_period,
            np,
        )
        axes[0].plot(spectrum, color=color, linewidth=0.75, label=item["current_label"])
        axes[1].plot(folded, color=color, linewidth=0.9, label=item["current_label"])
    axes[0].axvline(args.zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"ImageJ-style {args.recommended_multiplier:g}-sigma thresholded spectra")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean corrected detector value after threshold")
    axes[1].set_title("Thresholded folded ZLP spectra")
    axes[1].set_xlabel(f"Physical ZLP detector column modulo {args.zlp_period}")
    axes[1].set_ylabel("Mean corrected detector value after threshold")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(comparison_dir / "comparison_thresholded_spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for item, color in zip(current_results, colors):
        result = item["result"]
        spectrum = combined_mean(result["thresholded_sums"][recommended], result["valid_counts"], np)
        folded, _, _ = fold_zlp(
            result["thresholded_sums"][recommended].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            args.zlp_width,
            args.zlp_period,
            np,
        )
        axes[0].plot(np.ma.masked_less_equal(spectrum, 0.0), color=color, linewidth=0.75, label=item["current_label"])
        axes[1].plot(np.ma.masked_less_equal(folded, 0.0), color=color, linewidth=0.9, label=item["current_label"])
    axes[0].axvline(args.zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"ImageJ-style {args.recommended_multiplier:g}-sigma thresholded spectra, log scale")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean corrected detector value after threshold")
    axes[1].set_title("Thresholded folded ZLP spectra, log scale")
    axes[1].set_xlabel(f"Physical ZLP detector column modulo {args.zlp_period}")
    axes[1].set_ylabel("Mean corrected detector value after threshold")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(comparison_dir / "comparison_thresholded_spectra_log.png", dpi=180)
    plt.close(fig)

    subplot_rows = (len(current_results) + 1) // 2
    fig, axes = plt.subplots(
        subplot_rows,
        2,
        figsize=(17, 4.2 * subplot_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, item in zip(axes.flat, current_results):
        result = item["result"]
        before = combined_mean(result["unthresholded_sums"], result["valid_counts"], np)
        after = combined_mean(result["thresholded_sums"][recommended], result["valid_counts"], np)
        axis.plot(np.ma.masked_less_equal(before, 0.0), color="0.25", linewidth=0.8, label="before threshold")
        axis.plot(np.ma.masked_less_equal(after, 0.0), color="tab:blue", linewidth=0.8, label="after threshold")
        axis.axvline(args.zlp_width, color="tab:red", linestyle="--", linewidth=0.7)
        axis.set_yscale("log")
        axis.set_title(item["current_label"])
        axis.set_xlabel("Detector output column")
        axis.set_ylabel("Mean corrected detector value")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(current_results):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Spectra before and after ImageJ-style {args.recommended_multiplier:g}-sigma threshold",
        fontsize=14,
    )
    fig.savefig(comparison_dir / "comparison_before_after_spectra_log.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        subplot_rows,
        2,
        figsize=(17, 4.2 * subplot_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, item in zip(axes.flat, current_results):
        result = item["result"]
        before = combined_mean(result["unthresholded_sums"], result["valid_counts"], np)
        after = combined_mean(result["thresholded_sums"][recommended], result["valid_counts"], np)
        columns = np.arange(args.zlp_width, args.width)
        axis.plot(columns, before[args.zlp_width:], color="0.25", linewidth=0.8, label="before threshold")
        axis.plot(columns, after[args.zlp_width:], color="tab:blue", linewidth=0.8, label="after threshold")
        axis.set_title(item["current_label"])
        axis.set_xlabel("Detector output column")
        axis.set_ylabel("Mean corrected detector value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(current_results):]:
        axis.set_visible(False)
    fig.suptitle("CoreLoss spectra before and after threshold", fontsize=14)
    fig.savefig(comparison_dir / "comparison_before_after_coreloss.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        subplot_rows,
        2,
        figsize=(17, 4.2 * subplot_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, item in zip(axes.flat, current_results):
        result = item["result"]
        before, _, _ = fold_zlp(
            result["unthresholded_sums"].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            args.zlp_width,
            args.zlp_period,
            np,
        )
        after, _, _ = fold_zlp(
            result["thresholded_sums"][recommended].sum(axis=0),
            result["valid_counts"].sum(axis=0),
            args.zlp_width,
            args.zlp_period,
            np,
        )
        axis.plot(before, color="0.25", linewidth=0.9, label="before threshold")
        axis.plot(after, color="tab:blue", linewidth=0.9, label="after threshold")
        axis.set_title(item["current_label"])
        axis.set_xlabel(f"Physical ZLP detector column modulo {args.zlp_period}")
        axis.set_ylabel("Mean corrected detector value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(current_results):]:
        axis.set_visible(False)
    fig.suptitle("Folded ZLP spectra before and after threshold", fontsize=14)
    fig.savefig(comparison_dir / "comparison_before_after_zlp_folded.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        subplot_rows,
        2,
        figsize=(17, 4.2 * subplot_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, item in zip(axes.flat, current_results):
        result = item["result"]
        before = combined_mean(result["unthresholded_sums"], result["valid_counts"], np)
        after = combined_mean(result["thresholded_sums"][recommended], result["valid_counts"], np)
        meaningful = np.zeros(before.shape, dtype=bool)
        for start, stop in ((0, args.zlp_width), (args.zlp_width, args.width)):
            region = before[start:stop]
            region_peak = np.nanmax(region)
            meaningful[start:stop] = (
                np.isfinite(region)
                & (region > 0.0)
                & (region > region_peak * 0.01)
            )
        ratio = np.full(before.shape, np.nan, dtype=np.float64)
        ratio[meaningful] = after[meaningful] / before[meaningful]
        axis.plot(ratio, color="tab:green", linewidth=0.75)
        axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.7)
        axis.axvline(args.zlp_width, color="tab:red", linestyle="--", linewidth=0.7)
        axis.set_title(item["current_label"])
        axis.set_xlabel("Detector output column")
        axis.set_ylabel("After / before")
        axis.set_ylim(0.0, 1.2)
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(current_results):]:
        axis.set_visible(False)
    fig.suptitle(
        "Threshold-retained spectral signal (shown above 1% of each region's pre-threshold peak)",
        fontsize=14,
    )
    fig.savefig(comparison_dir / "comparison_threshold_retained_ratio.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    currents = np.asarray([item["current_pa"] for item in current_results], dtype=float)
    labels = [item["current_label"] for item in current_results]
    zlp_thresholds = [item["noise"]["zlp"]["stddev"] * args.recommended_multiplier for item in current_results]
    core_thresholds = [item["noise"]["core"]["stddev"] * args.recommended_multiplier for item in current_results]
    axes[0].semilogx(currents, zlp_thresholds, marker="o", label="ZLP")
    axes[0].semilogx(currents, core_thresholds, marker="o", label="CoreLoss")
    axes[0].set_title("Recommended ImageJ-style thresholds from matching dark residuals")
    axes[0].set_ylabel("Threshold value")
    axes[0].legend()

    for metric_name, label in (
        ("zlp_positive_area_retained_vs_unthresholded", "ZLP positive area"),
        ("core_early_retained_vs_unthresholded", "early CoreLoss mean"),
        ("zlp_sample_retained_fraction", "ZLP sample retention"),
        ("core_sample_retained_fraction", "Core sample retention"),
    ):
        values = []
        for item in current_results:
            values.append(item["metrics"][recommended][metric_name])
        axes[1].semilogx(currents, values, marker="o", label=label)
    axes[1].set_title(f"Effect of {args.recommended_multiplier:g}-sigma threshold")
    axes[1].set_ylabel("Fraction / ratio")
    axes[1].legend()

    for item in current_results:
        axes[2].plot(
            [row["multiplier"] for row in item["metrics"]],
            [row["core_sample_retained_fraction"] for row in item["metrics"]],
            marker="o",
            linewidth=0.8,
            label=item["current_label"],
        )
    axes[2].axvline(args.recommended_multiplier, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("CoreLoss sample retention over threshold sweep")
    axes[2].set_xlabel("Threshold multiplier (sigma)")
    axes[2].set_ylabel("Retained sample fraction")
    axes[2].legend(ncol=4, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlabel(axis.get_xlabel() or "Nominal beam current (pA)")
    fig.savefig(comparison_dir / "threshold_selection_summary.png", dpi=180)
    plt.close(fig)

    report = [
        "# NiO Threshold Study",
        "",
        "This study applies a positive threshold after dark subtraction, grouped BLR,",
        "valid-pixel masking, and dynamic half-column masking, but before summing into",
        "column spectra.",
        "",
        "The reference policy matches `Threshold_for_CountingSumming.ijm`: compute",
        "separate dark-subtracted dark residual standard deviations in the ZLP and",
        f"CoreLoss regions, then use `{args.recommended_multiplier:g} * stddev`.",
        "",
        "Generated comparison plots:",
        "",
        "- `comparison_thresholded_spectra.png`",
        "- `comparison_thresholded_spectra_log.png`",
        "- `comparison_before_after_spectra_log.png`",
        "- `comparison_before_after_coreloss.png`",
        "- `comparison_before_after_zlp_folded.png`",
        "- `comparison_threshold_retained_ratio.png`",
        "- `threshold_selection_summary.png`",
        "- `threshold_metrics_all_currents.csv`",
        "",
        "Per-current folders contain threshold sweeps and HDF5 spectra.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def current_sort_key(key: str, group: dict) -> tuple[int, str]:
    return int(group["current_pa"]), key


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--currents", nargs="*", default=None)
    parser.add_argument("--multipliers", default="0,1,2,3,4,5")
    parser.add_argument("--recommended-multiplier", type=float, default=3.0)
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask")
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
    parser.add_argument("--noise-clip-abs", type=float, default=1000.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate comparison plots from existing per-current HDF5 and CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multipliers = parse_multipliers(args.multipliers)

    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = json.loads((args.source_study_root / "manifest.json").read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "study_config.json").write_text(
        json.dumps({**vars(args), "multipliers": multipliers}, default=str, indent=2),
        encoding="utf-8",
    )

    selected = set(args.currents) if args.currents else None
    current_results = []
    for key, group in sorted(manifest["currents"].items(), key=lambda item: current_sort_key(*item)):
        if selected is not None and key not in selected and group["current_label"] not in selected:
            continue
        current_output = args.output_root / "currents" / key
        h5_path = current_output / "threshold_spectra.h5"
        metrics_path = current_output / "threshold_metrics.csv"
        dark_path = args.source_study_root / "currents" / key / "dark" / "dark_frame.h5"
        dark_files = [Path(item["path"]) for item in group["dark_files"]]
        spectrum_files = [Path(item["path"]) for item in group["spectrum_files"]]

        print(f"Processing {group['current_label']} ({key})", flush=True)
        if args.plots_only:
            if not h5_path.exists() or not metrics_path.exists():
                raise FileNotFoundError(
                    f"--plots-only requires existing outputs in {current_output}"
                )
            noise_path = current_output / "threshold_noise_summary.json"
            noise = json.loads(noise_path.read_text(encoding="utf-8"))
            with h5py.File(h5_path, "r") as h5:
                result = {name: h5[name][...] for name in h5.keys()}
            with metrics_path.open("r", newline="", encoding="utf-8") as stream:
                metrics = [
                    {key_: float(value) for key_, value in row.items()}
                    for row in csv.DictReader(stream)
                ]
        else:
            dark_frame, valid_mask = load_dark_assets(dark_path, args, h5py, np)
            noise = compute_imagej_dark_noise(dark_files, dark_frame, args, np)

        if not args.plots_only and (args.force or not h5_path.exists() or not metrics_path.exists()):
            result = accumulate_thresholded_spectra(
                spectrum_files,
                dark_frame,
                valid_mask,
                multipliers,
                noise["zlp"]["stddev"],
                noise["core"]["stddev"],
                args,
                np,
            )
            metrics = save_current_outputs(
                current_output,
                result,
                noise,
                spectrum_files,
                dark_files,
                dark_path,
                args,
                h5py,
                np,
                plt,
            )
        elif not args.plots_only:
            with h5py.File(h5_path, "r") as h5:
                result = {name: h5[name][...] for name in h5.keys()}
            with metrics_path.open("r", newline="", encoding="utf-8") as stream:
                metrics = [
                    {key_: float(value) for key_, value in row.items()}
                    for row in csv.DictReader(stream)
                ]

        current_results.append({
            "current_key": key,
            "current_label": group["current_label"],
            "current_pa": int(group["current_pa"]),
            "noise": noise,
            "metrics": metrics,
            "result": result,
        })

    if not current_results:
        raise ValueError("no currents were processed")
    save_comparison_outputs(args.output_root, current_results, args, np, plt)
    summary = {
        "completed_currents": [item["current_key"] for item in current_results],
        "recommended_multiplier": args.recommended_multiplier,
        "comparison_dir": str(args.output_root / "comparisons"),
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
