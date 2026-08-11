#!/usr/bin/env python3
"""Run standard STEMPy electron counting on corrected NiO spectrum frames.

The script deliberately keeps electron counting separate from the analog ZLP
analysis. It first applies the CPU/NumPy mirror of the Holoscan correction chain,
then uses STEMPy's standard threshold-plus-local-maximum counter in the CoreLoss
region. Dark frames determine the Gaussian background width and false-count
rate. Measured event occupancy determines where one-local-maximum-per-electron
is a defensible approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from stem_analysis import ProcessorConfig, process_tensor_block
from stem_analysis.dm4 import load_dm4, normalize_to_frame_stack


CURRENT_LABELS = {
    "0015pA": "15 pA",
    "0030pA": "30 pA",
    "0060pA": "60 pA",
    "0130pA": "130 pA",
    "0250pA": "250 pA",
    "0500pA": "500 pA",
    "1000pA": "1 nA",
}

RAW_ZLP_WIDTH = 768
FOLDED_ZLP_WIDTH = 192
RAW_TO_STITCHED_CORE_OFFSET = RAW_ZLP_WIDTH - FOLDED_ZLP_WIDTH
DEAD_ADC_RAW = (2272, 2288)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-key", default="0015pA", choices=tuple(CURRENT_LABELS))
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument(
        "--blr-mode",
        choices=("grouped", "columnwise"),
        default="columnwise",
        help=(
            "BLR grouping used before nonlinear local-maximum counting. Columnwise is the "
            "recommended counting mode because grouped offsets can suppress whole columns."
        ),
    )
    parser.add_argument(
        "--counting-common-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Subtract the per-frame, per-column imaging-row median independently in each "
            "detector half before counting. Recommended for sparse event counting."
        ),
    )
    parser.add_argument(
        "--max-count-batches-per-file",
        type=int,
        default=None,
        help="Optional diagnostic limit for the final counting pass; omit for all frames.",
    )
    parser.add_argument(
        "--dynamic-mask-window",
        type=int,
        default=31,
        help="Odd same-column median window used by the batch-level dynamic mask.",
    )
    parser.add_argument(
        "--candidate-sigmas", type=float, nargs="+", default=(4, 5, 6, 8, 10)
    )
    parser.add_argument(
        "--maximum-dark-fraction",
        type=float,
        default=0.01,
        help="Select the lowest sigma cut whose scaled dark count is at most this fraction.",
    )
    parser.add_argument("--xray-sigma", type=float, default=175.0)
    parser.add_argument("--quantization-scale", type=float, default=0.5)
    parser.add_argument("--quantization-offset", type=float, default=16384.0)
    parser.add_argument("--occupancy-neighborhood-pixels", type=int, default=9)
    parser.add_argument("--maximum-conditional-pileup", type=float, default=0.05)
    parser.add_argument("--occupancy-bin-columns", type=int, default=64)
    parser.add_argument("--noise-bin-columns", type=int, default=16)
    parser.add_argument("--first-loss-energy-ev", type=float, default=25.0)
    parser.add_argument("--tail-energy-ev", type=float, default=350.0)
    parser.add_argument(
        "--spectrum-calibration-batches-per-file",
        type=int,
        default=1,
        help="Number of corrected batches per spectrum file used for threshold selection.",
    )
    return parser.parse_args()


def load_manifest_group(study_root: Path, current_key: str):
    manifest_path = study_root / "nio_beam_current" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group = manifest["currents"].get(current_key)
    if group is None:
        raise ValueError(f"current {current_key} is absent from {manifest_path}")
    dark_paths = [Path(item["path"]) for item in group["dark_files"]]
    spectrum_paths = [Path(item["path"]) for item in group["spectrum_files"]]
    if not dark_paths or not spectrum_paths:
        raise ValueError(f"current {current_key} requires dark and spectrum files")
    return dark_paths, spectrum_paths


def load_analog_reference(study_root: Path, current_key: str, np):
    label = CURRENT_LABELS[current_key]
    path = (
        study_root
        / "nio_processing_chain_deck"
        / "final_spectra_all_adopted_corrections.csv"
    )
    columns = []
    values = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            columns.append(int(row["stitched_column"]))
            values.append(float(row[label]))
    return np.asarray(columns), np.asarray(values, dtype=np.float64), path


def energy_calibration(profile, first_loss_energy_ev: float, np, scipy_signal):
    zlp_peak = int(np.nanargmax(profile[:FOLDED_ZLP_WIDTH]))
    smoothed = scipy_signal.savgol_filter(profile[:FOLDED_ZLP_WIDTH], 21, 3)
    search_start = max(zlp_peak + 60, 100)
    first_loss = search_start + int(np.nanargmax(smoothed[search_start:190]))
    dispersion = first_loss_energy_ev / (first_loss - zlp_peak)
    return zlp_peak, first_loss, float(dispersion)


def processor_config(blr_mode: str, dynamic_mask_window: int) -> ProcessorConfig:
    zlp_group_columns = 1 if blr_mode == "columnwise" else 4
    core_group_columns = 1 if blr_mode == "columnwise" else 16
    return ProcessorConfig(
        noop=True,
        subtract_dark_frame=True,
        apply_valid_pixel_mask=True,
        apply_blr_correction=True,
        blr_rows=30,
        blr_zlp_width=768,
        blr_zlp_group_columns=zlp_group_columns,
        blr_core_group_columns=core_group_columns,
        apply_dynamic_half_column_mask=True,
        dynamic_mask_median_window_pixels=dynamic_mask_window,
        dynamic_mask_threshold_ratio=1.0,
        dynamic_mask_threshold_offset=500.0,
        dynamic_mask_excluded_edge_rows=32,
        dynamic_mask_two_sided=True,
    )


def load_dark_products(path: Path, np):
    import h5py

    with h5py.File(path, "r") as source:
        dark = source["processed"][0].astype(np.float32)
        valid = source["valid_pixel_mask"][0].astype(np.float32)
    return dark, valid


def compute_array(array, np):
    return np.asarray(array.compute() if hasattr(array, "compute") else array)


def corrected_batches(paths,
                      dark_frame,
                      valid_mask,
                      config,
                      tensor_frames: int,
                      reader_name: str,
                      np,
                      counting_common_mode: bool = False,
                      max_batches_per_file: int | None = None):
    """Yield fully corrected imaging-row CoreLoss batches and source metadata."""
    for source_index, path in enumerate(paths):
        data, _ = load_dm4(path, reader_name)
        stack = normalize_to_frame_stack(data, height=960, width=3840)
        batch_count = 0
        for start in range(0, int(stack.shape[0]), tensor_frames):
            if max_batches_per_file is not None and batch_count >= max_batches_per_file:
                break
            end = min(start + tensor_frames, int(stack.shape[0]))
            raw = compute_array(stack[start:end], np)
            corrected, zero_mask = process_tensor_block(
                raw, dark_frame, valid_mask, config, np
            )
            del raw
            imaging = np.concatenate(
                (corrected[:, 32:480, :], corrected[:, 480:928, :]), axis=1
            )
            core = np.ascontiguousarray(imaging[:, :, RAW_ZLP_WIDTH:])
            del corrected, imaging
            if counting_common_mode:
                subtract_counting_common_mode(core, np)
            yield core, {
                "source_index": source_index,
                "source_path": str(path),
                "start_frame": start,
                "end_frame": end,
                "frames": end - start,
                "masked_pixels": int(np.count_nonzero(zero_mask)),
            }
            batch_count += 1
        del stack, data


def subtract_counting_common_mode(core, np) -> None:
    """Remove robust column offsets while preserving sparse events in-place."""
    half_height = core.shape[1] // 2
    for frame in core:
        frame[:half_height] -= np.median(frame[:half_height], axis=0)[None, :]
        frame[half_height:] -= np.median(frame[half_height:], axis=0)[None, :]


def quantize_frames(frames, scale: float, offset: float, np):
    transformed = frames * scale + offset
    low = int(np.count_nonzero(transformed < 0.0))
    high = int(np.count_nonzero(transformed > 65535.0))
    quantized = np.rint(np.clip(transformed, 0.0, 65535.0)).astype(np.uint16)
    return quantized, low, high


def write_stempy_frames(path: Path, frames, np):
    import h5py

    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("frames", data=frames, chunks=(1,) + frames.shape[1:])
        dataset.attrs["scan_dimensions"] = np.asarray((frames.shape[0], 1), dtype=np.int64)


def calibrate_stempy_noise(path: Path, frame_shape, samples: int, xray_sigma: float, np):
    import h5py
    import stempy.image as stim
    import stempy.io as stio

    with h5py.File(path, "r") as source:
        reader = stio.reader(source)
        block = next(reader)
        result = stim._image.calculate_thresholds(
            [block._block],
            np.zeros(frame_shape, dtype=np.float32),
            samples,
            4.0,
            xray_sigma,
        )
    return {
        "optimized_mean": float(result.optimized_mean),
        "optimized_stddev": float(result.optimized_std_dev),
        "sample_mean": float(result.mean),
        "sample_stddev": float(result.std_dev),
        "sample_min": float(result.min_sample),
        "sample_max": float(result.max_sample),
    }


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


def candidate_event_counts(frames, thresholds, xray_threshold: float, np):
    counts = np.zeros((len(thresholds), frames.shape[2]), dtype=np.uint64)
    for frame in frames:
        local_maximum = frame > neighboring_maximum(frame, np)
        below_xray = frame < xray_threshold
        for index, threshold in enumerate(thresholds):
            columns = np.nonzero(local_maximum & below_xray & (frame > threshold))[1]
            counts[index] += np.bincount(columns, minlength=frames.shape[2]).astype(np.uint64)
    return counts


def copy_stempy_event_indices(sparse_frame, np):
    """Copy one STEMPy event vector, including with NumPy 2 zero-stride bindings."""
    view = np.asarray(sparse_frame)
    if view.ndim != 1 or view.dtype != np.uint32:
        raise ValueError(
            f"unexpected STEMPy event vector shape/dtype: {view.shape} {view.dtype}"
        )
    if view.size == 0:
        return np.empty(0, dtype=np.uint32)
    if view.strides == (view.dtype.itemsize,):
        return view.copy()
    if view.strides != (0,):
        raise ValueError(f"unexpected STEMPy event-vector strides: {view.strides}")

    # STEMPy 3.4.2's pybind vector wrapper exposes a zero stride with NumPy 2,
    # although the owned C++ vector is contiguous. Read that allocation directly.
    import ctypes

    pointer = view.__array_interface__["data"][0]
    buffer = (ctypes.c_uint32 * view.size).from_address(pointer)
    return np.ctypeslib.as_array(buffer).copy()


def stempy_count(path: Path,
                 frame_shape,
                 frame_count: int,
                 background_threshold: float,
                 xray_threshold: float,
                 np):
    """Count with STEMPy's standard strict 8-neighbor local-maximum method."""
    import h5py
    import stempy.image as stim
    import stempy.io as stio

    with h5py.File(path, "r") as source:
        reader = stio.reader(source)
        options = stim._image.ElectronCountOptionsClassic()
        # Supplying a zero reference avoids a null-reference bug in the classic HDF5 path.
        options.dark_reference = np.zeros(frame_shape, dtype=np.float32)
        options.background_threshold = background_threshold
        options.x_ray_threshold = xray_threshold
        options.scan_dimensions = (frame_count, 1)
        options.apply_row_dark_subtraction = False
        options.optimized_mean = 0.0
        options.apply_row_dark_use_mean = True
        counted = stim._image.electron_count(reader.begin(), reader.end(), options)

    per_column = np.zeros(frame_shape[1], dtype=np.uint64)
    per_half = np.zeros((2, frame_shape[1]), dtype=np.uint64)
    per_frame_columns = []
    event_count = 0
    half_height = frame_shape[0] // 2
    for scan_frames in counted.data:
        indices = copy_stempy_event_indices(scan_frames[0], np)
        columns = indices % frame_shape[1]
        rows = indices // frame_shape[1]
        frame_columns = np.bincount(columns, minlength=frame_shape[1]).astype(np.uint16)
        per_frame_columns.append(frame_columns)
        per_column += frame_columns
        for half in (0, 1):
            selected = columns[(rows >= half * half_height) & (rows < (half + 1) * half_height)]
            per_half[half] += np.bincount(
                selected, minlength=frame_shape[1]
            ).astype(np.uint64)
        event_count += int(indices.size)
    return per_column, per_half, event_count, np.stack(per_frame_columns)


def conditional_pileup(lam, np):
    at_least_one = 1.0 - np.exp(-lam)
    at_least_two = 1.0 - np.exp(-lam) * (1.0 + lam)
    return np.divide(
        at_least_two,
        at_least_one,
        out=np.zeros_like(lam, dtype=np.float64),
        where=at_least_one > 0,
    )


def occupancy_summary(counts,
                      frames: int,
                      rows: int,
                      bin_columns: int,
                      neighborhood_pixels: int,
                      maximum_pileup: float,
                      np):
    width = counts.size
    starts = np.arange(0, width, bin_columns)
    stops = np.minimum(starts + bin_columns, width)
    events = np.asarray([counts[start:stop].sum() for start, stop in zip(starts, stops)])
    pixels = frames * rows * (stops - starts)
    event_rate = events / pixels
    lam = event_rate * neighborhood_pixels
    pileup = conditional_pileup(lam, np)

    raw_starts = starts + RAW_ZLP_WIDTH
    raw_stops = stops + RAW_ZLP_WIDTH
    dead = (raw_starts < DEAD_ADC_RAW[1]) & (raw_stops > DEAD_ADC_RAW[0])
    good = (pileup <= maximum_pileup) | dead
    first_valid_bin = None
    for index in range(len(good)):
        if np.all(good[index:]):
            first_valid_bin = index
            break
    if first_valid_bin is None:
        first_valid_bin = len(starts) - 1
    boundary_raw = int(raw_starts[first_valid_bin])
    return {
        "starts": starts,
        "stops": stops,
        "centers_raw": (raw_starts + raw_stops - 1) / 2.0,
        "events": events,
        "event_rate": event_rate,
        "lambda_neighborhood": lam,
        "conditional_pileup": pileup,
        "dead_adc_bin": dead,
        "boundary_raw": boundary_raw,
        "boundary_stitched": boundary_raw - RAW_TO_STITCHED_CORE_OFFSET,
    }


def binned_noise_statistics(per_batch,
                            raw_start: int,
                            raw_stop: int,
                            bin_columns: int,
                            np):
    starts = np.arange(raw_start, raw_stop, bin_columns)
    stops = np.minimum(starts + bin_columns, raw_stop)
    core_starts = starts - RAW_ZLP_WIDTH
    core_stops = stops - RAW_ZLP_WIDTH
    values = np.stack(
        [per_batch[:, start:stop].sum(axis=1) for start, stop in zip(core_starts, core_stops)],
        axis=1,
    ).astype(np.float64)

    mean = values.mean(axis=0)
    stddev = values.std(axis=0, ddof=1)
    total = values.sum(axis=0)
    empirical_relative_sem = np.divide(
        stddev / np.sqrt(values.shape[0]),
        mean,
        out=np.full_like(mean, np.nan),
        where=mean > 0,
    )
    poisson_relative = np.divide(
        1.0,
        np.sqrt(total),
        out=np.full_like(total, np.nan),
        where=total > 0,
    )
    return {
        "starts_raw": starts,
        "stops_raw": stops,
        "centers_stitched": (starts + stops - 1) / 2.0 - RAW_TO_STITCHED_CORE_OFFSET,
        "mean_per_batch": mean,
        "stddev_per_batch": stddev,
        "total_counts": total,
        "empirical_relative_sem": empirical_relative_sem,
        "poisson_relative": poisson_relative,
    }


def write_outputs(args,
                  summary,
                  threshold_rows,
                  occupancy,
                  total_counts,
                  per_batch_counts,
                  batch_metadata,
                  per_frame_counts,
                  per_frame_metadata,
                  noise,
                  analog_profile,
                  np):
    import h5py

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "counting_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (args.output_dir / "threshold_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)

    raw_columns = np.arange(RAW_ZLP_WIDTH, 3840)
    stitched_columns = raw_columns - RAW_TO_STITCHED_CORE_OFFSET
    with (args.output_dir / "counted_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("raw_column", "stitched_column", "electron_count"))
        writer.writerows(zip(raw_columns, stitched_columns, total_counts))

    with h5py.File(args.output_dir / "counted_spectrum.h5", "w") as output:
        output.create_dataset("total_core_counts", data=total_counts, compression="gzip")
        output.create_dataset("per_batch_core_counts", data=per_batch_counts, compression="gzip")
        output.create_dataset(
            "per_frame_core_counts",
            data=per_frame_counts,
            chunks=(1, per_frame_counts.shape[1]),
            compression="gzip",
        )
        output.create_dataset("raw_core_columns", data=raw_columns)
        output.create_dataset("stitched_core_columns", data=stitched_columns)
        output.create_dataset(
            "per_batch_source_index",
            data=np.asarray([item["source_index"] for item in batch_metadata]),
        )
        output.create_dataset(
            "per_batch_start_frame",
            data=np.asarray([item["start_frame"] for item in batch_metadata]),
        )
        output.create_dataset(
            "per_frame_source_index",
            data=np.asarray([item["source_index"] for item in per_frame_metadata]),
        )
        output.create_dataset(
            "per_frame_source_frame",
            data=np.asarray([item["source_frame"] for item in per_frame_metadata]),
        )
        for key, value in summary["selected_threshold"].items():
            if isinstance(value, (int, float, str, bool)):
                output.attrs[key] = value

    tail_core = summary["energy_calibration"]["tail_raw_channel"] - RAW_ZLP_WIDTH
    frame_total = per_frame_counts.sum(axis=1)
    frame_tail = per_frame_counts[:, tail_core:].sum(axis=1)
    frame_peak_channel = per_frame_counts.max(axis=1)
    with (args.output_dir / "per_frame_event_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "global_frame", "source_index", "source_frame", "coreloss_events",
            "tail_events", "maximum_events_in_one_channel",
        ))
        for index, metadata in enumerate(per_frame_metadata):
            writer.writerow((
                index, metadata["source_index"], metadata["source_frame"],
                int(frame_total[index]), int(frame_tail[index]),
                int(frame_peak_channel[index]),
            ))

    with (args.output_dir / "tail_noise_binned.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "raw_start", "raw_stop", "stitched_center", "total_counts",
            "empirical_relative_sem", "poisson_relative_1_over_sqrt_N",
        ))
        writer.writerows(zip(
            noise["starts_raw"], noise["stops_raw"], noise["centers_stitched"],
            noise["total_counts"], noise["empirical_relative_sem"],
            noise["poisson_relative"],
        ))


def make_plots(args,
               summary,
               threshold_rows,
               occupancy,
               total_counts,
               per_frame_counts,
               per_frame_metadata,
               noise,
               analog_profile,
               np,
               plt):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    })

    sigmas = np.asarray([row["sigma_multiplier"] for row in threshold_rows])
    dark_rate = np.asarray([row["dark_events_per_frame_tail"] for row in threshold_rows])
    spectrum_rate = np.asarray([row["spectrum_events_per_frame_tail"] for row in threshold_rows])
    fraction = np.asarray([row["estimated_dark_fraction_tail"] for row in threshold_rows])
    selected_sigma = summary["selected_threshold"]["sigma_multiplier"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].semilogy(sigmas, np.maximum(dark_rate, 1e-6), "o-", label="processed dark")
    axes[0].semilogy(sigmas, np.maximum(spectrum_rate, 1e-6), "o-", label="spectrum sample")
    axes[0].axvline(selected_sigma, color="#B45309", linestyle="--", label="selected")
    axes[0].set(xlabel="background threshold [Gaussian σ]",
                ylabel="local maxima per frame (>350 eV)",
                title="Threshold calibration")
    axes[0].legend()
    axes[1].semilogy(sigmas, np.maximum(fraction, 1e-8), "o-", color="#0F766E")
    axes[1].axhline(args.maximum_dark_fraction, color="#B45309", linestyle="--",
                    label=f"target {args.maximum_dark_fraction:.1%}")
    axes[1].axvline(selected_sigma, color="#B45309", linestyle="--")
    axes[1].set(xlabel="background threshold [Gaussian σ]",
                ylabel="scaled dark / spectrum event rate",
                title="Estimated false-count fraction")
    axes[1].legend()
    fig.savefig(args.output_dir / "threshold_calibration.png", dpi=180)
    plt.close(fig)

    centers_stitched = occupancy["centers_raw"] - RAW_TO_STITCHED_CORE_OFFSET
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(centers_stitched, occupancy["conditional_pileup"], "o-", markersize=3)
    ax.axhline(args.maximum_conditional_pileup, color="#B45309", linestyle="--",
               label=f"validity target {args.maximum_conditional_pileup:.0%}")
    ax.axvline(occupancy["boundary_stitched"], color="#0F766E", linestyle="--",
               label=f"selected boundary {occupancy['boundary_stitched']}")
    ax.axvspan(DEAD_ADC_RAW[0] - RAW_TO_STITCHED_CORE_OFFSET,
               DEAD_ADC_RAW[1] - RAW_TO_STITCHED_CORE_OFFSET,
               color="#DC2626", alpha=0.12, label="dead ADC block")
    ax.set(xlabel="stitched spectral channel", ylabel="P(≥2 hits | ≥1) in 3×3 area",
           title="Measured occupancy criterion for standard counting")
    ax.legend()
    fig.savefig(args.output_dir / "counting_validity.png", dpi=180)
    plt.close(fig)

    stitched = np.arange(RAW_ZLP_WIDTH, 3840) - RAW_TO_STITCHED_CORE_OFFSET
    counted_frames = int(summary["counting"]["frames"])
    counts_plot = total_counts.astype(np.float64) / counted_frames
    counts_plot[:max(0, occupancy["boundary_raw"] - RAW_ZLP_WIDTH)] = np.nan
    dead_start = DEAD_ADC_RAW[0] - RAW_ZLP_WIDTH
    dead_stop = DEAD_ADC_RAW[1] - RAW_ZLP_WIDTH
    counts_plot[dead_start:dead_stop] = np.nan
    zlp_peak = summary["energy_calibration"]["zlp_peak_channel"]
    dispersion = summary["energy_calibration"]["dispersion_ev_per_channel"]
    energy = (stitched - zlp_peak) * dispersion

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    axes[0].plot(stitched, np.ma.masked_less_equal(analog_profile[stitched], 0),
                 color="#6B7280", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].axvline(occupancy["boundary_stitched"], color="#0F766E", linestyle="--")
    axes[0].set(ylabel="analog corrected intensity", title="Analog reference")
    axes[1].plot(stitched, np.ma.masked_invalid(counts_plot), color="#0F766E", linewidth=0.9)
    axes[1].set_yscale("log")
    axes[1].axvline(occupancy["boundary_stitched"], color="#0F766E", linestyle="--",
                    label="counting-valid boundary")
    axes[1].axvspan(DEAD_ADC_RAW[0] - RAW_TO_STITCHED_CORE_OFFSET,
                    DEAD_ADC_RAW[1] - RAW_TO_STITCHED_CORE_OFFSET,
                    color="#DC2626", alpha=0.12, label="excluded dead ADC block")
    axes[1].set(xlabel="stitched spectral channel", ylabel="counted electrons / frame",
                title="STEMPy-counted spectrum (all frames)")
    axes[1].legend()
    secondary = axes[1].secondary_xaxis(
        "top",
        functions=(lambda x: (x - zlp_peak) * dispersion,
                   lambda e: e / dispersion + zlp_peak),
    )
    secondary.set_xlabel("provisional energy loss [eV]")
    fig.savefig(args.output_dir / "counted_spectrum_full.png", dpi=180)
    plt.close(fig)

    full_stitched = np.arange(analog_profile.size)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True)
    axes[0].plot(
        full_stitched,
        np.ma.masked_less_equal(analog_profile, 0),
        color="#6B7280",
        linewidth=0.8,
    )
    axes[0].set_yscale("log")
    axes[0].axvspan(
        0,
        RAW_ZLP_WIDTH - RAW_TO_STITCHED_CORE_OFFSET,
        color="#F59E0B",
        alpha=0.12,
        label="folded ZLP",
    )
    axes[0].axvline(
        RAW_ZLP_WIDTH - RAW_TO_STITCHED_CORE_OFFSET,
        color="#B45309",
        linestyle=":",
        linewidth=1.0,
        label="ZLP/CoreLoss boundary",
    )
    axes[0].set(
        ylabel="analog corrected intensity",
        title="Final analog spectrum (per-frame regional 3σ threshold; ZLP included)",
    )
    axes[0].legend()
    axes[1].plot(stitched, np.ma.masked_invalid(counts_plot), color="#0F766E", linewidth=0.9)
    axes[1].set_yscale("log")
    axes[1].axvspan(
        0,
        RAW_ZLP_WIDTH - RAW_TO_STITCHED_CORE_OFFSET,
        facecolor="#E5E7EB",
        edgecolor="#9CA3AF",
        alpha=0.5,
        hatch="//",
        label="ZLP not electron-counted",
    )
    axes[1].axvline(occupancy["boundary_stitched"], color="#0F766E", linestyle="--",
                    label="counting-valid boundary")
    axes[1].axvspan(DEAD_ADC_RAW[0] - RAW_TO_STITCHED_CORE_OFFSET,
                    DEAD_ADC_RAW[1] - RAW_TO_STITCHED_CORE_OFFSET,
                    color="#DC2626", alpha=0.12, label="excluded dead ADC block")
    axes[1].set(
        xlabel="stitched spectral channel",
        ylabel="counted electrons / frame",
        title="Mean STEMPy-counted CoreLoss spectrum (ZLP intentionally omitted)",
    )
    axes[1].set_xlim(0, analog_profile.size - 1)
    axes[1].legend()
    secondary = axes[1].secondary_xaxis(
        "top",
        functions=(lambda x: (x - zlp_peak) * dispersion,
                   lambda e: e / dispersion + zlp_peak),
    )
    secondary.set_xlabel("provisional energy loss [eV]")
    fig.savefig(args.output_dir / "counted_spectrum_full_analog_zlp.png", dpi=180)
    plt.close(fig)

    centers = noise["centers_stitched"]
    energies = (centers - zlp_peak) * dispersion
    dead = (
        (noise["starts_raw"] < DEAD_ADC_RAW[1])
        & (noise["stops_raw"] > DEAD_ADC_RAW[0])
    )
    tail_counts = noise["total_counts"].astype(np.float64) / counted_frames
    empirical = noise["empirical_relative_sem"].copy()
    poisson = noise["poisson_relative"].copy()
    tail_counts[dead] = np.nan
    empirical[dead] = np.nan
    poisson[dead] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True)
    axes[0].plot(energies, tail_counts, "o-", markersize=3, color="#0F766E")
    axes[0].set_yscale("log")
    axes[0].set(
        ylabel=f"counted electrons / frame / {args.noise_bin_columns}-channel bin",
        title=f"{CURRENT_LABELS[args.current_key]} mean counted tail",
    )
    axes[1].plot(energies, empirical, "o-", markersize=3,
                 label="empirical relative SEM", color="#2563EB")
    axes[1].plot(energies, poisson, "o-", markersize=3,
                 label=r"counting limit $1/\sqrt{N}$", color="#DC2626")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="provisional energy loss [eV]", ylabel="relative uncertainty",
                title="Measured batch fluctuations versus Poisson counting statistics")
    axes[1].legend()
    for ax in axes:
        ax.set_xlim(args.tail_energy_ev, float(np.nanmax(energies)))
    fig.savefig(args.output_dir / "counted_tail_poisson.png", dpi=180)
    plt.close(fig)

    tail_core = summary["energy_calibration"]["tail_raw_channel"] - RAW_ZLP_WIDTH
    frame_total = per_frame_counts.sum(axis=1)
    frame_tail = per_frame_counts[:, tail_core:].sum(axis=1)
    frame_peak_channel = per_frame_counts.max(axis=1)
    frame_index = np.arange(per_frame_counts.shape[0])
    boundaries = np.flatnonzero(np.diff(
        np.asarray([item["source_index"] for item in per_frame_metadata])
    )) + 1

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True, sharex=True)
    axes[0].plot(frame_index, frame_total, linewidth=0.7, color="#0F766E")
    axes[0].set(ylabel="events / frame", title="Counted CoreLoss events by source frame")
    axes[1].plot(frame_index, frame_tail, linewidth=0.7, color="#2563EB")
    axes[1].set(ylabel="tail events / frame",
                title=f"Events at or beyond provisional {args.tail_energy_ev:g} eV")
    axes[2].plot(frame_index, frame_peak_channel, linewidth=0.7, color="#B45309")
    axes[2].set(xlabel="global source-frame index",
                ylabel="events in busiest channel",
                title="Largest single-channel count in each frame")
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="#6B7280", linewidth=0.7, alpha=0.6)
    fig.savefig(args.output_dir / "per_frame_event_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    extent = (
        RAW_ZLP_WIDTH - RAW_TO_STITCHED_CORE_OFFSET,
        3839 - RAW_TO_STITCHED_CORE_OFFSET,
        per_frame_counts.shape[0] - 0.5,
        -0.5,
    )
    image = ax.imshow(
        np.log1p(per_frame_counts), aspect="auto", interpolation="nearest",
        extent=extent, cmap="magma",
    )
    ax.axvline(
        summary["energy_calibration"]["tail_stitched_channel"],
        color="white", linestyle="--", linewidth=0.8,
        label=f"provisional {args.tail_energy_ev:g} eV",
    )
    ax.set(xlabel="stitched spectral channel", ylabel="global source-frame index",
           title="Frame-resolved counted-event density")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log(1 + events per frame/channel)")
    fig.savefig(args.output_dir / "frame_channel_event_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    import stempy
    from scipy import signal

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dark_paths, spectrum_paths = load_manifest_group(args.study_root, args.current_key)
    dark_frame_path = (
        args.study_root
        / "nio_beam_current"
        / "currents"
        / args.current_key
        / "dark"
        / "dark_frame.h5"
    )
    dark_frame, valid_mask = load_dark_products(dark_frame_path, np)
    config = processor_config(args.blr_mode, args.dynamic_mask_window)
    _, analog_profile, analog_path = load_analog_reference(
        args.study_root, args.current_key, np
    )
    zlp_peak, first_loss, dispersion = energy_calibration(
        analog_profile, args.first_loss_energy_ev, np, signal
    )
    tail_stitched = int(round(zlp_peak + args.tail_energy_ev / dispersion))
    tail_raw = tail_stitched + RAW_TO_STITCHED_CORE_OFFSET
    tail_core = max(0, tail_raw - RAW_ZLP_WIDTH)

    print(f"STEMPy {stempy.__version__ if hasattr(stempy, '__version__') else 'installed'}")
    print(f"Dark calibration source: {dark_frame_path}")
    print(f"Provisional {args.tail_energy_ev:g} eV tail begins at raw column {tail_raw}")

    candidate_sigmas = np.asarray(sorted(set(args.candidate_sigmas)), dtype=np.float64)
    calibration = None
    thresholds = None
    xray_threshold = None
    dark_candidate_counts = np.zeros((candidate_sigmas.size, 3072), dtype=np.uint64)
    spectrum_candidate_counts = np.zeros_like(dark_candidate_counts)
    dark_frames = 0
    sample_spectrum_frames = 0
    quantization_clipped_low = 0
    quantization_clipped_high = 0

    with tempfile.TemporaryDirectory(prefix="nio_stempy_count_") as temporary:
        temporary_path = Path(temporary) / "frames.h5"

        print("Processing dark frames for threshold calibration...", flush=True)
        for batch_index, (core, metadata) in enumerate(corrected_batches(
            dark_paths, dark_frame, valid_mask, config, args.tensor_frames,
            args.reader, np, counting_common_mode=args.counting_common_mode
        )):
            quantized, low, high = quantize_frames(
                core, args.quantization_scale, args.quantization_offset, np
            )
            quantization_clipped_low += low
            quantization_clipped_high += high
            if calibration is None:
                write_stempy_frames(temporary_path, quantized, np)
                calibration = calibrate_stempy_noise(
                    temporary_path,
                    quantized.shape[1:],
                    min(32, quantized.shape[0]),
                    args.xray_sigma,
                    np,
                )
                thresholds = (
                    calibration["optimized_mean"]
                    + candidate_sigmas * calibration["optimized_stddev"]
                )
                xray_threshold = (
                    calibration["optimized_mean"]
                    + args.xray_sigma * calibration["optimized_stddev"]
                )
                print(
                    "STEMPy Gaussian fit: "
                    f"mean={calibration['optimized_mean']:.3f}, "
                    f"sigma={calibration['optimized_stddev']:.3f}",
                    flush=True,
                )
            dark_candidate_counts += candidate_event_counts(
                quantized, thresholds, xray_threshold, np
            )
            dark_frames += quantized.shape[0]
            print(
                f"  dark batch {batch_index + 1}: {metadata['source_path']} "
                f"frames {metadata['start_frame']}:{metadata['end_frame']}",
                flush=True,
            )
            del core, quantized

        print("Processing spectrum calibration batches...", flush=True)
        for batch_index, (core, metadata) in enumerate(corrected_batches(
            spectrum_paths, dark_frame, valid_mask, config, args.tensor_frames,
            args.reader, np,
            counting_common_mode=args.counting_common_mode,
            max_batches_per_file=args.spectrum_calibration_batches_per_file,
        )):
            quantized, low, high = quantize_frames(
                core, args.quantization_scale, args.quantization_offset, np
            )
            quantization_clipped_low += low
            quantization_clipped_high += high
            spectrum_candidate_counts += candidate_event_counts(
                quantized, thresholds, xray_threshold, np
            )
            sample_spectrum_frames += quantized.shape[0]
            print(
                f"  spectrum calibration batch {batch_index + 1}: "
                f"{metadata['source_path']} frames {metadata['start_frame']}:{metadata['end_frame']}",
                flush=True,
            )
            del core, quantized

        threshold_rows = []
        tail_width = 3072 - tail_core
        for index, sigma in enumerate(candidate_sigmas):
            dark_tail = int(dark_candidate_counts[index, tail_core:].sum())
            spectrum_tail = int(spectrum_candidate_counts[index, tail_core:].sum())
            dark_per_frame = dark_tail / dark_frames
            spectrum_per_frame = spectrum_tail / sample_spectrum_frames
            dark_fraction = dark_per_frame / spectrum_per_frame if spectrum_per_frame else float("inf")
            threshold_rows.append({
                "sigma_multiplier": float(sigma),
                "threshold_quantized": float(thresholds[index]),
                "threshold_corrected_units": float(
                    (thresholds[index] - args.quantization_offset) / args.quantization_scale
                ),
                "dark_events_tail": dark_tail,
                "spectrum_sample_events_tail": spectrum_tail,
                "dark_frames": dark_frames,
                "spectrum_sample_frames": sample_spectrum_frames,
                "tail_pixels_per_frame": 896 * tail_width,
                "dark_events_per_frame_tail": dark_per_frame,
                "spectrum_events_per_frame_tail": spectrum_per_frame,
                "estimated_dark_fraction_tail": dark_fraction,
            })

        selected_index = None
        for index, row in enumerate(threshold_rows):
            if (
                row["estimated_dark_fraction_tail"] <= args.maximum_dark_fraction
                and row["spectrum_sample_events_tail"] > 0
            ):
                selected_index = index
                break
        if selected_index is None:
            selected_index = len(threshold_rows) - 1
        selected = threshold_rows[selected_index]
        selected_threshold = float(thresholds[selected_index])
        print(
            f"Selected {selected['sigma_multiplier']:g}σ threshold "
            f"({selected['threshold_corrected_units']:.2f} corrected units), "
            f"estimated tail dark fraction {selected['estimated_dark_fraction_tail']:.3%}",
            flush=True,
        )

        print("Counting every spectrum frame with STEMPy...", flush=True)
        total_counts = np.zeros(3072, dtype=np.uint64)
        per_batch_counts = []
        batch_metadata = []
        per_frame_counts = []
        per_frame_metadata = []
        total_frames = 0
        total_events = 0
        for batch_index, (core, metadata) in enumerate(corrected_batches(
            spectrum_paths, dark_frame, valid_mask, config, args.tensor_frames,
            args.reader, np,
            counting_common_mode=args.counting_common_mode,
            max_batches_per_file=args.max_count_batches_per_file,
        )):
            quantized, low, high = quantize_frames(
                core, args.quantization_scale, args.quantization_offset, np
            )
            quantization_clipped_low += low
            quantization_clipped_high += high
            write_stempy_frames(temporary_path, quantized, np)
            batch_counts, half_counts, event_count, batch_frame_counts = stempy_count(
                temporary_path,
                quantized.shape[1:],
                quantized.shape[0],
                selected_threshold,
                xray_threshold,
                np,
            )
            total_counts += batch_counts
            per_batch_counts.append(batch_counts)
            batch_metadata.append(metadata)
            per_frame_counts.append(batch_frame_counts)
            per_frame_metadata.extend({
                "source_index": metadata["source_index"],
                "source_frame": metadata["start_frame"] + local_frame,
            } for local_frame in range(batch_frame_counts.shape[0]))
            total_frames += quantized.shape[0]
            total_events += event_count
            print(
                f"  counted batch {batch_index + 1}: "
                f"{metadata['source_path']} {metadata['start_frame']}:{metadata['end_frame']} "
                f"events={event_count}",
                flush=True,
            )
            del core, quantized, batch_counts, half_counts, batch_frame_counts

    per_batch_counts = np.stack(per_batch_counts)
    per_frame_counts = np.concatenate(per_frame_counts, axis=0)
    occupancy = occupancy_summary(
        total_counts,
        total_frames,
        896,
        args.occupancy_bin_columns,
        args.occupancy_neighborhood_pixels,
        args.maximum_conditional_pileup,
        np,
    )
    noise_start_raw = max(tail_raw, occupancy["boundary_raw"])
    noise = binned_noise_statistics(
        per_batch_counts,
        noise_start_raw,
        3840,
        args.noise_bin_columns,
        np,
    )
    dead_bins = (
        (noise["starts_raw"] < DEAD_ADC_RAW[1])
        & (noise["stops_raw"] > DEAD_ADC_RAW[0])
    )
    valid_noise = (
        ~dead_bins
        & np.isfinite(noise["empirical_relative_sem"])
        & np.isfinite(noise["poisson_relative"])
        & (noise["poisson_relative"] > 0)
    )
    excess = np.divide(
        noise["empirical_relative_sem"],
        noise["poisson_relative"],
        out=np.full_like(noise["poisson_relative"], np.nan),
        where=noise["poisson_relative"] > 0,
    )

    summary = {
        "current_key": args.current_key,
        "current_label": CURRENT_LABELS[args.current_key],
        "method": "STEMPy standard threshold + strict 8-neighbor local maximum",
        "event_coordinate_decoder": (
            "Copies contiguous C++ event-vector storage when STEMPy 3.4.2 exposes "
            "zero-stride NumPy views under NumPy 2"
        ),
        "stempy_version": getattr(stempy, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "dark_frame": str(dark_frame_path),
        "analog_reference": str(analog_path),
        "dark_files": [str(path) for path in dark_paths],
        "spectrum_files": [str(path) for path in spectrum_paths],
        "processing": asdict(config),
        "blr_mode": args.blr_mode,
        "counting_common_mode": {
            "enabled": args.counting_common_mode,
            "estimator": "per-frame per-column median of 448 imaging rows, independently per half",
            "position_in_chain": "after standard corrections and imaging/CoreLoss crop; before quantization and STEMPy",
        },
        "quantization": {
            "scale": args.quantization_scale,
            "offset": args.quantization_offset,
            "clipped_low_samples_all_passes": quantization_clipped_low,
            "clipped_high_samples_all_passes": quantization_clipped_high,
        },
        "stempy_gaussian_calibration": calibration,
        "selected_threshold": {
            **selected,
            "xray_sigma_multiplier": args.xray_sigma,
            "xray_threshold_quantized": xray_threshold,
            "selection_target_dark_fraction": args.maximum_dark_fraction,
        },
        "energy_calibration": {
            "provisional": True,
            "zlp_peak_channel": zlp_peak,
            "first_loss_channel": first_loss,
            "first_loss_energy_ev": args.first_loss_energy_ev,
            "dispersion_ev_per_channel": dispersion,
            "tail_energy_ev": args.tail_energy_ev,
            "tail_stitched_channel": tail_stitched,
            "tail_raw_channel": tail_raw,
        },
        "counting_validity": {
            "criterion": "Poisson P(>=2 | >=1) in effective local-maximum neighborhood",
            "neighborhood_pixels": args.occupancy_neighborhood_pixels,
            "maximum_conditional_pileup": args.maximum_conditional_pileup,
            "bin_columns": args.occupancy_bin_columns,
            "first_valid_raw_column": occupancy["boundary_raw"],
            "first_valid_stitched_column": occupancy["boundary_stitched"],
            "first_valid_energy_ev": (
                occupancy["boundary_stitched"] - zlp_peak
            ) * dispersion,
        },
        "counting": {
            "frames": total_frames,
            "batches": int(per_batch_counts.shape[0]),
            "events_in_full_coreloss": total_events,
            "events_at_or_after_valid_boundary": int(
                total_counts[occupancy["boundary_raw"] - RAW_ZLP_WIDTH:].sum()
            ),
            "dead_adc_raw_columns_excluded_from_noise": list(DEAD_ADC_RAW),
        },
        "tail_noise": {
            "bin_columns": args.noise_bin_columns,
            "start_raw_column": noise_start_raw,
            "median_empirical_relative_sem": float(
                np.nanmedian(noise["empirical_relative_sem"][valid_noise])
            ),
            "median_poisson_1_over_sqrt_N": float(
                np.nanmedian(noise["poisson_relative"][valid_noise])
            ),
            "median_empirical_excess_factor": float(np.nanmedian(excess[valid_noise])),
        },
        "caveats": [
            "One STEMPy local maximum is treated as one electron only in the measured sparse region.",
            "The provisional energy calibration assigns the first broad low-loss peak to the requested energy.",
            "The known dead ADC block is omitted rather than reconstructed in counted-electron space.",
            "Columnwise BLR is used by default because grouped BLR steps bias strict horizontal local-maximum tests.",
            "The optional counting-only imaging-row common-mode median assumes sparse occupancy and is not applied in the ZLP.",
            "The empirical batch variation includes residual dose drift and detector correlations as well as counting noise.",
        ],
    }

    serializable_occupancy = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in occupancy.items()
    }
    write_outputs(
        args, summary, threshold_rows, serializable_occupancy, total_counts,
        per_batch_counts, batch_metadata, per_frame_counts, per_frame_metadata,
        noise, analog_profile, np
    )
    make_plots(
        args, summary, threshold_rows, occupancy, total_counts, per_frame_counts,
        per_frame_metadata, noise, analog_profile, np, plt
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_sigma": selected["sigma_multiplier"],
        "counting_valid_raw_column": occupancy["boundary_raw"],
        "counting_valid_energy_ev": summary["counting_validity"]["first_valid_energy_ev"],
        "total_frames": total_frames,
        "tail_empirical_excess_factor": summary["tail_noise"]["median_empirical_excess_factor"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
