#!/usr/bin/env python3
"""Count CoreLoss electrons with a fixed per-detector-pixel threshold map.

The first calibration dark files estimate a robust temporal mean and standard
deviation for every imaging-row CoreLoss pixel. Remaining dark files are held
out to measure false counts. Counting uses the same strict eight-neighbor local
maximum rule as STEMPy Classic, but a custom implementation is required because
STEMPy Classic accepts only one scalar background threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from scripts.studies.run_nio_counting_study import (
    CURRENT_LABELS,
    DEAD_ADC_RAW,
    RAW_TO_STITCHED_CORE_OFFSET,
    RAW_ZLP_WIDTH,
    binned_noise_statistics,
    corrected_batches,
    energy_calibration,
    load_analog_reference,
    load_dark_products,
    load_manifest_group,
    neighboring_maximum,
    occupancy_summary,
    processor_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dark-frame-path",
        type=Path,
        help=(
            "Dark reference and valid-mask HDF5. By default uses the current study dark "
            "product; for independent validation, provide one built only from calibration files."
        ),
    )
    parser.add_argument("--current-key", default="0015pA", choices=tuple(CURRENT_LABELS))
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--blr-mode", choices=("grouped", "columnwise"), default="columnwise")
    parser.add_argument("--dynamic-mask-window", type=int, default=31)
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=4.5,
        help="Per-pixel threshold multiplier k in T[p] = mean[p] + k * sigma[p].",
    )
    parser.add_argument(
        "--calibration-dark-files",
        type=int,
        default=3,
        help="Number of leading dark files used to fit the threshold map.",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=5.0,
        help="Second-pass symmetric clipping limit around preliminary per-pixel moments.",
    )
    parser.add_argument(
        "--minimum-calibration-samples",
        type=int,
        default=128,
        help="Pixels with fewer retained dark samples are disabled.",
    )
    parser.add_argument(
        "--sigma-floor-percentile",
        type=float,
        default=1.0,
        help="Floor per-pixel sigma at this percentile of valid fitted sigmas.",
    )
    parser.add_argument(
        "--xray-sigma",
        type=float,
        default=175.0,
        help="Global X-ray veto = median(pixel mean) + value * median(pixel sigma).",
    )
    parser.add_argument("--occupancy-neighborhood-pixels", type=int, default=9)
    parser.add_argument("--maximum-conditional-pileup", type=float, default=0.05)
    parser.add_argument("--occupancy-bin-columns", type=int, default=64)
    parser.add_argument("--noise-bin-columns", type=int, default=16)
    parser.add_argument("--first-loss-energy-ev", type=float, default=25.0)
    parser.add_argument("--tail-energy-ev", type=float, default=350.0)
    parser.add_argument(
        "--max-count-batches-per-file",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit to count every spectrum frame.",
    )
    return parser.parse_args()


def static_core_valid_map(valid_mask, np):
    imaging = np.concatenate((valid_mask[32:480], valid_mask[480:928]), axis=0)
    return np.ascontiguousarray(imaging[:, RAW_ZLP_WIDTH:] != 0)


def accumulate_moments(paths,
                       dark_frame,
                       valid_mask,
                       config,
                       args,
                       np,
                       preliminary=None):
    shape = (896, 3840 - RAW_ZLP_WIDTH)
    count = np.zeros(shape, dtype=np.uint32)
    total = np.zeros(shape, dtype=np.float64)
    total2 = np.zeros(shape, dtype=np.float64)
    source_frames = 0

    for core, metadata in corrected_batches(
        paths, dark_frame, valid_mask, config, args.tensor_frames, args.reader, np
    ):
        for chunk_start in range(0, core.shape[0], 8):
            values = core[chunk_start:chunk_start + 8]
            # Processor masks are represented by exact zeros and must not enter
            # the temporal noise fit as detector samples.
            keep = np.isfinite(values) & (values != 0.0)
            if preliminary is not None:
                center, width = preliminary
                keep &= np.abs(values - center[None]) <= args.clip_sigma * width[None]
            count += keep.sum(axis=0, dtype=np.uint32)
            total += np.where(keep, values, 0.0).sum(axis=0, dtype=np.float64)
            total2 += np.where(keep, values * values, 0.0).sum(axis=0, dtype=np.float64)
        source_frames += int(core.shape[0])
        print(
            f"  moments {Path(metadata['source_path']).name} "
            f"{metadata['start_frame']}:{metadata['end_frame']}",
            flush=True,
        )
        del core, values, keep

    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(total2, count, out=np.zeros_like(total2), where=count > 0) - mean * mean
    sigma = np.sqrt(np.maximum(variance, 0.0))
    return mean.astype(np.float32), sigma.astype(np.float32), count, source_frames


def calibrate_threshold_map(paths, dark_frame, valid_mask, config, args, np):
    print("Per-pixel calibration pass 1...", flush=True)
    mean0, sigma0, _, frames = accumulate_moments(
        paths, dark_frame, valid_mask, config, args, np
    )
    preliminary_width = np.maximum(sigma0, np.finfo(np.float32).eps)

    print("Per-pixel clipped calibration pass 2...", flush=True)
    mean, sigma, count, frames_second = accumulate_moments(
        paths,
        dark_frame,
        valid_mask,
        config,
        args,
        np,
        preliminary=(mean0, preliminary_width),
    )
    if frames_second != frames:
        raise RuntimeError("dark calibration frame count changed between passes")

    static_valid = static_core_valid_map(valid_mask, np)
    fitted = (
        static_valid
        & np.isfinite(mean)
        & np.isfinite(sigma)
        & (count >= args.minimum_calibration_samples)
        & (sigma > 0)
    )
    if not np.any(fitted):
        raise RuntimeError("no detector pixels passed per-pixel calibration validity cuts")
    sigma_floor = float(np.percentile(sigma[fitted], args.sigma_floor_percentile))
    sigma = np.where(fitted, np.maximum(sigma, sigma_floor), np.nan).astype(np.float32)
    threshold = (mean + args.sigma_multiplier * sigma).astype(np.float32)
    threshold[~fitted] = np.inf
    xray_threshold = float(np.median(mean[fitted]) + args.xray_sigma * np.median(sigma[fitted]))
    return {
        "mean": mean,
        "sigma": sigma,
        "threshold": threshold,
        "retained_samples": count,
        "valid": fitted,
        "frames": frames,
        "sigma_floor": sigma_floor,
        "xray_threshold": xray_threshold,
    }


def count_batch(frames, threshold, valid, xray_threshold: float, np):
    per_column = np.zeros(frames.shape[2], dtype=np.uint64)
    per_frame = np.zeros((frames.shape[0], frames.shape[2]), dtype=np.uint16)
    event_count = 0
    for frame_index, frame in enumerate(frames):
        candidates = (
            valid
            & (frame > threshold)
            & (frame < xray_threshold)
            & (frame > neighboring_maximum(frame, np))
        )
        columns = np.nonzero(candidates)[1]
        frame_counts = np.bincount(columns, minlength=frames.shape[2]).astype(np.uint16)
        per_frame[frame_index] = frame_counts
        per_column += frame_counts
        event_count += int(columns.size)
    return per_column, per_frame, event_count


def count_paths(paths,
                dark_frame,
                valid_mask,
                config,
                calibration,
                args,
                np,
                max_batches_per_file=None):
    total = np.zeros(3840 - RAW_ZLP_WIDTH, dtype=np.uint64)
    batches = []
    frames = []
    metadata_rows = []
    total_events = 0
    total_frames = 0
    for batch_index, (core, metadata) in enumerate(corrected_batches(
        paths,
        dark_frame,
        valid_mask,
        config,
        args.tensor_frames,
        args.reader,
        np,
        max_batches_per_file=max_batches_per_file,
    )):
        batch_counts, frame_counts, events = count_batch(
            core,
            calibration["threshold"],
            calibration["valid"],
            calibration["xray_threshold"],
            np,
        )
        total += batch_counts
        batches.append(batch_counts)
        frames.append(frame_counts)
        metadata_rows.extend({
            "source_index": metadata["source_index"],
            "source_frame": metadata["start_frame"] + index,
        } for index in range(frame_counts.shape[0]))
        total_events += events
        total_frames += frame_counts.shape[0]
        print(
            f"  counted batch {batch_index + 1}: {Path(metadata['source_path']).name} "
            f"{metadata['start_frame']}:{metadata['end_frame']} events={events}",
            flush=True,
        )
        del core, batch_counts, frame_counts
    if not batches:
        raise RuntimeError("no frames were counted")
    return {
        "total": total,
        "per_batch": np.stack(batches),
        "per_frame": np.concatenate(frames),
        "metadata": metadata_rows,
        "events": total_events,
        "frames": total_frames,
    }


def save_outputs(args, summary, calibration, validation, spectrum, noise, np):
    import h5py

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "counting_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with h5py.File(args.output_dir / "per_pixel_thresholds.h5", "w") as output:
        for name in ("mean", "sigma", "threshold", "retained_samples", "valid"):
            output.create_dataset(name, data=calibration[name], compression="gzip")
        output.attrs["sigma_multiplier"] = args.sigma_multiplier
        output.attrs["sigma_floor"] = calibration["sigma_floor"]
        output.attrs["xray_threshold"] = calibration["xray_threshold"]
        output.attrs["calibration_frames"] = calibration["frames"]

    raw_columns = np.arange(RAW_ZLP_WIDTH, 3840)
    stitched_columns = raw_columns - RAW_TO_STITCHED_CORE_OFFSET
    with h5py.File(args.output_dir / "counted_spectrum.h5", "w") as output:
        output.create_dataset("total_core_counts", data=spectrum["total"], compression="gzip")
        output.create_dataset("per_batch_core_counts", data=spectrum["per_batch"], compression="gzip")
        output.create_dataset(
            "per_frame_core_counts",
            data=spectrum["per_frame"],
            chunks=(1, spectrum["per_frame"].shape[1]),
            compression="gzip",
        )
        output.create_dataset("raw_core_columns", data=raw_columns)
        output.create_dataset("stitched_core_columns", data=stitched_columns)
        output.create_dataset("validation_dark_total_core_counts", data=validation["total"])

    with (args.output_dir / "counted_spectrum.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("raw_column", "stitched_column", "electrons", "electrons_per_frame"))
        writer.writerows(zip(
            raw_columns,
            stitched_columns,
            spectrum["total"],
            spectrum["total"] / spectrum["frames"],
        ))

    with (args.output_dir / "tail_noise_binned.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "raw_start", "raw_stop", "stitched_center", "total_counts",
            "empirical_relative_sem", "poisson_relative_1_over_sqrt_N",
        ))
        writer.writerows(zip(
            noise["starts_raw"], noise["stops_raw"], noise["centers_stitched"],
            noise["total_counts"], noise["empirical_relative_sem"], noise["poisson_relative"],
        ))


def make_plots(args, summary, calibration, validation, spectrum, analog, noise, np, plt):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    valid = calibration["valid"]
    images = (
        (calibration["mean"], "per-pixel dark residual mean", "coolwarm"),
        (calibration["sigma"], "per-pixel dark residual sigma", "magma"),
        (calibration["threshold"], f"threshold map: mean + {args.sigma_multiplier:g} sigma", "viridis"),
        (calibration["retained_samples"], "retained calibration samples", "viridis"),
    )
    for axis, (values, title, cmap) in zip(axes.flat, images):
        finite = values[valid]
        low, high = np.percentile(finite, (1, 99))
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=low, vmax=high)
        axis.set(title=title, xlabel="CoreLoss column", ylabel="imaging row")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(args.output_dir / "per_pixel_threshold_maps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].hist(calibration["sigma"][valid], bins=200, histtype="step", color="#2563EB")
    axes[0].axvline(calibration["sigma_floor"], color="#DC2626", linestyle="--", label="sigma floor")
    axes[0].set(xlabel="per-pixel dark residual sigma", ylabel="pixels", title="Noise-width distribution")
    axes[0].legend()
    stitched = np.arange(RAW_ZLP_WIDTH, 3840) - RAW_TO_STITCHED_CORE_OFFSET
    axes[1].plot(stitched, validation["total"] / validation["frames"], color="#DC2626", linewidth=0.8)
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[1].set(
        xlabel="stitched spectral channel",
        ylabel="false counts / validation dark frame",
        title=f"Held-out dark false counts ({validation['frames']} frames)",
    )
    fig.savefig(args.output_dir / "threshold_calibration_validation.png", dpi=180)
    plt.close(fig)

    counts_per_frame = spectrum["total"].astype(np.float64) / spectrum["frames"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True)
    full = np.arange(analog.size)
    axes[0].plot(full, np.ma.masked_less_equal(analog, 0), color="#6B7280", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="analog corrected intensity", title="Final analog reference")
    axes[1].plot(stitched, np.ma.masked_less_equal(counts_per_frame, 0), color="#0F766E", linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].axvspan(
        DEAD_ADC_RAW[0] - RAW_TO_STITCHED_CORE_OFFSET,
        DEAD_ADC_RAW[1] - RAW_TO_STITCHED_CORE_OFFSET,
        color="#DC2626",
        alpha=0.12,
        label="dead ADC block",
    )
    axes[1].set(
        xlabel="stitched spectral channel",
        ylabel="counted electrons / frame",
        title=f"Per-pixel {args.sigma_multiplier:g}-sigma counted CoreLoss spectrum",
    )
    axes[1].set_xlim(0, analog.size - 1)
    axes[1].legend()
    fig.savefig(args.output_dir / "counted_spectrum_full_analog_zlp.png", dpi=180)
    plt.close(fig)

    zlp_peak = summary["energy_calibration"]["zlp_peak_channel"]
    dispersion = summary["energy_calibration"]["dispersion_ev_per_channel"]
    energy = (noise["centers_stitched"] - zlp_peak) * dispersion
    dead = (
        (noise["starts_raw"] < DEAD_ADC_RAW[1])
        & (noise["stops_raw"] > DEAD_ADC_RAW[0])
    )
    empirical = noise["empirical_relative_sem"].copy()
    poisson = noise["poisson_relative"].copy()
    empirical[dead] = np.nan
    poisson[dead] = np.nan
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(energy, empirical, "o-", markersize=3, label="empirical relative SEM")
    ax.plot(energy, poisson, "o-", markersize=3, label=r"Poisson $1/\sqrt{N}$")
    ax.set_yscale("log")
    ax.set(
        xlabel="provisional energy loss [eV]",
        ylabel="relative uncertainty",
        title="Per-pixel-threshold tail fluctuations",
    )
    ax.legend()
    fig.savefig(args.output_dir / "counted_tail_poisson.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    if args.sigma_multiplier <= 0:
        raise ValueError("sigma multiplier must be positive")
    dark_paths, spectrum_paths = load_manifest_group(args.study_root, args.current_key)
    if not 1 <= args.calibration_dark_files < len(dark_paths):
        raise ValueError(
            "calibration-dark-files must leave at least one dark file for validation"
        )
    calibration_paths = dark_paths[:args.calibration_dark_files]
    validation_paths = dark_paths[args.calibration_dark_files:]
    dark_frame_path = args.dark_frame_path or (
        args.study_root / "nio_beam_current" / "currents" / args.current_key
        / "dark" / "dark_frame.h5"
    )
    dark_frame_path = dark_frame_path.resolve()
    dark_frame, valid_mask = load_dark_products(dark_frame_path, np)
    config = processor_config(args.blr_mode, args.dynamic_mask_window)
    _, analog, analog_path = load_analog_reference(args.study_root, args.current_key, np)
    zlp_peak, first_loss, dispersion = energy_calibration(
        analog, args.first_loss_energy_ev, np, signal
    )
    tail_stitched = int(round(zlp_peak + args.tail_energy_ev / dispersion))
    tail_raw = tail_stitched + RAW_TO_STITCHED_CORE_OFFSET

    calibration = calibrate_threshold_map(
        calibration_paths, dark_frame, valid_mask, config, args, np
    )
    print("Counting held-out dark validation files...", flush=True)
    validation = count_paths(
        validation_paths, dark_frame, valid_mask, config, calibration, args, np
    )
    print("Counting spectrum files...", flush=True)
    spectrum = count_paths(
        spectrum_paths,
        dark_frame,
        valid_mask,
        config,
        calibration,
        args,
        np,
        max_batches_per_file=args.max_count_batches_per_file,
    )

    occupancy = occupancy_summary(
        spectrum["total"],
        spectrum["frames"],
        896,
        args.occupancy_bin_columns,
        args.occupancy_neighborhood_pixels,
        args.maximum_conditional_pileup,
        np,
    )
    noise_start_raw = max(tail_raw, occupancy["boundary_raw"])
    noise = binned_noise_statistics(
        spectrum["per_batch"], noise_start_raw, 3840, args.noise_bin_columns, np
    )
    tail_core = max(0, tail_raw - RAW_ZLP_WIDTH)
    validation_tail = int(validation["total"][tail_core:].sum())
    spectrum_tail = int(spectrum["total"][tail_core:].sum())
    validation_tail_rate = validation_tail / validation["frames"]
    spectrum_tail_rate = spectrum_tail / spectrum["frames"]
    false_fraction = (
        validation_tail_rate / spectrum_tail_rate if spectrum_tail_rate else float("inf")
    )
    valid_sigma = calibration["sigma"][calibration["valid"]]
    summary = {
        "current_key": args.current_key,
        "current_label": CURRENT_LABELS[args.current_key],
        "method": "per-pixel dark threshold + strict 8-neighbor local maximum",
        "stempy_equivalence": (
            "The local-maximum rule matches STEMPy Classic; threshold comparison is custom "
            "because ElectronCountOptionsClassic.background_threshold is scalar."
        ),
        "dark_frame": str(dark_frame_path),
        "analog_reference": str(analog_path),
        "calibration_dark_files": [str(path) for path in calibration_paths],
        "validation_dark_files": [str(path) for path in validation_paths],
        "spectrum_files": [str(path) for path in spectrum_paths],
        "processing": asdict(config),
        "blr_mode": args.blr_mode,
        "threshold": {
            "definition": "mean[p] + sigma_multiplier * sigma[p]",
            "sigma_multiplier": args.sigma_multiplier,
            "clip_sigma": args.clip_sigma,
            "minimum_calibration_samples": args.minimum_calibration_samples,
            "sigma_floor_percentile": args.sigma_floor_percentile,
            "sigma_floor": calibration["sigma_floor"],
            "median_sigma": float(np.median(valid_sigma)),
            "valid_pixels": int(np.count_nonzero(calibration["valid"])),
            "total_pixels": int(calibration["valid"].size),
            "xray_threshold_corrected_units": calibration["xray_threshold"],
        },
        "held_out_dark_validation": {
            "frames": validation["frames"],
            "events_full_coreloss": validation["events"],
            "events_per_frame_full_coreloss": validation["events"] / validation["frames"],
            "events_tail": validation_tail,
            "events_per_frame_tail": validation_tail_rate,
            "estimated_tail_false_fraction": false_fraction,
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
            "first_valid_raw_column": occupancy["boundary_raw"],
            "first_valid_stitched_column": occupancy["boundary_stitched"],
            "first_valid_energy_ev": (occupancy["boundary_stitched"] - zlp_peak) * dispersion,
        },
        "counting": {
            "frames": spectrum["frames"],
            "batches": int(spectrum["per_batch"].shape[0]),
            "events_in_full_coreloss": spectrum["events"],
            "events_per_frame_full_coreloss": spectrum["events"] / spectrum["frames"],
            "events_tail": spectrum_tail,
            "events_per_frame_tail": spectrum_tail_rate,
        },
        "caveats": [
            "Per-pixel moments are estimated from a finite dark calibration set.",
            "The held-out dark file is the primary false-count diagnostic.",
            "One strict local maximum is treated as one electron only in the sparse region.",
            "Pixel-dependent electron detection efficiency is not yet corrected.",
        ],
    }
    save_outputs(args, summary, calibration, validation, spectrum, noise, np)
    make_plots(args, summary, calibration, validation, spectrum, analog, noise, np, plt)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "sigma_multiplier": args.sigma_multiplier,
        "calibration_frames": calibration["frames"],
        "validation_dark_events_per_frame_tail": validation_tail_rate,
        "spectrum_events_per_frame_tail": spectrum_tail_rate,
        "estimated_tail_false_fraction": false_fraction,
        "spectrum_frames": spectrum["frames"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
