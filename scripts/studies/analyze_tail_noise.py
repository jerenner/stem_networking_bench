#!/usr/bin/env python3
"""Compare pre/post-threshold tail noise with counting-like references.

The analysis uses matched 128-frame batches to estimate the relative standard
error of the complete acquisition. After thresholding, the number of retained
pixel samples provides a 1/sqrt(N) reference, but it is deliberately labelled
as a candidate-count benchmark rather than an absolute electron-count limit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CURRENT_LABELS = {
    "0015pA": "15 pA",
    "0030pA": "30 pA",
    "0060pA": "60 pA",
    "0130pA": "130 pA",
    "0250pA": "250 pA",
    "0500pA": "500 pA",
    "1000pA": "1 nA",
}

DEAD_BLOCK = (2272, 2288)
STITCHED_DEAD_BLOCK = (1696, 1712)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-key", default="0015pA", choices=tuple(CURRENT_LABELS))
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--first-loss-energy-ev",
        type=float,
        default=25.0,
        help="Provisional energy assigned to the first broad post-ZLP low-loss peak.",
    )
    parser.add_argument("--tail-energy-ev", type=float, default=350.0)
    parser.add_argument("--normalization-start", type=int, default=300)
    parser.add_argument("--normalization-end", type=int, default=900)
    return parser.parse_args()


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def estimate_linear_slope(profile, start: int, end: int, np) -> float:
    x = np.arange(start, end + 1, dtype=np.float64)
    y = profile[start:end + 1].astype(np.float64)
    valid = np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return 0.0
    centered = x - x.mean()
    denominator = np.sum(centered * centered)
    return 0.0 if denominator <= 0 else float(np.sum(centered * (y - y.mean())) / denominator)


def repair_stitch_boundary(profile, np):
    repaired = profile.copy()
    left, right = 190, 194
    y0, y1 = float(repaired[left]), float(repaired[right])
    m0 = estimate_linear_slope(repaired, 184, 190, np)
    m1 = estimate_linear_slope(repaired, 194, 200, np)
    span = float(right - left)
    for column in (191, 192, 193):
        t = (column - left) / span
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        repaired[column] = h00 * y0 + h10 * span * m0 + h01 * y1 + h11 * span * m1
    return repaired


def final_profile_from_halves(half_sums,
                              half_counts,
                              dead_gain: float,
                              dead_offset: float,
                              stitch_gain: float,
                              np):
    halves = safe_divide(half_sums, half_counts, np)
    top, bottom = halves
    first, last = DEAD_BLOCK
    reconstructed_top = top.copy()
    reconstructed_top[first:last] = dead_gain * bottom[first:last] + dead_offset
    corrected_sums = half_sums.copy()
    corrected_sums[0, first:last] = (
        reconstructed_top[first:last] * half_counts[0, first:last]
    )
    combined = safe_divide(corrected_sums.sum(axis=0), half_counts.sum(axis=0), np)
    folded_zlp = combined[:768].reshape(4, 192).sum(axis=0)
    stitched = np.concatenate((folded_zlp, combined[768:])).astype(np.float64)
    stitched[:192] *= stitch_gain
    return repair_stitch_boundary(stitched, np)


def fold_candidate_counts(counts, np):
    folded_zlp = counts[:768].reshape(4, 192).sum(axis=0)
    return np.concatenate((folded_zlp, counts[768:])).astype(np.float64)


def load_stitch_gain(path: Path, current_key: str) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["current_key"] == current_key:
                return float(row["applied_no_blr_gain"])
    raise ValueError(f"no stitch gain for {current_key} in {path}")


def normalized_batch_statistics(pre_batches, post_batches, start: int, end: int, np):
    reference = np.nanmedian(pre_batches, axis=0)
    scales = np.ones(pre_batches.shape[0], dtype=np.float64)
    valid_reference = np.isfinite(reference[start:end]) & (reference[start:end] > 0.0)
    for index, batch in enumerate(pre_batches):
        ratio = safe_divide(batch[start:end], reference[start:end], np)
        scales[index] = float(np.nanmedian(ratio[valid_reference]))
    pre_normalized = pre_batches / scales[:, None]
    post_normalized = post_batches / scales[:, None]

    def summarize(values):
        mean = np.nanmean(values, axis=0)
        stddev = np.nanstd(values, axis=0, ddof=1)
        sem = stddev / np.sqrt(values.shape[0])
        relative_sem = safe_divide(sem, np.abs(mean), np)
        return mean, sem, relative_sem

    return summarize(pre_normalized), summarize(post_normalized), scales


def smooth_positive(values, scipy_signal, np, window: int = 31):
    finite = np.asarray(values, dtype=np.float64)
    replacement = float(np.nanmedian(finite[np.isfinite(finite)]))
    prepared = np.nan_to_num(finite, nan=replacement, posinf=replacement, neginf=replacement)
    kernel = min(window, prepared.size if prepared.size % 2 else prepared.size - 1)
    kernel = max(kernel, 3)
    return scipy_signal.medfilt(prepared, kernel_size=kernel)


def main():
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = args.current_key
    label = CURRENT_LABELS[key]
    pre_path = (
        args.study_root / "nio_beam_current" / "currents" / key / "spectrum"
        / "final_spectrum.h5"
    )
    post_path = (
        args.study_root / "nio_threshold_study_3sigma" / "currents" / key
        / "threshold_spectra.h5"
    )
    pre_dead_path = (
        args.study_root / "nio_dead_adc_correction" / "currents" / key
        / "dead_adc_corrected_spectrum.h5"
    )
    final_summary_path = (
        args.study_root / "nio_processing_chain_deck"
        / "final_spectra_processing_summary.json"
    )
    stitch_metrics_path = (
        args.study_root / "nio_stitch_study"
        / "final_grouped_blr_no_blr_calibration_metrics.csv"
    )

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    post_calibration = final_summary["per_current"][key]
    stitch_gain = load_stitch_gain(stitch_metrics_path, key)
    with h5py.File(pre_dead_path, "r") as source:
        pre_dead_gain = float(source.attrs["top_bottom_gain"])
        pre_dead_offset = float(source.attrs["local_offset"])

    with h5py.File(pre_path, "r") as pre, h5py.File(post_path, "r") as post:
        for name in ("per_batch_source_index", "per_batch_start_frame", "per_batch_end_frame"):
            if not np.array_equal(pre[name][...], post[name][...]):
                raise ValueError(f"pre/post batch indexing differs for {name}")
        multipliers = post["multipliers"][...].astype(np.float64)
        threshold_index = int(np.argmin(np.abs(multipliers - args.threshold_multiplier)))
        if not np.isclose(multipliers[threshold_index], args.threshold_multiplier):
            raise ValueError(
                f"{post_path} does not contain multiplier {args.threshold_multiplier:g}"
            )
        pre_sums = pre["per_batch_full_columns_sum"][...].astype(np.float64)
        pre_counts = pre["per_batch_full_columns_valid_count"][...].astype(np.float64)
        post_sums = post["per_batch_sums"][:, threshold_index, ...].astype(np.float64)
        retained_counts = post[
            "per_batch_retained_counts"
        ][:, threshold_index, ...].astype(np.float64)
        source_indices = pre["per_batch_source_index"][...].astype(np.int32)
        start_frames = pre["per_batch_start_frame"][...].astype(np.int64)
        end_frames = pre["per_batch_end_frame"][...].astype(np.int64)
        zlp_threshold = float(post["zlp_thresholds"][threshold_index])
        core_threshold = float(post["core_thresholds"][threshold_index])

    pre_batches = np.stack([
        final_profile_from_halves(
            sums,
            counts,
            pre_dead_gain,
            pre_dead_offset,
            stitch_gain,
            np,
        )
        for sums, counts in zip(pre_sums, pre_counts)
    ])
    post_batches = np.stack([
        final_profile_from_halves(
            sums,
            counts,
            float(post_calibration["dead_adc_top_bottom_gain"]),
            float(post_calibration["dead_adc_local_offset"]),
            stitch_gain,
            np,
        )
        for sums, counts in zip(post_sums, pre_counts)
    ])

    (pre_stats, post_stats, dose_scales) = normalized_batch_statistics(
        pre_batches,
        post_batches,
        args.normalization_start,
        args.normalization_end,
        np,
    )
    pre_mean, pre_sem, pre_relative_sem = pre_stats
    post_mean, post_sem, post_relative_sem = post_stats
    batch_count = pre_batches.shape[0]
    columns = np.arange(pre_mean.size)

    zlp_peak_channel = int(np.nanargmax(post_mean[:192]))
    smoothed_zlp = signal.savgol_filter(post_mean[:192], 21, 3)
    first_loss_start = max(zlp_peak_channel + 60, 100)
    first_loss_channel = first_loss_start + int(
        np.nanargmax(smoothed_zlp[first_loss_start:190])
    )
    dispersion_ev_per_channel = (
        args.first_loss_energy_ev / (first_loss_channel - zlp_peak_channel)
    )
    tail_start_channel = int(round(
        zlp_peak_channel + args.tail_energy_ev / dispersion_ev_per_channel
    ))
    tail_start_channel = min(max(tail_start_channel, 192), pre_mean.size - 2)

    candidate_total = retained_counts.sum(axis=(0, 1))
    candidate_stitched = fold_candidate_counts(candidate_total, np)
    candidate_poisson = safe_divide(
        np.ones_like(candidate_stitched),
        np.sqrt(candidate_stitched),
        np,
    )
    candidate_poisson[
        STITCHED_DEAD_BLOCK[0]:STITCHED_DEAD_BLOCK[1]
    ] = np.nan

    pre_relative_smooth = smooth_positive(pre_relative_sem, signal, np)
    post_relative_smooth = smooth_positive(post_relative_sem, signal, np)
    poisson_smooth = smooth_positive(candidate_poisson, signal, np)
    tail = slice(tail_start_channel, pre_mean.size)
    excluded = np.zeros(pre_mean.size, dtype=bool)
    excluded[STITCHED_DEAD_BLOCK[0]:STITCHED_DEAD_BLOCK[1]] = True
    tail_valid = (
        (columns >= tail_start_channel)
        & ~excluded
        & np.isfinite(pre_mean)
        & np.isfinite(post_mean)
        & (pre_mean > 0.0)
        & (post_mean > 0.0)
        & np.isfinite(candidate_poisson)
    )

    analog_k = float(np.nanmedian(
        pre_relative_smooth[tail_valid] * np.sqrt(pre_mean[tail_valid])
    ))
    analog_counting_reference = safe_divide(
        np.full_like(pre_mean, analog_k),
        np.sqrt(np.maximum(pre_mean, 0.0)),
        np,
    )
    analog_reference_smooth = smooth_positive(analog_counting_reference, signal, np)

    def channel_to_energy(channel):
        return (channel - zlp_peak_channel) * dispersion_ev_per_channel

    def energy_to_channel(energy):
        return energy / dispersion_ev_per_channel + zlp_peak_channel

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFA",
        "grid.color": "#D9DED8",
    })

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    axes[0].plot(columns, np.ma.masked_less_equal(pre_mean, 0.0), color="#4B5563",
                 linewidth=0.9, label="before 3σ threshold")
    axes[0].plot(columns, np.ma.masked_less_equal(post_mean, 0.0), color="#0F8B8D",
                 linewidth=1.0, label="after 3σ threshold")
    axes[0].set_yscale("log")
    axes[0].axvline(tail_start_channel, color="#C8553D", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{label}: final spectrum before and after the regional 3σ threshold",
                      loc="left", fontweight="bold")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.6, which="both")

    axes[1].plot(columns[tail], np.ma.masked_less_equal(pre_mean[tail], 0.0),
                 color="#4B5563", linewidth=0.9, label="before threshold")
    axes[1].plot(columns[tail], np.ma.masked_less_equal(post_mean[tail], 0.0),
                 color="#0F8B8D", linewidth=1.0, label="after threshold")
    axes[1].set_yscale("log")
    axes[1].axvspan(*STITCHED_DEAD_BLOCK, color="#E9C46A", alpha=0.20)
    axes[1].set_title(
        f"Tail zoom: nominally > {args.tail_energy_ev:g} eV under provisional calibration",
        loc="left",
    )
    axes[1].set_xlabel("Stitched spectral channel")
    axes[1].set_ylabel("Mean corrected detector value")
    axes[1].grid(alpha=0.6, which="both")
    axes[1].legend(frameon=False)
    secondary = axes[1].secondary_xaxis(
        "top", functions=(channel_to_energy, energy_to_channel)
    )
    secondary.set_xlabel(
        "Provisional energy loss (eV): ZLP=0; first broad low-loss peak=25 eV"
    )
    fig.savefig(args.output_dir / "tail_spectrum_before_after_log.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True, sharex=True)
    axes[0].plot(columns[tail], pre_relative_sem[tail], color="#9CA3AF",
                 linewidth=0.45, alpha=0.55)
    axes[0].plot(columns[tail], post_relative_sem[tail], color="#5BC0BE",
                 linewidth=0.45, alpha=0.55)
    axes[0].plot(columns[tail], pre_relative_smooth[tail], color="#374151",
                 linewidth=1.5, label="empirical relative SEM, before threshold")
    axes[0].plot(columns[tail], post_relative_smooth[tail], color="#0F8B8D",
                 linewidth=1.5, label="empirical relative SEM, after threshold")
    axes[0].plot(columns[tail], analog_reference_smooth[tail], color="#6B7280",
                 linestyle="--", linewidth=1.2,
                 label=r"best-fit analog $K/\sqrt{S}$ reference")
    axes[0].plot(columns[tail], poisson_smooth[tail], color="#C8553D",
                 linestyle="--", linewidth=1.5,
                 label=r"$1/\sqrt{N_{>T}}$ candidate-count benchmark")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative uncertainty of complete acquisition")
    axes[0].set_title(
        f"{label} tail-noise comparison from {batch_count} matched 128-frame batches",
        loc="left",
        fontweight="bold",
    )
    axes[0].grid(alpha=0.6, which="both")
    axes[0].legend(frameon=False, ncol=2)

    ratio = safe_divide(post_relative_smooth, poisson_smooth, np)
    axes[1].plot(columns[tail], ratio[tail], color="#0F8B8D", linewidth=1.0)
    axes[1].axhline(1.0, color="#C8553D", linestyle="--", linewidth=1.2)
    axes[1].axvspan(*STITCHED_DEAD_BLOCK, color="#E9C46A", alpha=0.20)
    axes[1].set_ylim(0.0, min(10.0, float(np.nanpercentile(ratio[tail_valid], 99))))
    axes[1].set_xlabel("Stitched spectral channel")
    axes[1].set_ylabel(r"Empirical SEM / $1/\sqrt{N_{>T}}$")
    axes[1].set_title(
        "Values above 1 indicate fluctuations beyond independent threshold-crossing samples",
        loc="left",
    )
    axes[1].grid(alpha=0.6)
    secondary = axes[1].secondary_xaxis(
        "top", functions=(channel_to_energy, energy_to_channel)
    )
    secondary.set_xlabel("Provisional energy loss (eV)")
    fig.savefig(args.output_dir / "tail_noise_counting_comparison.png", dpi=220)
    plt.close(fig)

    metrics = {
        "current_key": key,
        "current_label": label,
        "batch_count": int(batch_count),
        "frames_per_batch": 128,
        "zlp_threshold": zlp_threshold,
        "core_threshold": core_threshold,
        "zlp_peak_channel": zlp_peak_channel,
        "first_loss_peak_channel": first_loss_channel,
        "assigned_first_loss_energy_ev": args.first_loss_energy_ev,
        "provisional_dispersion_ev_per_channel": dispersion_ev_per_channel,
        "tail_energy_ev": args.tail_energy_ev,
        "tail_start_channel": tail_start_channel,
        "median_relative_sem_before_tail": float(
            np.nanmedian(pre_relative_smooth[tail_valid])
        ),
        "median_relative_sem_after_tail": float(
            np.nanmedian(post_relative_smooth[tail_valid])
        ),
        "median_candidate_poisson_tail": float(
            np.nanmedian(poisson_smooth[tail_valid])
        ),
        "median_post_excess_factor": float(
            np.nanmedian(ratio[tail_valid])
        ),
        "analog_k_fitted_before_threshold": analog_k,
        "candidate_count_warning": (
            "N_>T counts threshold-crossing pixel samples, not independently "
            "identified electrons; charge sharing and event overlap are not corrected."
        ),
        "energy_warning": (
            "DM4 axes contain no energy calibration. The eV axis is provisional, "
            "anchored by assigning the first broad post-ZLP peak to the requested energy."
        ),
        "dose_normalization": {
            "reference_stitched_channels": [
                args.normalization_start,
                args.normalization_end - 1,
            ],
            "batch_scale_min": float(np.nanmin(dose_scales)),
            "batch_scale_max": float(np.nanmax(dose_scales)),
        },
        "sources": {
            "pre_threshold": str(pre_path),
            "post_threshold": str(post_path),
        },
    }
    (args.output_dir / "tail_noise_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    with (args.output_dir / "tail_noise_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "stitched_channel",
            "provisional_energy_ev",
            "mean_before_threshold",
            "mean_after_threshold",
            "relative_sem_before",
            "relative_sem_after",
            "candidate_count",
            "candidate_poisson_relative",
        ])
        for column in columns:
            writer.writerow([
                column,
                channel_to_energy(column),
                pre_mean[column],
                post_mean[column],
                pre_relative_sem[column],
                post_relative_sem[column],
                candidate_stitched[column],
                candidate_poisson[column],
            ])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
