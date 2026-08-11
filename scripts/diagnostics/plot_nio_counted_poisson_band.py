#!/usr/bin/env python3
"""Plot a counted EELS tail against a local Poisson fluctuation band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RAW_ZLP_WIDTH = 768
RAW_TO_STITCHED_CORE_OFFSET = 576
DEAD_ADC_RAW = (2272, 2288)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bin-columns", type=int, default=16)
    parser.add_argument(
        "--smoothing-window-bins",
        type=int,
        default=5,
        help="Odd Savitzky-Golay window used to estimate the local expected spectrum.",
    )
    parser.add_argument("--smoothing-polynomial-order", type=int, default=2)
    return parser.parse_args()


def binned_counts(counts, raw_start: int, bin_columns: int, np):
    core_start = raw_start - RAW_ZLP_WIDTH
    starts_core = np.arange(core_start, counts.size, bin_columns)
    stops_core = np.minimum(starts_core + bin_columns, counts.size)
    values = np.asarray(
        [counts[start:stop].sum() for start, stop in zip(starts_core, stops_core)],
        dtype=np.float64,
    )
    starts_raw = starts_core + RAW_ZLP_WIDTH
    stops_raw = stops_core + RAW_ZLP_WIDTH
    centers_raw = (starts_raw + stops_raw - 1) / 2.0
    dead = (starts_raw < DEAD_ADC_RAW[1]) & (stops_raw > DEAD_ADC_RAW[0])
    return starts_raw, stops_raw, centers_raw, values, dead


def smooth_poisson_expectation(values, valid, window: int, order: int, np, signal):
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    if window <= order:
        raise ValueError("smoothing window must exceed polynomial order")
    if window > values.size:
        raise ValueError("smoothing window exceeds the number of tail bins")

    indices = np.arange(values.size)
    log_values = np.log(np.maximum(values, 1.0))
    interpolated = np.interp(indices, indices[valid], log_values[valid])
    smooth_log = signal.savgol_filter(
        interpolated, window_length=window, polyorder=order, mode="interp"
    )
    return np.exp(smooth_log)


def main() -> None:
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    counting_dir = args.counting_dir.resolve()
    output = args.output or counting_dir / "counted_tail_local_poisson_band.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads((counting_dir / "counting_summary.json").read_text())
    with h5py.File(counting_dir / "counted_spectrum.h5", "r") as source:
        counts = source["total_core_counts"][:].astype(np.float64)
    counted_frames = int(summary["counting"]["frames"])

    raw_start = int(summary["energy_calibration"]["tail_raw_channel"])
    _, _, centers_raw, observed, dead = binned_counts(
        counts, raw_start, args.bin_columns, np
    )
    valid = ~dead & np.isfinite(observed) & (observed > 0)
    expected = smooth_poisson_expectation(
        observed,
        valid,
        args.smoothing_window_bins,
        args.smoothing_polynomial_order,
        np,
        signal,
    )

    stitched = centers_raw - RAW_TO_STITCHED_CORE_OFFSET
    zlp_peak = summary["energy_calibration"]["zlp_peak_channel"]
    dispersion = summary["energy_calibration"]["dispersion_ev_per_channel"]
    energy = (stitched - zlp_peak) * dispersion
    sigma = np.sqrt(expected)
    observed_per_frame = observed / counted_frames
    expected_per_frame = expected / counted_frames
    lower_per_frame = np.maximum(
        (expected - sigma) / counted_frames,
        np.finfo(np.float64).tiny,
    )
    upper_per_frame = (expected + sigma) / counted_frames
    ratio = observed / expected
    relative_sigma = 1.0 / np.sqrt(expected)
    standardized = (observed - expected) / sigma

    inside_one = float(np.mean(np.abs(standardized[valid]) <= 1.0))
    inside_two = float(np.mean(np.abs(standardized[valid]) <= 2.0))
    reduced_chi_square = float(np.mean(np.square(standardized[valid])))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)

    plot_observed = observed_per_frame.copy()
    plot_observed[dead] = np.nan
    axes[0].fill_between(
        energy, lower_per_frame, upper_per_frame, color="#93C5FD", alpha=0.55,
        label=r"$(\mu_\mathrm{total} \pm \sqrt{\mu_\mathrm{total}}) / N_\mathrm{frames}$",
    )
    axes[0].plot(energy, expected_per_frame, color="#1D4ED8", linewidth=1.2,
                 label="smoothed local expectation")
    axes[0].plot(energy, plot_observed, "o-", color="#0F766E", markersize=3,
                 linewidth=0.8, label="counted spectrum")
    axes[0].set_yscale("log")
    axes[0].set(
        ylabel=f"counted electrons / frame / {args.bin_columns}-channel bin",
        title=(
            f"15 pA mean counted EELS tail with one-sigma Poisson band "
            f"({counted_frames:,} frames)"
        ),
    )
    axes[0].legend()

    ratio_plot = ratio.copy()
    ratio_plot[dead] = np.nan
    axes[1].fill_between(
        energy, 1.0 - relative_sigma, 1.0 + relative_sigma,
        color="#93C5FD", alpha=0.55, label=r"$1 \pm 1/\sqrt{\mu}$",
    )
    axes[1].axhline(1.0, color="#1D4ED8", linewidth=1.0)
    axes[1].plot(energy, ratio_plot, "o-", color="#0F766E", markersize=3,
                 linewidth=0.8, label=r"observed / local expectation")
    axes[1].set(
        xlabel="provisional energy loss [eV]", ylabel=r"$N/\mu$",
        title=(
            f"Observed local fluctuations: {inside_one:.1%} within 1σ, "
            f"{inside_two:.1%} within 2σ; mean squared standardized residual "
            f"={reduced_chi_square:.2f}"
        ),
    )
    axes[1].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    metadata = {
        "counting_dir": str(counting_dir),
        "output": str(output),
        "counted_frames": counted_frames,
        "count_normalization": "counted electrons per input frame",
        "bin_columns": args.bin_columns,
        "smoothing_window_bins": args.smoothing_window_bins,
        "smoothing_polynomial_order": args.smoothing_polynomial_order,
        "fraction_within_one_sigma": inside_one,
        "fraction_within_two_sigma": inside_two,
        "mean_squared_standardized_residual": reduced_chi_square,
        "interpretation": (
            "Visual local-smoothness diagnostic; the expectation is estimated from the "
            "same spectrum and is not an independent Poisson test."
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
