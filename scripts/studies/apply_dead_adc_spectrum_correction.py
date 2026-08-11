#!/usr/bin/env python3
"""Reconstruct a dead top-half ADC block in completed NiO 1D spectra."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def estimate_gain(top, bottom, first: int, last: int, percentile: float, np) -> float:
    top_region = top[first:last]
    bottom_region = bottom[first:last]
    minimum = np.nanpercentile(bottom_region, percentile)
    selected = (
        np.isfinite(top_region)
        & np.isfinite(bottom_region)
        & (top_region > 0)
        & (bottom_region > minimum)
    )
    if np.count_nonzero(selected) < 16:
        raise ValueError("insufficient positive high-signal columns for gain calibration")
    return float(np.nanmedian(top_region[selected] / bottom_region[selected]))


def sideband_indices(left: tuple[int, int], right: tuple[int, int], np):
    return np.concatenate((np.arange(*left), np.arange(*right)))


def estimate_local_offset(top, bottom, gain: float, indices, np) -> tuple[float, float]:
    residual = top[indices] - gain * bottom[indices]
    offset = float(np.nanmedian(residual))
    mad = float(1.4826 * np.nanmedian(np.abs(residual - offset)))
    return offset, mad


def reconstruct(top, bottom, block: tuple[int, int], gain: float, offset: float, np):
    corrected_top = top.copy()
    first, last = block
    corrected_top[first:last] = gain * bottom[first:last] + offset
    return corrected_top


def cross_validate(top,
                   bottom,
                   gain: float,
                   block_width: int,
                   first_block: int,
                   last_block: int,
                   dead_block: tuple[int, int],
                   sideband_width: int,
                   np):
    errors = []
    for first in range(first_block, last_block, block_width):
        last = first + block_width
        if (first, last) == dead_block:
            continue
        left = (first - sideband_width, first)
        right = (last, last + sideband_width)
        indices = sideband_indices(left, right, np)
        offset, _ = estimate_local_offset(top, bottom, gain, indices, np)
        predicted = gain * bottom[first:last] + offset
        actual = top[first:last]
        errors.append(float(np.sqrt(np.nanmean(np.square(predicted - actual)))))
    return {
        "validation_block_count": len(errors),
        "validation_rmse_median": float(np.nanmedian(errors)),
        "validation_rmse_max": float(np.nanmax(errors)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dead-start", type=int, default=2272)
    parser.add_argument("--dead-width", type=int, default=16)
    parser.add_argument("--sideband-width", type=int, default=64)
    parser.add_argument("--gain-start", type=int, default=800)
    parser.add_argument("--gain-end", type=int, default=1600)
    parser.add_argument("--gain-percentile", type=float, default=60.0)
    parser.add_argument("--validation-start", type=int, default=2048)
    parser.add_argument("--validation-end", type=int, default=2512)
    return parser.parse_args()


def main():
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = json.loads((args.study_root / "manifest.json").read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    block = (args.dead_start, args.dead_start + args.dead_width)
    left = (block[0] - args.sideband_width, block[0])
    right = (block[1], block[1] + args.sideband_width)
    indices = sideband_indices(left, right, np)
    results = []

    for key, group in sorted(
        manifest["currents"].items(), key=lambda item: int(item[1]["current_pa"])
    ):
        source_path = args.study_root / "currents" / key / "spectrum" / "final_spectrum.h5"
        if not source_path.exists():
            continue
        with h5py.File(source_path, "r") as source:
            sums = source["full_columns_sum"][...].astype(np.float64)
            counts = source["full_columns_valid_count"][...].astype(np.float64)
        halves = safe_divide(sums, counts, np)
        top, bottom = halves
        gain = estimate_gain(
            top,
            bottom,
            args.gain_start,
            args.gain_end,
            args.gain_percentile,
            np,
        )
        offset, sideband_mad = estimate_local_offset(top, bottom, gain, indices, np)
        reconstructed_top = reconstruct(top, bottom, block, gain, offset, np)
        corrected_sums = sums.copy()
        corrected_sums[0, block[0]:block[1]] = (
            reconstructed_top[block[0]:block[1]] * counts[0, block[0]:block[1]]
        )
        before = safe_divide(sums.sum(axis=0), counts.sum(axis=0), np)
        after = safe_divide(corrected_sums.sum(axis=0), counts.sum(axis=0), np)
        validation = cross_validate(
            top,
            bottom,
            gain,
            args.dead_width,
            args.validation_start,
            args.validation_end,
            block,
            args.sideband_width,
            np,
        )
        block_uncertainty = float(
            np.sqrt(
                sideband_mad ** 2
                + np.nanvar(gain * bottom[block[0]:block[1]])
            )
        )
        metrics = {
            "current_key": key,
            "current_label": group["current_label"],
            "current_pa": int(group["current_pa"]),
            "top_bottom_gain": gain,
            "local_offset": offset,
            "sideband_residual_mad": sideband_mad,
            "reconstructed_block_uncertainty": block_uncertainty,
            "before_block_mean": float(np.nanmean(before[block[0]:block[1]])),
            "after_block_mean": float(np.nanmean(after[block[0]:block[1]])),
            "bottom_block_mean": float(np.nanmean(bottom[block[0]:block[1]])),
            "reconstructed_top_block_mean": float(
                np.nanmean(reconstructed_top[block[0]:block[1]])
            ),
            **validation,
        }
        results.append({
            "metrics": metrics,
            "top": top,
            "bottom": bottom,
            "reconstructed_top": reconstructed_top,
            "before": before,
            "after": after,
            "counts": counts,
            "corrected_sums": corrected_sums,
            "source_path": source_path,
        })

        current_output = args.output_root / "currents" / key
        current_output.mkdir(parents=True, exist_ok=True)
        with h5py.File(current_output / "dead_adc_corrected_spectrum.h5", "w") as output:
            output.create_dataset("top_mean_before", data=top)
            output.create_dataset("bottom_mean", data=bottom)
            output.create_dataset("top_mean_reconstructed", data=reconstructed_top)
            output.create_dataset("combined_mean_before", data=before)
            output.create_dataset("combined_mean_corrected", data=after)
            output.create_dataset("corrected_half_sums", data=corrected_sums)
            output.create_dataset("half_valid_counts", data=counts)
            output.attrs["source_spectrum"] = str(source_path)
            output.attrs["dead_columns"] = np.asarray(block, dtype=np.int32)
            output.attrs["gain_calibration_columns"] = np.asarray(
                (args.gain_start, args.gain_end), dtype=np.int32
            )
            output.attrs["offset_sidebands"] = np.asarray((*left, *right), dtype=np.int32)
            output.attrs["top_bottom_gain"] = gain
            output.attrs["local_offset"] = offset
            output.attrs["reconstruction_level"] = "1D spectrum after row aggregation"
            output.attrs["estimated_values"] = True
        (current_output / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    if not results:
        raise ValueError("no completed current spectra found")

    rows = [item["metrics"] for item in results]
    with (args.output_root / "dead_adc_correction_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subplot_rows = (len(results) + 1) // 2
    fig, axes = plt.subplots(
        subplot_rows,
        2,
        figsize=(17, 4.2 * subplot_rows),
        constrained_layout=True,
        squeeze=False,
    )
    x = np.arange(block[0] - 80, block[1] + 80)
    for axis, item in zip(axes.flat, results):
        label = item["metrics"]["current_label"]
        axis.plot(x, item["before"][x], color="0.25", linewidth=0.9, label="before")
        axis.plot(x, item["after"][x], color="tab:blue", linewidth=1.0, label="corrected")
        axis.plot(
            x,
            item["bottom"][x],
            color="tab:green",
            linewidth=0.7,
            alpha=0.7,
            label="bottom-half reference",
        )
        uncertainty = item["metrics"]["reconstructed_block_uncertainty"] / 2.0
        axis.fill_between(
            np.arange(*block),
            item["after"][block[0]:block[1]] - uncertainty,
            item["after"][block[0]:block[1]] + uncertainty,
            color="tab:blue",
            alpha=0.18,
            label="estimated uncertainty",
        )
        axis.axvspan(block[0], block[1] - 1, color="tab:orange", alpha=0.12)
        axis.set_title(
            f"{label}: gain={item['metrics']['top_bottom_gain']:.3f}, "
            f"offset={item['metrics']['local_offset']:.2f}"
        )
        axis.set_xlabel("Detector output column")
        axis.set_ylabel("Mean corrected detector value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    for axis in axes.flat[len(results):]:
        axis.set_visible(False)
    fig.suptitle("Dead top-half ADC block: spectrum before and after reconstruction")
    fig.savefig(args.output_root / "dead_adc_correction_before_after.png", dpi=180)
    plt.close(fig)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(results)))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for item, color in zip(results, colors):
        label = item["metrics"]["current_label"]
        axes[0].plot(
            np.ma.masked_less_equal(item["after"], 0.0),
            color=color,
            linewidth=0.75,
            label=label,
        )
        axes[1].plot(
            np.arange(block[0] - 96, block[1] + 96),
            item["after"][block[0] - 96:block[1] + 96],
            color=color,
            linewidth=0.85,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].axvline(768, color="black", linestyle="--", linewidth=0.7)
    axes[0].set_title("Dead-ADC-corrected spectra, logarithmic scale")
    axes[1].axvspan(block[0], block[1] - 1, color="tab:orange", alpha=0.12)
    axes[1].set_title("Corrected spectra around the reconstructed block")
    for axis in axes:
        axis.set_xlabel("Detector output column")
        axis.set_ylabel("Mean corrected detector value")
        axis.grid(alpha=0.2, which="both")
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(args.output_root / "comparison_corrected_spectra_log.png", dpi=180)
    plt.close(fig)

    summary = {
        "dead_columns": list(block),
        "affected_region": "top detector half",
        "method": (
            "Estimate top/bottom gain from high-signal CoreLoss columns; estimate a "
            "local offset from healthy sidebands; reconstruct the missing top-half "
            "1D spectral contribution from the measured bottom-half spectrum."
        ),
        "gain_columns": [args.gain_start, args.gain_end - 1],
        "offset_sidebands": [list(left), list(right)],
        "completed_currents": [row["current_key"] for row in rows],
    }
    (args.output_root / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
