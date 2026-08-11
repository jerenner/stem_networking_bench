#!/usr/bin/env python3
"""Diagnose fixed-column and high-loss features in the NiO current study outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def group_response(profile, first: int, last: int, np) -> float:
    center = np.nanmean(profile[first:last])
    neighbors = np.nanmean(
        [np.nanmean(profile[first - (last - first):first]),
         np.nanmean(profile[last:last + (last - first)])]
    )
    return float(center / neighbors)


def normalized_tail(profile, pre_slice: slice, np):
    return profile / np.nanmean(profile[pre_slice])


def smooth_profile(profile, np, width: int = 16):
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    return np.convolve(profile, kernel, mode="same")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dip-start", type=int, default=2272)
    parser.add_argument("--dip-width", type=int, default=16)
    parser.add_argument("--tail-pre-start", type=int, default=3344)
    parser.add_argument("--tail-pre-end", type=int, default=3424)
    parser.add_argument("--tail-post-start", type=int, default=3456)
    parser.add_argument("--tail-post-end", type=int, default=3520)
    return parser.parse_args()


def main():
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = json.loads((args.study_root / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for key, group in sorted(
        manifest["currents"].items(), key=lambda item: int(item[1]["current_pa"])
    ):
        current_dir = args.study_root / "currents" / key
        spectrum_path = current_dir / "spectrum" / "final_spectrum.h5"
        dark_path = current_dir / "dark" / "dark_frame.h5"
        if not spectrum_path.exists() or not dark_path.exists():
            continue
        with h5py.File(spectrum_path, "r") as h5:
            spectra = h5["full_columns_static_mask_only_mean"][...]
            per_file_sums = h5["per_file_full_columns_sum"][...]
            per_file_counts = h5["per_file_full_columns_valid_count"][...]
        with h5py.File(dark_path, "r") as h5:
            dark_stddev = np.squeeze(h5["dark_stddev"][...])
            valid_mask = np.squeeze(h5["valid_pixel_mask"][...]).astype(bool)

        per_file = safe_divide(
            per_file_sums.sum(axis=1), per_file_counts.sum(axis=1), np
        )
        entries.append({
            "key": key,
            "label": group["current_label"],
            "current_pa": int(group["current_pa"]),
            "spectra": spectra,
            "per_file": per_file,
            "dark_stddev": dark_stddev,
            "valid_mask": valid_mask,
            "current_dir": current_dir,
        })

    if not entries:
        raise ValueError("no completed current-study outputs found")

    dip_start = args.dip_start
    dip_end = dip_start + args.dip_width
    detector_regions = (slice(32, 480), slice(480, 928))
    metrics = []
    for entry in entries:
        row = {
            "current_key": entry["key"],
            "current_label": entry["label"],
            "current_pa": entry["current_pa"],
        }
        for half, name in enumerate(("top", "bottom")):
            profile = entry["spectra"][half]
            std_profile = np.nanmean(entry["dark_stddev"][detector_regions[half]], axis=0)
            row[f"dip_response_{name}"] = group_response(
                profile, dip_start, dip_end, np
            )
            row[f"dark_stddev_response_{name}"] = group_response(
                std_profile, dip_start, dip_end, np
            )
            row[f"invalid_fraction_{name}"] = float(
                np.mean(~entry["valid_mask"][detector_regions[half], dip_start:dip_end])
            )
            row[f"tail_delta_{name}"] = float(
                np.nanmean(profile[args.tail_post_start:args.tail_post_end])
                - np.nanmean(profile[args.tail_pre_start:args.tail_pre_end])
            )
        metrics.append(row)

    with (args.output_dir / "coreloss_feature_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    reference = min(entries, key=lambda item: abs(item["current_pa"] - 500))
    x_dip = np.arange(2200, 2360)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for half, (name, color) in enumerate((("top", "tab:red"), ("bottom", "tab:blue"))):
        profile = reference["spectra"][half]
        local_scale = np.nanmean(
            np.concatenate((profile[dip_start - 16:dip_start], profile[dip_end:dip_end + 16]))
        )
        axes[0, 0].plot(x_dip, profile[x_dip] / local_scale, color=color, label=name)
        std_profile = np.nanmean(reference["dark_stddev"][detector_regions[half]], axis=0)
        std_scale = np.nanmean(
            np.concatenate((std_profile[dip_start - 16:dip_start], std_profile[dip_end:dip_end + 16]))
        )
        axes[0, 1].plot(x_dip, std_profile[x_dip] / std_scale, color=color, label=name)
    for axis in axes[0]:
        axis.axvspan(dip_start, dip_end - 1, color="tab:orange", alpha=0.18)
        axis.axhline(1.0, color="0.4", linestyle="--", linewidth=0.7)
        axis.set_xlabel("Detector output column")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0, 0].set_title(f"{reference['label']} corrected response around the 16-column dip")
    axes[0, 0].set_ylabel("Response / neighboring-block response")
    axes[0, 1].set_title("Matching dark temporal standard deviation")
    axes[0, 1].set_ylabel("Stddev / neighboring-block stddev")

    labels = [entry["label"] for entry in entries]
    positions = np.arange(len(entries))
    width = 0.36
    axes[1, 0].bar(
        positions - width / 2,
        [row["dip_response_top"] for row in metrics],
        width,
        label="top",
        color="tab:red",
    )
    axes[1, 0].bar(
        positions + width / 2,
        [row["dip_response_bottom"] for row in metrics],
        width,
        label="bottom",
        color="tab:blue",
    )
    axes[1, 1].bar(
        positions - width / 2,
        [row["dark_stddev_response_top"] for row in metrics],
        width,
        label="top",
        color="tab:red",
    )
    axes[1, 1].bar(
        positions + width / 2,
        [row["dark_stddev_response_bottom"] for row in metrics],
        width,
        label="bottom",
        color="tab:blue",
    )
    for axis in axes[1]:
        axis.axhline(1.0, color="0.4", linestyle="--", linewidth=0.7)
        axis.set_xticks(positions, labels, rotation=30)
        axis.set_ylabel("Affected block / neighboring blocks")
        axis.grid(alpha=0.2, axis="y")
        axis.legend()
    axes[1, 0].set_title("Illuminated response across beam currents")
    axes[1, 1].set_title("Dark temporal response across beam currents")
    fig.suptitle(f"CoreLoss dip diagnosis: detector columns {dip_start}..{dip_end - 1}")
    fig.savefig(args.output_dir / "dip_2272_diagnostics.png", dpi=180)
    plt.close(fig)

    selected = [
        entry for entry in entries
        if entry["current_pa"] >= 130 and entry["current_pa"] != 250
    ]
    x_tail = np.arange(3150, 3820)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    pre_slice = slice(args.tail_pre_start, args.tail_pre_end)
    for entry, color in zip(selected, colors):
        combined = smooth_profile(np.nanmean(entry["spectra"], axis=0), np)
        axes[0, 0].plot(
            x_tail,
            normalized_tail(combined, pre_slice, np)[x_tail],
            color=color,
            label=entry["label"],
        )
        for half, linestyle in enumerate(("-", "--")):
            axes[0, 1].plot(
                x_tail,
                normalized_tail(
                    smooth_profile(entry["spectra"][half], np), pre_slice, np
                )[x_tail],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                label=f"{entry['label']} {'top' if half == 0 else 'bottom'}",
            )
    for file_index, profile in enumerate(reference["per_file"]):
        axes[1, 0].plot(
            x_tail,
            normalized_tail(smooth_profile(profile, np), pre_slice, np)[x_tail],
            linewidth=0.8,
            label=f"source file {file_index + 1}",
        )

    blr_path = reference["current_dir"] / "blr_comparison" / "blr_comparison.h5"
    if blr_path.exists():
        with h5py.File(blr_path, "r") as h5:
            blr_spectra = h5["spectra"][...]
        for profile, label, color in zip(
            blr_spectra,
            ("no BLR", "grouped BLR", "columnwise BLR"),
            ("0.25", "tab:red", "tab:blue"),
        ):
            axes[1, 1].plot(
                x_tail,
                normalized_tail(smooth_profile(profile, np), pre_slice, np)[x_tail],
                color=color,
                linewidth=0.8,
                label=label,
            )

    for axis in axes.flat:
        axis.axvspan(
            args.tail_pre_start,
            args.tail_pre_end - 1,
            color="0.6",
            alpha=0.1,
            label="pre window" if axis is axes[0, 0] else None,
        )
        axis.axvspan(
            args.tail_post_start,
            args.tail_post_end - 1,
            color="tab:orange",
            alpha=0.12,
            label="post window" if axis is axes[0, 0] else None,
        )
        axis.axhline(1.0, color="0.4", linestyle="--", linewidth=0.7)
        axis.set_xlabel("Detector output column")
        axis.set_ylabel("Spectrum / pre-window mean")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    axes[0, 0].set_title("High-current combined spectra")
    axes[0, 1].set_title("Top/bottom detector-half coherence")
    axes[1, 0].set_title(f"{reference['label']} source-file repeatability")
    axes[1, 1].set_title(f"{reference['label']} sensitivity to BLR strategy")
    fig.suptitle("High-loss rise near detector column 3500 (16-column display average)")
    fig.savefig(args.output_dir / "tail_3500_diagnostics.png", dpi=180)
    plt.close(fig)

    if blr_path.exists():
        no_blr = smooth_profile(blr_spectra[0], np)
        grouped = smooth_profile(blr_spectra[1], np)
        columnwise = smooth_profile(blr_spectra[2], np)
        inferred_baseline = no_blr - grouped
        fig, axes = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
        axes[0].plot(x_tail, no_blr[x_tail], color="0.25", linewidth=1.0)
        axes[0].set_title(f"{reference['label']} dark-subtracted imaging spectrum before BLR")
        axes[1].plot(x_tail, inferred_baseline[x_tail], color="tab:orange", linewidth=1.0)
        axes[1].set_title("Inferred grouped BLR estimate (no BLR minus grouped-BLR result)")
        axes[2].plot(x_tail, grouped[x_tail], color="tab:red", linewidth=1.0, label="grouped BLR")
        axes[2].plot(
            x_tail,
            columnwise[x_tail],
            color="tab:blue",
            linewidth=0.9,
            label="columnwise BLR",
        )
        axes[2].set_title("Corrected spectrum after subtracting the BLR estimate")
        for axis in axes:
            axis.axvspan(args.tail_pre_start, args.tail_pre_end - 1, color="0.6", alpha=0.1)
            axis.axvspan(
                args.tail_post_start,
                args.tail_post_end - 1,
                color="tab:orange",
                alpha=0.12,
            )
            axis.set_xlabel("Detector output column")
            axis.set_ylabel("Mean detector value")
            axis.grid(alpha=0.2)
        axes[2].legend()
        fig.suptitle("BLR decomposition of the apparent high-loss rise (16-column display average)")
        fig.savefig(args.output_dir / "tail_3500_blr_decomposition.png", dpi=180)
        plt.close(fig)

    summary = {
        "dip_columns": [dip_start, dip_end - 1],
        "dip_interpretation": (
            "Stable top-half low-response 16-column detector/readout block; "
            "not caused by grouped BLR or the current high-variance blinker mask."
        ),
        "tail_windows": {
            "pre": [args.tail_pre_start, args.tail_pre_end - 1],
            "post": [args.tail_post_start, args.tail_post_end - 1],
        },
        "tail_interpretation": (
            "The broad onset near columns 3450..3460 is present before BLR, appears in "
            "both detector halves, strengthens at high beam current, and has no matching "
            "dark-noise feature. Grouped and columnwise BLR preserve nearly the same "
            "onset, although BLR changes its contrast. This favors a real sample "
            "core-loss feature over a dead-column or BLR-grouping artifact. The DM4 "
            "axes contain no energy calibration, so a specific edge assignment remains "
            "provisional."
        ),
    }
    (args.output_dir / "coreloss_feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
