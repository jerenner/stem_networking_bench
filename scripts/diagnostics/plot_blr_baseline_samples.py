#!/usr/bin/env python3
"""Plot effective BLR baseline samples reconstructed from a BLR comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_nio_processing_analysis import configure_matplotlib_cache


def phase_statistics(profile, group_width, np):
    usable = profile[: profile.size // group_width * group_width]
    groups = usable.reshape(-1, group_width)
    residuals = groups - groups.mean(axis=1, keepdims=True)
    return residuals.mean(axis=0), residuals.std(axis=0), residuals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_h5", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label",
        default="Dataset",
        help="Dataset label used in plot titles (for example, '500pA').",
    )
    return parser.parse_args()


def plot_grouped_influence(output_path,
                           x,
                           no_blr,
                           grouped,
                           region_title,
                           xlabel,
                           zoom,
                           plt,
                           np):
    baseline = no_blr - grouped
    applied_correction = grouped - no_blr
    selections = (slice(None), slice(*zoom))
    titles = ("full region", f"zoom: columns {zoom[0]}..{zoom[1] - 1}")
    fig, axes = plt.subplots(
        3, 2, figsize=(18, 11), sharex="col", constrained_layout=True
    )
    for column, (selection, subtitle) in enumerate(zip(selections, titles)):
        selected_x = x[selection]
        axes[0, column].plot(
            selected_x, no_blr[selection], color="black", linewidth=0.9,
            label="Dark-subtracted spectrum, no BLR"
        )
        axes[0, column].plot(
            selected_x, grouped[selection], color="tab:red", linewidth=0.9,
            label="After grouped BLR"
        )
        axes[0, column].plot(
            selected_x, applied_correction[selection], color="tab:blue",
            linewidth=0.75, alpha=0.8, label="BLR contribution to spectrum (-B)"
        )
        axes[0, column].set_title(f"{region_title}: {subtitle}")
        axes[0, column].set_ylabel("Detector value")

        axes[1, column].plot(
            selected_x, baseline[selection], color="tab:orange", linewidth=0.9
        )
        axes[1, column].axhline(0.0, color="gray", linewidth=0.7)
        axes[1, column].set_ylabel("Baseline B")
        axes[1, column].set_title("Grouped edge-row baseline estimate removed")

        axes[2, column].plot(
            selected_x, applied_correction[selection], color="tab:blue",
            linewidth=0.9, label="Measured grouped-BLR minus no-BLR"
        )
        axes[2, column].plot(
            selected_x, -baseline[selection], color="black", linestyle="--",
            linewidth=0.7, label="Expected change: -B"
        )
        axes[2, column].axhline(0.0, color="gray", linewidth=0.7)
        axes[2, column].set_ylabel("Change in spectrum")
        axes[2, column].set_title("Applied BLR contribution (grouped - no BLR)")
        axes[2, column].set_xlabel(xlabel)

        for row in range(3):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_full_region_overlays(output_path, folded, spectra, label, plt, np):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    regions = (
        (
            axes[0],
            np.arange(folded.shape[-1]),
            folded[0],
            folded[1],
            f"{label} folded ZLP: grouped-BLR influence",
            "Physical ZLP detector column modulo 192",
        ),
        (
            axes[1],
            np.arange(spectra.shape[-1] - 768),
            spectra[0, 768:],
            spectra[1, 768:],
            f"{label} CoreLoss: grouped-BLR influence",
            "CoreLoss-relative detector column",
        ),
    )
    for axis, x, no_blr, grouped, title, xlabel in regions:
        applied_correction = grouped - no_blr
        axis.plot(
            x,
            no_blr,
            color="black",
            linewidth=0.9,
            label="Dark-subtracted spectrum, no BLR",
        )
        axis.plot(
            x,
            grouped,
            color="tab:red",
            linewidth=0.9,
            label="After grouped BLR",
        )
        axis.plot(
            x,
            applied_correction,
            color="tab:blue",
            linewidth=0.75,
            alpha=0.8,
            label="BLR contribution to spectrum (-B)",
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Detector value")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.comparison_h5, "r") as comparison:
        spectra = comparison["spectra"][...]
        folded = comparison["folded_zlp"][...]

    # All paths use the same data and static mask, so their difference is the
    # effective top/bottom-combined baseline removed by the selected BLR path.
    zlp_columnwise = folded[0] - folded[2]
    zlp_grouped = folded[0] - folded[1]
    core_columnwise = spectra[0, 768:] - spectra[2, 768:]
    core_grouped = spectra[0, 768:] - spectra[1, 768:]

    zlp_phase, zlp_phase_std, zlp_residuals = phase_statistics(
        zlp_columnwise, 4, np
    )
    core_phase, core_phase_std, core_residuals = phase_statistics(
        core_columnwise, 16, np
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].plot(zlp_columnwise, linewidth=0.9, label="per-column edge-row mean")
    axes[0, 0].plot(zlp_grouped, linewidth=1.1, label="4-column grouped estimate")
    axes[0, 0].set_title("Effective folded-ZLP BLR baseline")
    axes[0, 0].set_xlabel("Physical ZLP detector column modulo 192")
    axes[0, 0].set_ylabel("Baseline removed from imaging pixels")

    axes[0, 1].errorbar(
        np.arange(4), zlp_phase, yerr=zlp_phase_std, marker="o", capsize=3
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Mean ZLP baseline residual by phase modulo 4")
    axes[0, 1].set_xlabel("Column position inside each 4-column group")
    axes[0, 1].set_ylabel("Baseline minus its group mean")
    axes[0, 1].set_xticks(np.arange(4))

    zoom = min(512, core_columnwise.size)
    axes[1, 0].plot(
        core_columnwise[:zoom], linewidth=0.8, label="per-column edge-row mean"
    )
    axes[1, 0].plot(
        core_grouped[:zoom], linewidth=1.0, label="16-column grouped estimate"
    )
    for boundary in range(16, zoom, 16):
        axes[1, 0].axvline(boundary, color="gray", alpha=0.12, linewidth=0.6)
    axes[1, 0].set_title(f"Effective CoreLoss BLR baseline, first {zoom} columns")
    axes[1, 0].set_xlabel("CoreLoss-relative detector column")
    axes[1, 0].set_ylabel("Baseline removed from imaging pixels")

    axes[1, 1].errorbar(
        np.arange(16), core_phase, yerr=core_phase_std, marker="o", capsize=3
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Mean CoreLoss baseline residual by phase modulo 16")
    axes[1, 1].set_xlabel("Column position inside each 16-column group")
    axes[1, 1].set_ylabel("Baseline minus its group mean")
    axes[1, 1].set_xticks(np.arange(16))

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.savefig(args.output_dir / "blr_baseline_samples.png", dpi=180)
    plt.close(fig)

    plot_grouped_influence(
        args.output_dir / "zlp_grouped_blr_influence.png",
        np.arange(folded.shape[-1]),
        folded[0],
        folded[1],
        f"{args.label} folded ZLP",
        "Physical ZLP detector column modulo 192",
        (72, 192),
        plt,
        np,
    )
    plot_grouped_influence(
        args.output_dir / "coreloss_grouped_blr_influence.png",
        np.arange(spectra.shape[-1] - 768),
        spectra[0, 768:],
        spectra[1, 768:],
        f"{args.label} CoreLoss",
        "CoreLoss-relative detector column",
        (0, 512),
        plt,
        np,
    )
    plot_full_region_overlays(
        args.output_dir / "grouped_blr_full_region_overlay.png",
        folded,
        spectra,
        args.label,
        plt,
        np,
    )

    summary = {
        "comparison_h5": str(args.comparison_h5),
        "dataset_label": args.label,
        "interpretation": (
            "Path differences reconstruct the effective detector-half-combined "
            "edge-row baseline because all paths use identical input and static masks."
        ),
        "zlp_phase_modulo_4_mean": zlp_phase.tolist(),
        "zlp_phase_modulo_4_stddev": zlp_phase_std.tolist(),
        "zlp_within_group_residual_rms": float(np.sqrt(np.mean(zlp_residuals**2))),
        "core_phase_modulo_16_mean": core_phase.tolist(),
        "core_phase_modulo_16_stddev": core_phase_std.tolist(),
        "core_within_group_residual_rms": float(np.sqrt(np.mean(core_residuals**2))),
        "influence_plots": [
            "zlp_grouped_blr_influence.png",
            "coreloss_grouped_blr_influence.png",
            "grouped_blr_full_region_overlay.png",
        ],
    }
    (args.output_dir / "blr_baseline_samples.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
