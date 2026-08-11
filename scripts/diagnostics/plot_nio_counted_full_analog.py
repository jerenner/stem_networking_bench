#!/usr/bin/env python3
"""Replot a counting study with the full analog ZLP + CoreLoss reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RAW_ZLP_WIDTH = 768
FOLDED_ZLP_WIDTH = 192
RAW_TO_STITCHED_CORE_OFFSET = RAW_ZLP_WIDTH - FOLDED_ZLP_WIDTH
DEAD_ADC_RAW = (2272, 2288)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counting-dir", type=Path, required=True)
    parser.add_argument(
        "--analog-reference",
        type=Path,
        help="Final analog-spectrum CSV; defaults to counting_summary.json.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_reference(path: Path, counting_dir: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in counting_dir.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve analog reference '{path}'")


def load_analog_profile(path: Path, current_label: str, np):
    with path.open(newline="") as source:
        rows = csv.DictReader(source)
        values = [float(row[current_label]) for row in rows]
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counting_dir = args.counting_dir.resolve()
    summary = json.loads((counting_dir / "counting_summary.json").read_text())
    reference_arg = args.analog_reference or Path(summary["analog_reference"])
    analog_reference = resolve_reference(reference_arg, counting_dir)
    analog = load_analog_profile(analog_reference, summary["current_label"], np)

    with h5py.File(counting_dir / "counted_spectrum.h5", "r") as source:
        counts = source["total_core_counts"][:].astype(np.float64)
        stitched_core = source["stitched_core_columns"][:].astype(np.int64)
    counted_frames = int(summary["counting"]["frames"])
    counts /= counted_frames

    validity = summary["counting_validity"]
    boundary = int(validity["first_valid_stitched_column"])
    counts[stitched_core < boundary] = np.nan
    dead_stitched = (
        DEAD_ADC_RAW[0] - RAW_TO_STITCHED_CORE_OFFSET,
        DEAD_ADC_RAW[1] - RAW_TO_STITCHED_CORE_OFFSET,
    )
    counts[(stitched_core >= dead_stitched[0]) & (stitched_core < dead_stitched[1])] = np.nan

    zlp_peak = float(summary["energy_calibration"]["zlp_peak_channel"])
    dispersion = float(summary["energy_calibration"]["dispersion_ev_per_channel"])
    full_stitched = np.arange(analog.size)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True)
    axes[0].plot(
        full_stitched,
        np.ma.masked_less_equal(analog, 0),
        color="#6B7280",
        linewidth=0.8,
    )
    axes[0].set_yscale("log")
    axes[0].axvspan(0, FOLDED_ZLP_WIDTH, color="#F59E0B", alpha=0.12,
                    label="folded ZLP")
    axes[0].axvline(FOLDED_ZLP_WIDTH, color="#B45309", linestyle=":",
                    linewidth=1.0, label="ZLP/CoreLoss boundary")
    axes[0].set(
        ylabel="analog corrected intensity",
        title="Final analog spectrum (per-frame regional 3σ threshold; ZLP included)",
    )
    axes[0].legend()

    axes[1].plot(stitched_core, np.ma.masked_invalid(counts), color="#0F766E",
                 linewidth=0.9)
    axes[1].set_yscale("log")
    axes[1].axvspan(0, FOLDED_ZLP_WIDTH, facecolor="#E5E7EB", edgecolor="#9CA3AF",
                    alpha=0.5, hatch="//", label="ZLP not electron-counted")
    axes[1].axvline(boundary, color="#0F766E", linestyle="--",
                    label="counting-valid boundary")
    axes[1].axvspan(*dead_stitched, color="#DC2626", alpha=0.12,
                    label="excluded dead ADC block")
    axes[1].set(
        xlabel="stitched spectral channel",
        ylabel="counted electrons / frame",
        title="Mean STEMPy-counted CoreLoss spectrum (ZLP intentionally omitted)",
    )
    axes[1].set_xlim(0, analog.size - 1)
    axes[1].legend()
    secondary = axes[1].secondary_xaxis(
        "top",
        functions=(lambda x: (x - zlp_peak) * dispersion,
                   lambda e: e / dispersion + zlp_peak),
    )
    secondary.set_xlabel("provisional energy loss [eV]")

    output = args.output or counting_dir / "counted_spectrum_full_analog_zlp.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
