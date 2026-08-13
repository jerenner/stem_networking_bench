#!/usr/bin/env python3
"""Plot temporal dark distributions and thresholds for representative pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.studies.run_nio_counting_study import (
    corrected_batches,
    load_dark_products,
    neighboring_maximum,
)
from stem_analysis import ProcessorConfig


RAW_ZLP_WIDTH = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--bins", type=int, default=36)
    return parser.parse_args()


def detector_row(imaging_row: int) -> int:
    return imaging_row + 32 if imaging_row < 448 else imaging_row + 32


def collect_values(paths,
                   coordinates,
                   dark_frame,
                   valid_mask,
                   config,
                   args,
                   np):
    values = [[] for _ in coordinates]
    for core, metadata in corrected_batches(
        paths,
        dark_frame,
        valid_mask,
        config,
        args.tensor_frames,
        args.reader,
        np,
    ):
        for index, (row, column) in enumerate(coordinates):
            pixel_values = core[:, row, column]
            values[index].append(pixel_values[np.isfinite(pixel_values) & (pixel_values != 0)])
        print(
            f"  sampled {Path(metadata['source_path']).name} "
            f"{metadata['start_frame']}:{metadata['end_frame']}",
            flush=True,
        )
    return [np.concatenate(parts) if parts else np.empty(0) for parts in values]


def held_out_false_counts(paths,
                          threshold,
                          valid,
                          xray_threshold,
                          dark_frame,
                          valid_mask,
                          config,
                          args,
                          np):
    counts = np.zeros(threshold.shape, dtype=np.uint16)
    for core, metadata in corrected_batches(
        paths,
        dark_frame,
        valid_mask,
        config,
        args.tensor_frames,
        args.reader,
        np,
    ):
        for frame in core:
            events = (
                valid
                & (frame > threshold)
                & (frame < xray_threshold)
                & (frame > neighboring_maximum(frame, np))
            )
            counts += events
        print(
            f"  false-count scan {Path(metadata['source_path']).name} "
            f"{metadata['start_frame']}:{metadata['end_frame']}",
            flush=True,
        )
    return counts


def separated_top_pixels(false_counts, valid, count: int, np):
    candidates = np.argsort(false_counts.ravel())[::-1]
    selected = []
    for flat_index in candidates:
        row, column = np.unravel_index(flat_index, false_counts.shape)
        if not valid[row, column] or false_counts[row, column] == 0:
            continue
        if all(abs(row - old_row) > 4 or abs(column - old_column) > 4
               for old_row, old_column in selected):
            selected.append((int(row), int(column)))
        if len(selected) == count:
            break
    return selected


def percentile_pixel(sigma, valid, percentile: float, excluded, np):
    target = float(np.percentile(sigma[valid], percentile))
    distance = np.abs(sigma - target)
    distance[~valid] = np.inf
    for row, column in excluded:
        row_start, row_stop = max(0, row - 4), min(sigma.shape[0], row + 5)
        col_start, col_stop = max(0, column - 4), min(sigma.shape[1], column + 5)
        distance[row_start:row_stop, col_start:col_stop] = np.inf
    return tuple(int(value) for value in np.unravel_index(np.argmin(distance), sigma.shape))


def main() -> None:
    args = parse_args()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counting_dir = args.counting_dir.resolve()
    output = (args.output or counting_dir / "per_pixel_dark_threshold_examples.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((counting_dir / "counting_summary.json").read_text())
    config = ProcessorConfig(**summary["processing"])
    dark_frame, valid_mask = load_dark_products(Path(summary["dark_frame"]), np)

    with h5py.File(counting_dir / "per_pixel_thresholds.h5", "r") as source:
        mean = source["mean"][:]
        sigma = source["sigma"][:]
        threshold = source["threshold"][:]
        valid = source["valid"][:].astype(bool)
        xray_threshold = float(source.attrs["xray_threshold"])

    calibration_paths = [Path(path) for path in summary["calibration_dark_files"]]
    validation_paths = [Path(path) for path in summary["validation_dark_files"]]
    false_counts = held_out_false_counts(
        validation_paths,
        threshold,
        valid,
        xray_threshold,
        dark_frame,
        valid_mask,
        config,
        args,
        np,
    )

    false_pixels = separated_top_pixels(false_counts, valid, 3, np)
    representative = [
        percentile_pixel(sigma.copy(), valid, 10, false_pixels, np),
        percentile_pixel(sigma.copy(), valid, 50, false_pixels, np),
        percentile_pixel(sigma.copy(), valid, 90, false_pixels, np),
    ]
    coordinates = representative + false_pixels
    labels = ("low noise (P10)", "median noise (P50)", "high noise (P90)") + tuple(
        f"held-out false-count pixel #{index + 1}" for index in range(len(false_pixels))
    )

    print("Collecting calibration distributions...", flush=True)
    calibration_values = collect_values(
        calibration_paths,
        coordinates,
        dark_frame,
        valid_mask,
        config,
        args,
        np,
    )
    print("Collecting held-out distributions...", flush=True)
    validation_values = collect_values(
        validation_paths,
        coordinates,
        dark_frame,
        valid_mask,
        config,
        args,
        np,
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.facecolor": "white",
    })
    rows = 2
    columns = 3
    fig, axes = plt.subplots(rows, columns, figsize=(17, 9), constrained_layout=True)
    metadata = []
    for axis, label, coordinate, calibration, held_out in zip(
        axes.flat, labels, coordinates, calibration_values, validation_values
    ):
        row, column = coordinate
        center = float(mean[row, column])
        width = float(sigma[row, column])
        cutoff = float(threshold[row, column])
        combined = np.concatenate((calibration, held_out))
        low = min(float(np.min(combined)), center - 5.5 * width)
        high = max(float(np.max(combined)), cutoff + 1.0 * width)
        bins = np.linspace(low, high, args.bins + 1)
        axis.hist(
            calibration,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color="#93C5FD",
            alpha=0.55,
            label=f"calibration (n={calibration.size})",
        )
        axis.hist(
            held_out,
            bins=bins,
            density=True,
            histtype="step",
            color="#DC2626",
            linewidth=1.2,
            label=f"held-out dark (n={held_out.size})",
        )
        x = np.linspace(low, high, 500)
        gaussian = np.exp(-0.5 * np.square((x - center) / width)) / (
            width * np.sqrt(2.0 * np.pi)
        )
        axis.plot(x, gaussian, color="#1D4ED8", linewidth=1.0, label="fitted Gaussian")
        axis.axvline(center, color="#1D4ED8", linestyle=":", linewidth=1.0,
                     label=f"mean={center:.1f}")
        axis.axvline(cutoff, color="#B45309", linestyle="--", linewidth=1.3,
                     label=f"threshold={cutoff:.1f}")
        held_out_above = int(np.count_nonzero(held_out > cutoff))
        raw_row = detector_row(row)
        raw_column = column + RAW_ZLP_WIDTH
        axis.set(
            title=(
                f"{label}\nraw pixel ({raw_row}, {raw_column})\n"
                f"sigma={width:.1f}; held-out false maxima={int(false_counts[row, column])}"
            ),
            xlabel="corrected detector value",
            ylabel="probability density",
        )
        axis.title.set_fontsize(10)
        axis.legend(fontsize=7)
        metadata.append({
            "label": label,
            "imaging_row": row,
            "coreloss_column": column,
            "raw_detector_row": raw_row,
            "raw_detector_column": raw_column,
            "mean": center,
            "sigma": width,
            "threshold": cutoff,
            "calibration_samples": int(calibration.size),
            "held_out_samples": int(held_out.size),
            "held_out_samples_above_threshold": held_out_above,
            "held_out_strict_local_maximum_false_counts": int(false_counts[row, column]),
        })
    fig.suptitle(
        "Per-pixel dark distributions and fixed 4.5-sigma thresholds\n"
        "Blue: dark files 1-3 calibration; red: independently held-out dark file 4",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps({
        "counting_dir": str(counting_dir),
        "output": str(output),
        "threshold_definition": summary["threshold"]["definition"],
        "sigma_multiplier": summary["threshold"]["sigma_multiplier"],
        "pixels": metadata,
    }, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
