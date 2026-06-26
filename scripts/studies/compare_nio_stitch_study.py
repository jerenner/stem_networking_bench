#!/usr/bin/env python3
"""Fit and compare empirical ZLP/CoreLoss stitch factors across beam currents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plot_nio_processing_analysis import configure_matplotlib_cache


MODE_LABELS = {
    "no_blr": "No BLR",
    "grouped_blr": "Grouped BLR",
}
IMAGEJ_GAIN = 16000.0 / 6500.0 * 1.12
FIT_CONFIGURATIONS = (
    ("narrow_linear", 170, 191, 194, 214, 1),
    ("narrow_quadratic", 170, 191, 194, 214, 2),
    ("default_quadratic", 160, 191, 194, 224, 2),
    ("wide_quadratic", 145, 191, 194, 240, 2),
    ("wide_cubic", 145, 191, 194, 240, 3),
)
HERMITE_LEFT_ENDPOINT = 190
HERMITE_RIGHT_ENDPOINT = 194
HERMITE_FILL_COLUMNS = (191, 192, 193)
HERMITE_LEFT_SLOPE_WINDOW = (184, 190)
HERMITE_RIGHT_SLOPE_WINDOW = (194, 200)


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def estimate_linear_slope(profile, start, end, np):
    x = np.arange(start, end + 1, dtype=np.float64)
    y = profile[start:end + 1].astype(np.float64)
    valid = np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return 0.0
    x = x[valid]
    y = y[valid]
    x_centered = x - x.mean()
    denominator = np.sum(x_centered * x_centered)
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def apply_hermite_boundary_repair(profile, np):
    repaired = profile.copy()
    left = HERMITE_LEFT_ENDPOINT
    right = HERMITE_RIGHT_ENDPOINT
    y0 = float(repaired[left])
    y1 = float(repaired[right])
    if not np.isfinite(y0) or not np.isfinite(y1):
        return repaired, {
            "hermite_left_slope": float("nan"),
            "hermite_right_slope": float("nan"),
            "hermite_factor_191": float("nan"),
            "hermite_factor_192": float("nan"),
            "hermite_factor_193": float("nan"),
        }

    left_slope = estimate_linear_slope(
        repaired, HERMITE_LEFT_SLOPE_WINDOW[0], HERMITE_LEFT_SLOPE_WINDOW[1], np
    )
    right_slope = estimate_linear_slope(
        repaired, HERMITE_RIGHT_SLOPE_WINDOW[0], HERMITE_RIGHT_SLOPE_WINDOW[1], np
    )
    span = float(right - left)
    factors = {}
    for column in HERMITE_FILL_COLUMNS:
        observed = float(repaired[column])
        t = (column - left) / span
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        repaired[column] = (
            h00 * y0
            + h10 * span * left_slope
            + h01 * y1
            + h11 * span * right_slope
        )
        factors[f"hermite_factor_{column}"] = (
            float(repaired[column] / observed)
            if np.isfinite(observed) and abs(observed) > 1e-12
            else float("nan")
        )

    return repaired, {
        "hermite_left_slope": left_slope,
        "hermite_right_slope": right_slope,
        **factors,
    }


def apply_stitch_calibration(profile, gain, np):
    stitched = profile.copy()
    stitched[:192] *= gain
    repaired, metrics = apply_hermite_boundary_repair(stitched, np)
    return repaired, metrics


def collapsed_profile(sums, counts, np):
    """Return [summed four-read ZLP (192), CoreLoss (3072)]."""
    if sums.ndim == 2:
        sums = sums.sum(axis=0)
        counts = counts.sum(axis=0)
    detector_profile = safe_divide(sums, counts, np)
    zlp = detector_profile[:768].reshape(4, 192).sum(axis=0)
    return np.concatenate([zlp, detector_profile[768:]])


def robust_fit(profile,
               np,
               left_start=160,
               left_end=191,
               right_start=194,
               right_end=224,
               degree=2,
               huber_delta=1.345):
    left_x = np.arange(left_start, left_end)
    right_x = np.arange(right_start, right_end)
    x = np.concatenate([left_x, right_x])
    y = profile[x]
    valid = np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.size < degree + 4 or not np.any(x < 192) or not np.any(x >= 192):
        raise ValueError("insufficient positive samples for stitch fit")

    center = 191.5
    scale_x = max(right_end - left_start, 1)
    normalized_x = (x - center) / scale_x
    polynomial = np.column_stack([normalized_x**power for power in range(degree + 1)])
    left_indicator = (x < 192).astype(np.float64)
    design = np.column_stack([polynomial, -left_indicator])
    target = np.log(y)
    weights = np.ones_like(target)
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(30):
        residual = target - design @ beta
        robust_scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if robust_scale <= 1e-12:
            break
        cutoff = huber_delta * robust_scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(
            design * root_weights[:, None], target * root_weights, rcond=None
        )[0]
        if np.max(np.abs(updated - beta)) < 1e-10:
            beta = updated
            break
        beta = updated

    log_gain = float(beta[-1])
    gain = float(np.exp(log_gain))
    residual = target - design @ beta
    corrected = profile.copy()
    corrected[:192] *= gain

    def predict(columns):
        normalized = (np.asarray(columns) - center) / scale_x
        basis = np.column_stack([
            normalized**power for power in range(degree + 1)
        ])
        return np.exp(basis @ beta[: degree + 1])

    boundary_columns = np.array([191, 192, 193])
    predicted_boundary = predict(boundary_columns)
    observed_boundary = corrected[boundary_columns]
    boundary_factors = safe_divide(predicted_boundary, observed_boundary, np)
    hermite_corrected, hermite_metrics = apply_hermite_boundary_repair(corrected, np)
    direct_ratio_before = float(profile[192] / profile[191])
    direct_ratio_after = float(corrected[192] / corrected[191])
    return {
        "gain": gain,
        "log_gain": log_gain,
        "fit_log_rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
        "fit_log_mad": float(np.median(np.abs(residual))),
        "fit_point_count": int(x.size),
        "positive_fraction": float(valid.mean()),
        "direct_core0_to_zlp191_before": direct_ratio_before,
        "direct_core0_to_zlp191_after": direct_ratio_after,
        "boundary_factor_191": float(boundary_factors[0]),
        "boundary_factor_192": float(boundary_factors[1]),
        "boundary_factor_193": float(boundary_factors[2]),
        **hermite_metrics,
        "corrected": corrected,
        "hermite_corrected": hermite_corrected,
        "predict": predict,
    }


def load_current(current_dir, h5py, np):
    metadata = json.loads((current_dir / "current_metadata.json").read_text())
    with h5py.File(current_dir / "stitch_spectra.h5", "r") as source:
        mode_names = json.loads(source.attrs["mode_names"])
        data = {
            "per_file_sums": source["per_file_sums"][...],
            "per_file_counts": source["per_file_valid_counts"][...],
            "per_file_frames": source["per_file_frame_count"][...],
            "per_batch_sums": source["per_batch_sums"][...],
            "per_batch_counts": source["per_batch_valid_counts"][...],
            "batch_source": source["per_batch_source_index"][...],
            "batch_start": source["per_batch_start_frame"][...],
            "batch_end": source["per_batch_end_frame"][...],
        }
    return {"metadata": metadata, "mode_names": mode_names, **data}


def fit_row(study, mode, mode_index, scope, profile, np, **identifiers):
    fit = robust_fit(profile, np)
    row = {
        "current_key": study["metadata"]["current_key"],
        "current_label": study["metadata"]["current_label"],
        "current_pa": study["metadata"]["current_pa"],
        "mode": mode,
        "scope": scope,
        "source_index": "",
        "batch_index": "",
        "batch_start_frame": "",
        "batch_end_frame": "",
        "half": "combined",
        **identifiers,
    }
    row.update({
        key: value for key, value in fit.items()
        if key not in ("corrected", "hermite_corrected", "predict")
    })
    return row, fit


def fit_study(study, np):
    rows = []
    combined_fits = {}
    per_file_sums = study["per_file_sums"]
    per_file_counts = study["per_file_counts"]
    per_batch_sums = study["per_batch_sums"]
    per_batch_counts = study["per_batch_counts"]
    for mode_index, mode in enumerate(study["mode_names"]):
        combined_profile = collapsed_profile(
            per_file_sums[mode_index].sum(axis=0), per_file_counts.sum(axis=0), np
        )
        row, fit = fit_row(
            study, mode, mode_index, "combined", combined_profile, np
        )
        rows.append(row)
        combined_fits[mode] = {"profile": combined_profile, "fit": fit}

        for half_index, half_name in enumerate(("top", "bottom")):
            profile = collapsed_profile(
                per_file_sums[mode_index, :, half_index].sum(axis=0),
                per_file_counts[:, half_index].sum(axis=0),
                np,
            )
            row, _ = fit_row(
                study, mode, mode_index, "half", profile, np, half=half_name
            )
            rows.append(row)

        for file_index in range(per_file_sums.shape[1]):
            profile = collapsed_profile(
                per_file_sums[mode_index, file_index],
                per_file_counts[file_index],
                np,
            )
            row, _ = fit_row(
                study,
                mode,
                mode_index,
                "file",
                profile,
                np,
                source_index=file_index,
            )
            rows.append(row)

        for batch_index in range(per_batch_sums.shape[0]):
            profile = collapsed_profile(
                per_batch_sums[batch_index, mode_index],
                per_batch_counts[batch_index],
                np,
            )
            row, _ = fit_row(
                study,
                mode,
                mode_index,
                "batch",
                profile,
                np,
                source_index=int(study["batch_source"][batch_index]),
                batch_index=batch_index,
                batch_start_frame=int(study["batch_start"][batch_index]),
                batch_end_frame=int(study["batch_end"][batch_index]),
            )
            rows.append(row)
    return rows, combined_fits


def select_rows(rows, scope, mode=None):
    return [
        row for row in rows
        if row["scope"] == scope and (mode is None or row["mode"] == mode)
    ]


def model_sensitivity_rows(studies, combined_fits, np):
    rows = []
    for study in studies:
        key = study["metadata"]["current_key"]
        for mode in ("no_blr", "grouped_blr"):
            profile = combined_fits[key][mode]["profile"]
            for name, left_start, left_end, right_start, right_end, degree in FIT_CONFIGURATIONS:
                fit = robust_fit(
                    profile,
                    np,
                    left_start=left_start,
                    left_end=left_end,
                    right_start=right_start,
                    right_end=right_end,
                    degree=degree,
                )
                rows.append({
                    "current_key": key,
                    "current_label": study["metadata"]["current_label"],
                    "current_pa": study["metadata"]["current_pa"],
                    "mode": mode,
                    "configuration": name,
                    "left_start": left_start,
                    "left_end_inclusive": left_end - 1,
                    "right_start": right_start,
                    "right_end_inclusive": right_end - 1,
                    "degree": degree,
                    "gain": fit["gain"],
                    "fit_log_rmse": fit["fit_log_rmse"],
                })
    return rows


def save_hermite_test_plot(output_dir, studies, combined_fits, colors, plt, np):
    candidate_keys = ("0130pA", "0500pA", "0060pA", "0030pA", "0015pA", "1000pA")
    studies_by_key = {study["metadata"]["current_key"]: study for study in studies}
    key = next((item for item in candidate_keys if item in studies_by_key), None)
    if key is None:
        key = next(
            study["metadata"]["current_key"] for study in studies
            if study["metadata"]["current_key"] != "0250pA"
        )
    study = studies_by_key[key]
    color = colors[studies.index(study)]
    fit = combined_fits[key]["no_blr"]["fit"]
    x = np.arange(176, 210)
    scaled = fit["corrected"]
    hermite = fit["hermite_corrected"]
    predicted = fit["predict"](x)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(x, scaled[x], color="gray", linewidth=1.0, label="After fitted ZLP gain")
    axes[0].plot(x, hermite[x], color=color, linewidth=1.4, label="Hermite-filled columns 191..193")
    axes[0].plot(x, predicted, color="black", linestyle="--", linewidth=0.9, label="Robust smooth fit")
    axes[0].scatter(
        HERMITE_FILL_COLUMNS,
        hermite[list(HERMITE_FILL_COLUMNS)],
        color="tab:red",
        s=32,
        zorder=4,
        label="Hermite replacement values",
    )
    axes[0].axvline(191.5, color="tab:red", linestyle=":", linewidth=0.8)
    axes[0].axvspan(191, 194, color="tab:red", alpha=0.08)
    axes[0].set_title(
        f"Hermite boundary repair test on {study['metadata']['current_label']} no-BLR spectrum"
    )
    axes[0].set_ylabel("Mean detector value")
    axes[0].legend(fontsize=8)

    delta = hermite[x] - scaled[x]
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].plot(x, delta, color=color, linewidth=1.2)
    axes[1].scatter(HERMITE_FILL_COLUMNS, delta[list(np.asarray(HERMITE_FILL_COLUMNS) - x[0])], color="tab:red", s=30)
    axes[1].axvspan(191, 194, color="tab:red", alpha=0.08)
    axes[1].set_title("Applied change relative to gain-only stitched spectrum")
    axes[1].set_xlabel("Collapsed stitched-spectrum column")
    axes[1].set_ylabel("Hermite - gain-only")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / f"stitch_hermite_boundary_test_{key}.png", dpi=180)
    plt.close(fig)


def save_absolute_stitched_spectrum_plots(output_dir, studies, combined_fits, colors, plt, np):
    x = None
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    linear_fig, linear_axis = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)
    for study, color in zip(studies, colors):
        key = study["metadata"]["current_key"]
        label = study["metadata"]["current_label"]
        spectrum = combined_fits[key]["no_blr"]["fit"]["hermite_corrected"]
        if x is None:
            x = np.arange(spectrum.size)
        masked = np.ma.masked_invalid(np.ma.masked_less_equal(spectrum, 0.0))
        axes[0].plot(x, masked, color=color, linewidth=0.75, label=label)
        axes[1].plot(x[150:235], masked[150:235], color=color, linewidth=0.95, label=label)
        linear_axis.plot(x, spectrum, color=color, linewidth=0.75, label=label)

    axes[0].axvline(191.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].axvspan(191, 194, color="tab:red", alpha=0.08)
    axes[0].set_title("Absolute stitched spectra after ZLP gain and Hermite boundary repair")
    axes[0].set_xlabel("Collapsed stitched-spectrum column (summed ZLP 0..191, CoreLoss 192..3263)")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[1].axvline(191.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].axvspan(191, 194, color="tab:red", alpha=0.08)
    axes[1].set_title("ZLP/CoreLoss transition zoom after Hermite repair")
    axes[1].set_xlabel("Collapsed stitched-spectrum column")
    axes[1].set_ylabel("Mean corrected detector value")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(output_dir / "comparison_absolute_spectra_log.png", dpi=180)
    plt.close(fig)

    linear_axis.axvline(191.5, color="black", linestyle="--", linewidth=0.8)
    linear_axis.axvspan(191, 194, color="tab:red", alpha=0.08)
    linear_axis.set_title("Absolute stitched spectra after ZLP gain and Hermite boundary repair")
    linear_axis.set_xlabel("Collapsed stitched-spectrum column (summed ZLP 0..191, CoreLoss 192..3263)")
    linear_axis.set_ylabel("Mean corrected detector value")
    linear_axis.grid(alpha=0.2)
    linear_axis.legend(ncol=4, fontsize=8)
    linear_fig.savefig(output_dir / "comparison_absolute_spectra.png", dpi=180)
    plt.close(linear_fig)


def final_grouped_blr_spectra(studies, combined_fits, np):
    spectra = []
    metrics = []
    for study in studies:
        key = study["metadata"]["current_key"]
        no_blr_fit = combined_fits[key]["no_blr"]["fit"]
        grouped_profile = combined_fits[key]["grouped_blr"]["profile"]
        spectrum, hermite_metrics = apply_stitch_calibration(
            grouped_profile, no_blr_fit["gain"], np
        )
        spectra.append(spectrum)
        metrics.append({
            "current_key": key,
            "current_label": study["metadata"]["current_label"],
            "current_pa": study["metadata"]["current_pa"],
            "applied_no_blr_gain": no_blr_fit["gain"],
            **hermite_metrics,
        })
    return spectra, metrics


def save_final_grouped_blr_stitched_plots(output_dir, studies, combined_fits, colors, plt, np):
    spectra, _ = final_grouped_blr_spectra(studies, combined_fits, np)
    x = np.arange(spectra[0].size)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    linear_fig, linear_axis = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)
    for study, spectrum, color in zip(studies, spectra, colors):
        label = study["metadata"]["current_label"]
        masked = np.ma.masked_invalid(np.ma.masked_less_equal(spectrum, 0.0))
        axes[0].plot(x, masked, color=color, linewidth=0.75, label=label)
        axes[1].plot(x[150:235], masked[150:235], color=color, linewidth=0.95, label=label)
        linear_axis.plot(x, spectrum, color=color, linewidth=0.75, label=label)

    axes[0].axvline(191.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].axvspan(191, 194, color="tab:red", alpha=0.08)
    axes[0].set_title(
        "Final grouped-BLR spectra stitched with no-BLR gain calibration and Hermite repair"
    )
    axes[0].set_xlabel("Collapsed stitched-spectrum column (summed ZLP 0..191, CoreLoss 192..3263)")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[1].axvline(191.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].axvspan(191, 194, color="tab:red", alpha=0.08)
    axes[1].set_title("ZLP/CoreLoss transition zoom")
    axes[1].set_xlabel("Collapsed stitched-spectrum column")
    axes[1].set_ylabel("Mean corrected detector value")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(output_dir / "comparison_final_grouped_blr_spectra_log.png", dpi=180)
    plt.close(fig)

    linear_axis.axvline(191.5, color="black", linestyle="--", linewidth=0.8)
    linear_axis.axvspan(191, 194, color="tab:red", alpha=0.08)
    linear_axis.set_title(
        "Final grouped-BLR spectra stitched with no-BLR gain calibration and Hermite repair"
    )
    linear_axis.set_xlabel(
        "Collapsed stitched-spectrum column (summed ZLP 0..191, CoreLoss 192..3263)"
    )
    linear_axis.set_ylabel("Mean corrected detector value")
    linear_axis.grid(alpha=0.2)
    linear_axis.legend(ncol=4, fontsize=8)
    linear_fig.savefig(output_dir / "comparison_final_grouped_blr_spectra.png", dpi=180)
    plt.close(linear_fig)


def write_stitched_spectra(path, studies, spectra):
    labels = [study["metadata"]["current_label"] for study in studies]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["column", *labels])
        for column in range(len(spectra[0])):
            writer.writerow([column, *[spectrum[column] for spectrum in spectra]])


def write_final_grouped_blr_metrics(path, metrics):
    fields = list(metrics[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def save_plots(output_dir, studies, rows, combined_fits, sensitivity_rows, plt, np):
    labels = [study["metadata"]["current_label"] for study in studies]
    currents = np.array([study["metadata"]["current_pa"] for study in studies])
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(studies)))

    combined = select_rows(rows, "combined")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for mode, marker in (("no_blr", "o"), ("grouped_blr", "s")):
        selected = sorted(
            (row for row in combined if row["mode"] == mode),
            key=lambda row: row["current_pa"],
        )
        axes[0].semilogx(
            [row["current_pa"] for row in selected],
            [row["gain"] for row in selected],
            marker=marker,
            label=MODE_LABELS[mode],
        )
    file_rows = select_rows(rows, "file", "no_blr")
    for current_index, current in enumerate(currents):
        values = [row["gain"] for row in file_rows if row["current_pa"] == current]
        axes[0].scatter(
            np.full(len(values), current), values, color=colors[current_index], alpha=0.45, s=20
        )
    axes[0].axhline(
        IMAGEJ_GAIN, color="gray", linestyle="--", linewidth=0.9,
        label=f"ImageJ fixed gain ({IMAGEJ_GAIN:.3f})"
    )
    axes[0].set_title("Empirical ZLP gain needed for a smooth ZLP/CoreLoss stitch")
    axes[0].set_ylabel("ZLP multiplicative gain")
    axes[0].legend()

    no_blr = {row["current_pa"]: row for row in combined if row["mode"] == "no_blr"}
    grouped = {row["current_pa"]: row for row in combined if row["mode"] == "grouped_blr"}
    axes[1].semilogx(
        currents,
        [grouped[current]["gain"] / no_blr[current]["gain"] for current in currents],
        marker="o",
    )
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Grouped-BLR bias relative to no-BLR stitch gain")
    axes[1].set_xlabel("Nominal beam current (pA)")
    axes[1].set_ylabel("Grouped-BLR gain / no-BLR gain")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    fig.savefig(output_dir / "stitch_gain_vs_current.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    for axis, study, color in zip(axes.ravel(), studies, colors):
        current = study["metadata"]["current_pa"]
        for mode, marker in (("no_blr", "o"), ("grouped_blr", "s")):
            selected = sorted(
                (
                    row for row in rows
                    if row["scope"] == "batch"
                    and row["mode"] == mode
                    and row["current_pa"] == current
                ),
                key=lambda row: row["batch_index"],
            )
            axis.plot(
                [row["batch_index"] for row in selected],
                [row["gain"] for row in selected],
                marker=marker,
                markersize=3,
                linewidth=0.8,
                label=MODE_LABELS[mode],
            )
        source_rows = sorted(
            (
                row for row in rows
                if row["scope"] == "batch"
                and row["mode"] == "no_blr"
                and row["current_pa"] == current
            ),
            key=lambda row: row["batch_index"],
        )
        for index in range(1, len(source_rows)):
            if source_rows[index]["source_index"] != source_rows[index - 1]["source_index"]:
                axis.axvline(index - 0.5, color="gray", linestyle=":", linewidth=0.7)
        axis.set_title(study["metadata"]["current_label"])
        axis.set_xlabel("Sequential 128-frame batch")
        axis.set_ylabel("ZLP gain")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes.ravel()[len(studies):]:
        axis.set_visible(False)
    fig.savefig(output_dir / "stitch_gain_temporal.png", dpi=180)
    plt.close(fig)

    for mode in ("no_blr", "grouped_blr"):
        fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
        for axis, study, color in zip(axes.ravel(), studies, colors):
            key = study["metadata"]["current_key"]
            profile = combined_fits[key][mode]["profile"]
            fit = combined_fits[key][mode]["fit"]
            x = np.arange(150, 235)
            scaled = profile.copy()
            scaled[:192] *= fit["gain"]
            axis.plot(x[x < 192], profile[x[x < 192]], color="gray", linestyle="--", linewidth=0.8, label="Unscaled ZLP")
            axis.plot(x, scaled[x], color=color, linewidth=1.0, label="Stitched spectrum")
            axis.plot(
                x,
                fit["hermite_corrected"][x],
                color="tab:red",
                linewidth=0.9,
                alpha=0.8,
                label="Hermite boundary repair",
            )
            fit_x = np.arange(150, 235)
            axis.plot(fit_x, fit["predict"](fit_x), color="black", linewidth=0.9, label="Robust smooth fit")
            axis.axvline(191.5, color="tab:red", linestyle=":", linewidth=0.8)
            axis.axvspan(191, 194, color="tab:red", alpha=0.08)
            axis.set_title(
                f"{study['metadata']['current_label']}: gain={fit['gain']:.3f}"
            )
            axis.set_xlabel("Collapsed stitched-spectrum column")
            axis.set_ylabel("Detector value")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        for axis in axes.ravel()[len(studies):]:
            axis.set_visible(False)
        fig.savefig(output_dir / f"stitch_boundary_fits_{mode}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for mode, marker in (("no_blr", "o"), ("grouped_blr", "s")):
        selected = sorted(
            (row for row in combined if row["mode"] == mode),
            key=lambda row: row["current_pa"],
        )
        x = [row["current_pa"] for row in selected]
        axes[0, 0].semilogx(x, [row["direct_core0_to_zlp191_before"] for row in selected], marker=marker, label=MODE_LABELS[mode])
        axes[0, 1].semilogx(x, [row["direct_core0_to_zlp191_after"] for row in selected], marker=marker, label=MODE_LABELS[mode])
        for boundary, field in ((191, "boundary_factor_191"), (192, "boundary_factor_192"), (193, "boundary_factor_193")):
            if mode == "no_blr":
                axes[1, 0].semilogx(x, [row[field] for row in selected], marker="o", label=f"column {boundary}")
    batch_rows = select_rows(rows, "batch", "no_blr")
    batch_cv = []
    for current in currents:
        values = [row["gain"] for row in batch_rows if row["current_pa"] == current]
        batch_cv.append(np.std(values) / np.mean(values))
    axes[1, 1].semilogx(currents, batch_cv, marker="o")
    axes[0, 0].set_title("Direct Core[0] / ZLP[191], before fitted gain")
    axes[0, 1].set_title("Direct Core[0] / ZLP[191], after fitted gain")
    axes[1, 0].set_title("Residual boundary-column correction factors, no BLR")
    axes[1, 1].set_title("No-BLR batch-to-batch stitch-gain variability")
    axes[1, 1].set_ylabel("Coefficient of variation")
    for axis in axes.ravel():
        axis.set_xlabel("Nominal beam current (pA)")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=8) if axis is not axes[1, 1] else None
    fig.savefig(output_dir / "stitch_boundary_diagnostics.png", dpi=180)
    plt.close(fig)

    save_hermite_test_plot(output_dir, studies, combined_fits, colors, plt, np)
    save_absolute_stitched_spectrum_plots(output_dir, studies, combined_fits, colors, plt, np)
    save_final_grouped_blr_stitched_plots(output_dir, studies, combined_fits, colors, plt, np)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    no_blr_sensitivity = [row for row in sensitivity_rows if row["mode"] == "no_blr"]
    for configuration in [item[0] for item in FIT_CONFIGURATIONS]:
        selected = sorted(
            (row for row in no_blr_sensitivity if row["configuration"] == configuration),
            key=lambda row: row["current_pa"],
        )
        axes[0].semilogx(
            [row["current_pa"] for row in selected],
            [row["gain"] for row in selected],
            marker="o",
            linewidth=0.9,
            label=configuration.replace("_", " "),
        )
    axes[0].axhline(IMAGEJ_GAIN, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("No-BLR stitch gain sensitivity to fit window and polynomial degree")
    axes[0].set_ylabel("ZLP multiplicative gain")
    axes[0].legend(fontsize=8, ncol=2)
    sensitivity_cv = []
    sensitivity_range = []
    for current in currents:
        values = [row["gain"] for row in no_blr_sensitivity if row["current_pa"] == current]
        sensitivity_cv.append(np.std(values) / np.mean(values))
        sensitivity_range.append((max(values) - min(values)) / np.mean(values))
    axes[1].semilogx(currents, sensitivity_cv, marker="o", label="coefficient of variation")
    axes[1].semilogx(currents, sensitivity_range, marker="s", label="range / mean")
    axes[1].set_title("Fit-model sensitivity by beam current")
    axes[1].set_xlabel("Nominal beam current (pA)")
    axes[1].set_ylabel("Relative variation")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    fig.savefig(output_dir / "stitch_model_sensitivity.png", dpi=180)
    plt.close(fig)


def write_report(path, studies, rows, sensitivity_rows, np):
    combined_no = {
        row["current_key"]: row
        for row in rows if row["scope"] == "combined" and row["mode"] == "no_blr"
    }
    combined_grouped = {
        row["current_key"]: row
        for row in rows if row["scope"] == "combined" and row["mode"] == "grouped_blr"
    }
    no_blr_sensitivity = [row for row in sensitivity_rows if row["mode"] == "no_blr"]
    lines = [
        "# NiO ZLP/CoreLoss Stitch Study",
        "",
        "## Method",
        "",
        "The four repeated 192-column ZLP reads are summed before stitching to CoreLoss. "
        "The primary calibration uses dark-subtracted data with the static valid-pixel mask "
        "and no BLR. Grouped BLR (4-column ZLP, 16-column CoreLoss) is analyzed as a "
        "sensitivity path. Dynamic masking is disabled.",
        "",
        "A robust log-quadratic curve is fit across collapsed ZLP columns 160..190 and "
        "stitched CoreLoss columns 194..223. Columns 191..193 are excluded. The fitted "
        "step coefficient is converted to a multiplicative ZLP gain. This is an empirical "
        "continuity factor, not yet a physical gain calibration.",
        "",
        "After the ZLP gain is applied, columns 191..193 are filled with a cubic Hermite "
        "interpolant anchored at columns 190 and 194. The left derivative is estimated "
        "from columns 184..190 and the right derivative from columns 194..200, so the "
        "repair enforces local value and first-derivative continuity while only touching "
        "the three transition columns.",
        "",
        "The final fully processed spectra are generated by applying the no-BLR stitch "
        "gain calibration to the grouped-BLR spectra, then applying the same Hermite "
        "boundary repair to columns 191..193. This keeps the calibration from absorbing "
        "grouped-BLR artifacts while still producing stitched spectra after the full "
        "current processing chain.",
        "",
        "## Combined Fits",
        "",
        "| Current | No-BLR gain | Grouped-BLR gain | Grouped/no-BLR | No-BLR batch CV | Model CV | Boundary factors 191 / 192 / 193 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for study in studies:
        key = study["metadata"]["current_key"]
        no = combined_no[key]
        grouped = combined_grouped[key]
        batches = [
            row["gain"] for row in rows
            if row["current_key"] == key and row["scope"] == "batch" and row["mode"] == "no_blr"
        ]
        cv = np.std(batches) / np.mean(batches)
        sensitivity = [
            row["gain"] for row in no_blr_sensitivity if row["current_key"] == key
        ]
        model_cv = np.std(sensitivity) / np.mean(sensitivity)
        lines.append(
            f"| {study['metadata']['current_label']} | {no['gain']:.4f} | "
            f"{grouped['gain']:.4f} | {grouped['gain'] / no['gain']:.4f} | "
            f"{cv:.2%} | {model_cv:.2%} | {no['boundary_factor_191']:.3f} / "
            f"{no['boundary_factor_192']:.3f} / {no['boundary_factor_193']:.3f} |"
        )
    stable_keys = [
        study["metadata"]["current_key"] for study in studies
        if 130 <= study["metadata"]["current_pa"] <= 500
    ]
    stable_gains = [combined_no[key]["gain"] for key in stable_keys]
    boundary_191 = [combined_no[key]["boundary_factor_191"] for key in combined_no]
    boundary_192 = [combined_no[key]["boundary_factor_192"] for key in combined_no]
    boundary_193 = [combined_no[key]["boundary_factor_193"] for key in combined_no]
    lines.extend([
        "",
        f"The historical ImageJ fixed ZLP gain is `{IMAGEJ_GAIN:.6f}`. It is shown as a "
        "reference rather than imposed on the fit.",
        "",
        "## Main findings",
        "",
        f"- The 130–500pA no-BLR gains average `{np.mean(stable_gains):.4f}` "
        f"(range `{min(stable_gains):.4f}..{max(stable_gains):.4f}`), close to the "
        f"historical `{IMAGEJ_GAIN:.4f}` factor.",
        f"- Median no-BLR residual boundary factors are "
        f"`{np.median(boundary_191):.3f} / {np.median(boundary_192):.3f} / "
        f"{np.median(boundary_193):.3f}` for columns 191/192/193.",
        "- The final stitched comparison plots use the no-BLR gain plus the Hermite "
        "boundary repair, not the historical fixed ImageJ boundary multipliers.",
        "- `comparison_final_grouped_blr_spectra_log.png` is the current best "
        "'final result after all processing': grouped-BLR spectra stitched with the "
        "no-BLR calibration.",
        f"- At 250pA grouped BLR changes the inferred gain from "
        f"`{combined_no['0250pA']['gain']:.4f}` to "
        f"`{combined_grouped['0250pA']['gain']:.4f}`, so BLR-contaminated spectra "
        "must not define the stitch calibration.",
        "- Low-current per-batch gains are noise-sensitive; combined/file estimates are "
        "more trustworthy there. The 1nA sequence shows a monotonic temporal gain drift "
        "that merits separate acquisition-stability investigation.",
        "",
        "## Outputs",
        "",
        "- `stitch_fit_results.csv` and `stitch_fit_results.json`",
        "- `stitch_gain_vs_current.png`",
        "- `stitch_gain_temporal.png`",
        "- `stitch_boundary_fits_no_blr.png`",
        "- `stitch_boundary_fits_grouped_blr.png`",
        "- `stitch_boundary_diagnostics.png`",
        "- `stitch_hermite_boundary_test_0130pA.png`",
        "- `comparison_absolute_spectra.png`",
        "- `comparison_absolute_spectra_log.png`",
        "- `stitched_spectra_hermite_no_blr.csv`",
        "- `comparison_final_grouped_blr_spectra.png`",
        "- `comparison_final_grouped_blr_spectra_log.png`",
        "- `stitched_spectra_final_grouped_blr_no_blr_calibration.csv`",
        "- `final_grouped_blr_no_blr_calibration_metrics.csv`",
        "- `stitch_model_sensitivity.png` and `stitch_fit_sensitivity.csv`",
        "",
        "## Interpretation cautions",
        "",
        "The fitted factor assumes the underlying spectrum is smooth across the selected "
        "windows. The Hermite fill additionally assumes no real spectral feature lives in "
        "columns 191..193. Real edges, unequal energy dispersion, or charge-sharing physics "
        "that spreads beyond those columns would invalidate that assumption. Boundary "
        "factors are diagnostics only. The final grouped-BLR spectra still contain any "
        "remaining grouped-BLR baseline artifacts, especially in the known-problematic "
        "250pA run, so this stitch step should not be read as validating the BLR strategy "
        "itself. Exposure time and energy calibration are absent from the DM4 metadata.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    current_dirs = [
        path for path in (args.study_root / "currents").iterdir()
        if (path / "stitch_spectra.h5").exists()
    ]
    current_dirs.sort(
        key=lambda path: json.loads((path / "current_metadata.json").read_text())["current_pa"]
    )
    if not current_dirs:
        raise ValueError("no completed stitch spectra found")
    studies = [load_current(path, h5py, np) for path in current_dirs]
    rows = []
    combined_fits = {}
    for study in studies:
        study_rows, fits = fit_study(study, np)
        rows.extend(study_rows)
        combined_fits[study["metadata"]["current_key"]] = fits

    sensitivity_rows = model_sensitivity_rows(studies, combined_fits, np)
    no_blr_spectra = [
        combined_fits[study["metadata"]["current_key"]]["no_blr"]["fit"]["hermite_corrected"]
        for study in studies
    ]
    write_stitched_spectra(
        args.study_root / "stitched_spectra_hermite_no_blr.csv", studies, no_blr_spectra
    )
    final_grouped_spectra, final_grouped_metrics = final_grouped_blr_spectra(
        studies, combined_fits, np
    )
    write_stitched_spectra(
        args.study_root / "stitched_spectra_final_grouped_blr_no_blr_calibration.csv",
        studies,
        final_grouped_spectra,
    )
    write_final_grouped_blr_metrics(
        args.study_root / "final_grouped_blr_no_blr_calibration_metrics.csv",
        final_grouped_metrics,
    )

    fields = list(rows[0].keys())
    with (args.study_root / "stitch_fit_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.study_root / "stitch_fit_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (args.study_root / "stitch_fit_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    (args.study_root / "stitch_fit_sensitivity.json").write_text(
        json.dumps(sensitivity_rows, indent=2), encoding="utf-8"
    )
    save_plots(args.study_root, studies, rows, combined_fits, sensitivity_rows, plt, np)
    write_report(args.study_root / "report.md", studies, rows, sensitivity_rows, np)
    print(json.dumps({
        "currents": [study["metadata"]["current_label"] for study in studies],
        "fit_rows": len(rows),
        "report": str(args.study_root / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
