#!/usr/bin/env python3
"""Generate cross-current comparisons for a completed NiO current study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plot_nio_processing_analysis import configure_matplotlib_cache, subtract_imagej_blr


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def combined_mean(sums, counts, np):
    return safe_divide(sums.sum(axis=0), counts.sum(axis=0), np)


def fold_zlp(sums, counts, zlp_width: int, zlp_period: int, np):
    repeats = zlp_width // zlp_period
    folded_sums = sums[..., :zlp_width].reshape(
        *sums.shape[:-1], repeats, zlp_period
    ).sum(axis=-2)
    folded_counts = counts[..., :zlp_width].reshape(
        *counts.shape[:-1], repeats, zlp_period
    ).sum(axis=-2)
    return safe_divide(folded_sums, folded_counts, np)


def baseline_for_zlp(profile, np):
    return float(np.nanmedian(np.concatenate([profile[:25], profile[80:]])))


def peak_metrics(profile, np):
    baseline = baseline_for_zlp(profile, np)
    peak_index = int(np.nanargmax(profile))
    peak_height = float(profile[peak_index])
    amplitude = peak_height - baseline
    half_level = baseline + 0.5 * amplitude
    left = peak_index
    while left > 0 and profile[left] >= half_level:
        left -= 1
    right = peak_index
    while right < profile.size - 1 and profile[right] >= half_level:
        right += 1
    area = float(np.nansum(np.maximum(profile - baseline, 0.0)))
    return {
        "zlp_baseline": baseline,
        "zlp_peak_column": peak_index,
        "zlp_peak_height": peak_height,
        "zlp_peak_amplitude": amplitude,
        "zlp_fwhm_columns": float(right - left),
        "zlp_positive_area": area,
    }


def moving_average(values, width: int, np):
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def load_current(current_dir: Path, h5py, np):
    metadata = json.loads((current_dir / "current_metadata.json").read_text(encoding="utf-8"))
    spectrum_path = current_dir / "spectrum" / "final_spectrum.h5"
    dark_path = current_dir / "dark" / "dark_frame.h5"
    dark_summary = json.loads(
        (current_dir / "dark" / "dark_summary.json").read_text(encoding="utf-8")
    )
    spectrum_summary = json.loads(
        (current_dir / "spectrum" / "spectrum_analysis_summary.json").read_text(encoding="utf-8")
    )
    with h5py.File(spectrum_path, "r") as spectrum_h5:
        sums = spectrum_h5["full_columns_sum"][...]
        counts = spectrum_h5["full_columns_valid_count"][...]
        pipeline_counts = spectrum_h5["full_columns_pipeline_count"][...]
        static_sums = spectrum_h5["full_columns_static_mask_only_sum"][...]
        static_counts = spectrum_h5["full_columns_static_mask_only_valid_count"][...]
        per_file_sums = spectrum_h5["per_file_full_columns_sum"][...]
        per_file_counts = spectrum_h5["per_file_full_columns_valid_count"][...]
        batch_sums = spectrum_h5["per_batch_full_columns_sum"][...]
        batch_counts = spectrum_h5["per_batch_full_columns_valid_count"][...]
        batch_masks = spectrum_h5["per_batch_masked_pixel_count"][...]
    with h5py.File(dark_path, "r") as dark_h5:
        dark_raw = dark_h5["raw_dark_mean"][0].astype(np.float32)
        dark_stddev = dark_h5["dark_stddev"][0].astype(np.float32)
        dark_sources = dark_h5["source_file_mean"][...].astype(np.float32)

    spectrum = combined_mean(sums, counts, np)
    static_spectrum = combined_mean(static_sums, static_counts, np)
    folded = fold_zlp(sums.sum(axis=0), counts.sum(axis=0), 768, 192, np)
    static_folded = fold_zlp(
        static_sums.sum(axis=0), static_counts.sum(axis=0), 768, 192, np
    )
    top_folded = fold_zlp(sums[0], counts[0], 768, 192, np)
    bottom_folded = fold_zlp(sums[1], counts[1], 768, 192, np)
    repeats = []
    for repeat in range(4):
        start = repeat * 192
        end = start + 192
        repeats.append(
            safe_divide(
                sums[:, start:end].sum(axis=0),
                counts[:, start:end].sum(axis=0),
                np,
            )
        )
    repeats = np.stack(repeats)
    per_file = np.stack([
        combined_mean(per_file_sums[index], per_file_counts[index], np)
        for index in range(per_file_sums.shape[0])
    ])
    per_batch = np.stack([
        combined_mean(batch_sums[index], batch_counts[index], np)
        for index in range(batch_sums.shape[0])
    ])
    batch_folded = np.stack([
        fold_zlp(
            batch_sums[index].sum(axis=0),
            batch_counts[index].sum(axis=0),
            768,
            192,
            np,
        )
        for index in range(batch_sums.shape[0])
    ])

    metrics = peak_metrics(folded, np)
    static_metrics = peak_metrics(static_folded, np)
    repeat_peaks = np.max(repeats, axis=1)
    repeat_positions = np.argmax(repeats, axis=1)
    file_peaks = []
    for file_spectrum in per_file:
        file_folded = file_spectrum[:768].reshape(4, 192).mean(axis=0)
        file_peaks.append(np.max(file_folded))
    batch_peaks = np.max(batch_folded, axis=1)
    core_tail_baseline = float(np.nanmedian(spectrum[-512:]))
    core_region = spectrum[768:1536] - core_tail_baseline
    mask_fraction = 1.0 - safe_divide(counts, pipeline_counts, np)

    dark_residual_rms = []
    for source in dark_sources:
        residual = subtract_imagej_blr(
            (source - dark_raw)[None], np, 30, 768, 4, 16
        )[0]
        dark_residual_rms.append(float(np.sqrt(np.mean(np.square(residual[32:928])))))

    metrics.update({
        "current_key": current_dir.name,
        "current_label": metadata["current_label"],
        "current_pa": int(metadata["current_pa"]),
        "dark_frames": int(metadata["dark_frames"]),
        "spectrum_frames": int(metadata["spectrum_frames"]),
        "dark_file_count": len(metadata["dark_files"]),
        "spectrum_file_count": len(metadata["spectrum_files"]),
        "dark_blinker_pixels": int(dark_summary["blinker_pixels"]),
        "dark_mean_temporal_stddev": float(np.mean(dark_stddev[32:928])),
        "dark_source_residual_rms_mean": float(np.mean(dark_residual_rms)),
        "dark_source_residual_rms_max": float(np.max(dark_residual_rms)),
        "top_bottom_zlp_peak_ratio": float(np.max(top_folded) / np.max(bottom_folded)),
        "zlp_repeat_peak_cv": float(np.std(repeat_peaks) / np.mean(repeat_peaks)),
        "zlp_repeat_peak_position_span": int(np.max(repeat_positions) - np.min(repeat_positions)),
        "zlp_file_peak_cv": float(np.std(file_peaks) / np.mean(file_peaks)),
        "zlp_batch_peak_cv": float(np.std(batch_peaks) / np.mean(batch_peaks)),
        "core_early_mean": float(np.nanmean(spectrum[768:1024])),
        "core_positive_area_768_1535": float(np.nansum(np.maximum(core_region, 0.0))),
        "dynamic_mask_difference_rms": float(np.sqrt(np.nanmean(np.square(spectrum - static_spectrum)))),
        "dynamic_mask_zlp_peak_retained_fraction": float(
            metrics["zlp_peak_height"] / static_metrics["zlp_peak_height"]
        ),
        "dynamic_mask_zlp_area_retained_fraction": float(
            metrics["zlp_positive_area"] / static_metrics["zlp_positive_area"]
        ),
        "dynamic_mask_core_early_retained_fraction": float(
            np.nanmean(spectrum[768:1024]) / np.nanmean(static_spectrum[768:1024])
        ),
        "mask_fraction_zlp_mean": float(np.nanmean(mask_fraction[:, :768])),
        "mask_fraction_core_mean": float(np.nanmean(mask_fraction[:, 768:])),
        "batch_masked_pixels_mean": float(np.mean(batch_masks)),
        "batch_masked_pixels_max": int(np.max(batch_masks)),
    })
    return {
        "metadata": metadata,
        "summary": spectrum_summary,
        "metrics": metrics,
        "spectrum": spectrum,
        "static_spectrum": static_spectrum,
        "folded": folded,
        "static_folded": static_folded,
        "top_folded": top_folded,
        "bottom_folded": bottom_folded,
        "repeats": repeats,
        "per_file": per_file,
        "per_batch": per_batch,
        "batch_folded": batch_folded,
        "dark_raw": dark_raw,
        "dark_stddev": dark_stddev,
    }


def save_metrics(path: Path, studies):
    rows = [study["metrics"] for study in studies]
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_file_boundaries(axis, current_count):
    axis.grid(alpha=0.2)


def save_comparison_plots(output_dir: Path, studies, plt, np):
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(studies)))
    labels = [study["metrics"]["current_label"] for study in studies]
    currents = np.array([study["metrics"]["current_pa"] for study in studies], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for study, color, label in zip(studies, colors, labels):
        axes[0].plot(study["spectrum"], color=color, linewidth=0.75, label=label)
        axes[1].plot(study["folded"], color=color, linewidth=0.9, label=label)
    axes[0].axvline(768, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Absolute corrected detector-column spectra")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[1].set_title("Absolute folded ZLP spectra")
    axes[1].set_xlabel("Physical ZLP detector column modulo 192")
    axes[1].set_ylabel("Mean corrected detector value")
    for axis in axes:
        add_file_boundaries(axis, len(studies))
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(output_dir / "comparison_absolute_spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for study, color, label in zip(studies, colors, labels):
        spectrum = np.ma.masked_less_equal(study["spectrum"], 0.0)
        folded = np.ma.masked_less_equal(study["folded"], 0.0)
        axes[0].plot(spectrum, color=color, linewidth=0.75, label=label)
        axes[1].plot(folded, color=color, linewidth=0.9, label=label)
    axes[0].axvline(768, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Absolute corrected detector-column spectra, logarithmic scale")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean corrected detector value")
    axes[1].set_title("Absolute folded ZLP spectra, logarithmic scale")
    axes[1].set_xlabel("Physical ZLP detector column modulo 192")
    axes[1].set_ylabel("Mean corrected detector value")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(output_dir / "comparison_absolute_spectra_log.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    for study, color, label, current in zip(studies, colors, labels, currents):
        axes[0].plot(study["spectrum"] / current, color=color, linewidth=0.75, label=label)
        folded = study["folded"]
        peak = int(np.argmax(folded))
        aligned = np.roll(folded / np.max(folded), 52 - peak)
        axes[1].plot(aligned, color=color, linewidth=0.9, label=label)
        core = moving_average(study["spectrum"][768:], 11, np)
        axes[2].plot(core / study["metrics"]["zlp_positive_area"], color=color, linewidth=0.85, label=label)
    axes[0].axvline(768, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Spectrum normalized by nominal beam current")
    axes[0].set_ylabel("Detector value / pA")
    axes[1].set_title("Peak-aligned folded ZLP shapes normalized to unit height")
    axes[1].set_ylabel("Relative intensity")
    axes[2].set_title("Smoothed CoreLoss spectra normalized by folded-ZLP positive area")
    axes[2].set_ylabel("Relative intensity")
    for axis in axes:
        axis.set_xlabel("Detector output column" if axis is not axes[1] else "Aligned physical ZLP column")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    fig.savefig(output_dir / "comparison_normalized_spectra.png", dpi=180)
    plt.close(fig)

    peak_height = np.array([s["metrics"]["zlp_peak_height"] for s in studies])
    peak_area = np.array([s["metrics"]["zlp_positive_area"] for s in studies])
    core_area = np.array([s["metrics"]["core_positive_area_768_1535"] for s in studies])
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for axis, values, title in (
        (axes[0, 0], peak_height, "Folded ZLP peak height"),
        (axes[0, 1], peak_area, "Folded ZLP positive area"),
        (axes[1, 0], core_area, "Early CoreLoss positive area"),
    ):
        axis.loglog(currents, values, marker="o")
        reference = values[0] * currents / currents[0]
        axis.loglog(currents, reference, linestyle="--", color="gray", label="linear in nominal current")
        axis.set_title(title)
        axis.set_xlabel("Nominal beam current (pA)")
        axis.set_ylabel("Detector metric")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=8)
    axes[1, 1].semilogx(currents, peak_area / currents, marker="o", label="ZLP area / pA")
    axes[1, 1].semilogx(currents, core_area / currents, marker="o", label="CoreLoss area / pA")
    axes[1, 1].set_title("Nominal-current-normalized integrated intensity")
    axes[1, 1].set_xlabel("Nominal beam current (pA)")
    axes[1, 1].set_ylabel("Detector metric / pA")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend()
    fig.savefig(output_dir / "comparison_current_scaling.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    diagnostic_fields = [
        ("zlp_peak_column", "Folded ZLP peak column"),
        ("zlp_fwhm_columns", "Folded ZLP FWHM (columns)"),
        ("top_bottom_zlp_peak_ratio", "Top/bottom ZLP peak ratio"),
        ("zlp_repeat_peak_cv", "ZLP repeat peak coefficient of variation"),
        ("mask_fraction_core_mean", "Mean CoreLoss masked fraction"),
        ("dynamic_mask_difference_rms", "Dynamic-mask spectral RMS effect"),
    ]
    for axis, (field, title) in zip(axes.ravel(), diagnostic_fields):
        values = [study["metrics"][field] for study in studies]
        axis.semilogx(currents, values, marker="o")
        axis.set_title(title)
        axis.set_xlabel("Nominal beam current (pA)")
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "comparison_detector_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    dynamic_area_per_current = np.array(
        [study["metrics"]["zlp_positive_area"] for study in studies]
    ) / currents
    static_area_per_current = np.array([
        peak_metrics(study["static_folded"], np)["zlp_positive_area"]
        for study in studies
    ]) / currents
    axes[0, 0].semilogx(
        currents, static_area_per_current, marker="o", label="static valid mask only"
    )
    axes[0, 0].semilogx(
        currents, dynamic_area_per_current, marker="o", label="static + dynamic masks"
    )
    axes[0, 0].set_title("Folded-ZLP area per nominal pA")
    axes[0, 0].set_ylabel("Detector metric / pA")
    axes[0, 0].legend()

    axes[0, 1].semilogx(
        currents,
        [s["metrics"]["dynamic_mask_zlp_peak_retained_fraction"] for s in studies],
        marker="o",
        label="ZLP peak height",
    )
    axes[0, 1].semilogx(
        currents,
        [s["metrics"]["dynamic_mask_zlp_area_retained_fraction"] for s in studies],
        marker="o",
        label="ZLP positive area",
    )
    axes[0, 1].semilogx(
        currents,
        [s["metrics"]["dynamic_mask_core_early_retained_fraction"] for s in studies],
        marker="o",
        label="early CoreLoss mean",
    )
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Signal retained after dynamic masking")
    axes[0, 1].set_ylabel("Dynamic / static-only")
    axes[0, 1].legend()

    axes[1, 0].semilogx(
        currents,
        [s["metrics"]["mask_fraction_zlp_mean"] for s in studies],
        marker="o",
        label="ZLP",
    )
    axes[1, 0].semilogx(
        currents,
        [s["metrics"]["mask_fraction_core_mean"] for s in studies],
        marker="o",
        label="CoreLoss",
    )
    axes[1, 0].set_title("Mean dynamically masked detector fraction")
    axes[1, 0].set_ylabel("Masked fraction")
    axes[1, 0].legend()

    axes[1, 1].semilogx(
        currents,
        [s["metrics"]["batch_masked_pixels_mean"] for s in studies],
        marker="o",
        label="mean",
    )
    axes[1, 1].semilogx(
        currents,
        [s["metrics"]["batch_masked_pixels_max"] for s in studies],
        marker="o",
        label="maximum",
    )
    axes[1, 1].set_title("Masked pixels per 128-frame batch")
    axes[1, 1].set_ylabel("Pixels")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.set_xlabel("Nominal beam current (pA)")
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "comparison_dynamic_mask_sensitivity.png", dpi=180)
    plt.close(fig)

    reference_dark = studies[0]["dark_raw"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    for study, color, label in zip(studies, colors, labels):
        dark = study["dark_raw"]
        delta = subtract_imagej_blr((dark - reference_dark)[None], np, 30, 768, 4, 16)[0]
        axes[0].plot(delta[32:928].mean(axis=0), color=color, linewidth=0.7, label=label)
        folded_delta = delta[32:928, :768].mean(axis=0).reshape(4, 192).mean(axis=0)
        axes[1].plot(folded_delta, color=color, linewidth=0.9, label=label)
    blinkers = [study["metrics"]["dark_blinker_pixels"] for study in studies]
    dark_rms = [study["metrics"]["dark_source_residual_rms_mean"] for study in studies]
    axes[2].semilogx(currents, blinkers, marker="o", label="blinker pixels")
    twin = axes[2].twinx()
    twin.semilogx(currents, dark_rms, marker="s", color="tab:red", label="source residual RMS")
    axes[0].axvline(768, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Dark means relative to {labels[0]} dark, after BLR")
    axes[0].set_xlabel("Detector output column")
    axes[0].set_ylabel("Mean residual")
    axes[1].set_title("Folded ZLP dark-mean differences")
    axes[1].set_xlabel("Physical ZLP detector column modulo 192")
    axes[1].set_ylabel("Mean residual")
    axes[2].set_title("Dark calibration diagnostics")
    axes[2].set_xlabel("Nominal beam current label (pA)")
    axes[2].set_ylabel("Blinker pixels")
    twin.set_ylabel("Within-current dark source RMS")
    for axis in axes[:2]:
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    axes[2].grid(alpha=0.2)
    lines, line_labels = axes[2].get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    axes[2].legend(lines + twin_lines, line_labels + twin_labels)
    fig.savefig(output_dir / "comparison_dark_stability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    file_cv = [study["metrics"]["zlp_file_peak_cv"] for study in studies]
    batch_cv = [study["metrics"]["zlp_batch_peak_cv"] for study in studies]
    axes[0].semilogx(currents, file_cv, marker="o", label="file-to-file CV")
    axes[0].semilogx(currents, batch_cv, marker="o", label="128-frame batch CV")
    axes[0].set_title("Folded ZLP peak temporal variability")
    axes[0].set_ylabel("Coefficient of variation")
    axes[0].legend()
    repeat_matrix = np.stack([np.max(study["repeats"], axis=1) for study in studies])
    repeat_matrix /= repeat_matrix.mean(axis=1, keepdims=True)
    image = axes[1].imshow(repeat_matrix, cmap="coolwarm", vmin=0.95, vmax=1.05, aspect="auto")
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["ZLP read 1", "ZLP read 2", "ZLP read 3", "ZLP read 4"])
    axes[1].set_title("Repeated ZLP peak heights relative to each-current mean")
    fig.colorbar(image, ax=axes[1], label="Relative peak height")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Nominal beam current (pA)" if axis is axes[0] else "Repeated read")
    fig.savefig(output_dir / "comparison_repeat_and_temporal_stability.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, studies, metrics_path: Path):
    first = studies[0]["metrics"]
    last = studies[-1]["metrics"]
    peak_columns = [study["metrics"]["zlp_peak_column"] for study in studies]
    repeat_cvs = [study["metrics"]["zlp_repeat_peak_cv"] for study in studies]
    top_bottom = [study["metrics"]["top_bottom_zlp_peak_ratio"] for study in studies]
    dark_blinkers = [study["metrics"]["dark_blinker_pixels"] for study in studies]
    lines = [
        "# NiO Beam-Current Study",
        "",
        "## Scope",
        "",
        "All spectra were dark-subtracted with their matching current-labelled dark calibration, "
        "BLR-corrected, and accumulated with both static-only and static+dynamic masks. DM4 files "
        "contain no usable exposure-time or energy-axis calibration, so detector columns and nominal "
        "current are reported without assigning eV values.",
        "",
        "## Inventory",
        "",
        "| Current | Dark frames | Spectrum frames | Dark files | Spectrum files |",
        "|---|---:|---:|---:|---:|",
    ]
    for study in studies:
        metric = study["metrics"]
        lines.append(
            f"| {metric['current_label']} | {metric['dark_frames']} | {metric['spectrum_frames']} | "
            f"{metric['dark_file_count']} | {metric['spectrum_file_count']} |"
        )
    lines.extend([
        "",
        "## Generated Comparisons",
        "",
        "- `comparison_absolute_spectra.png`",
        "- `comparison_absolute_spectra_log.png`",
        "- `comparison_normalized_spectra.png`",
        "- `comparison_current_scaling.png`",
        "- `comparison_detector_diagnostics.png`",
        "- `comparison_dynamic_mask_sensitivity.png`",
        "- `comparison_dark_stability.png`",
        "- `comparison_repeat_and_temporal_stability.png`",
        "",
        f"Numerical metrics are in `{metrics_path.name}`.",
        "",
        "## Main Findings",
        "",
        f"- The four ZLP reads are internally consistent: their peak-height coefficient of variation "
        f"ranges from {min(repeat_cvs):.3%} to {max(repeat_cvs):.3%}.",
        f"- The folded ZLP peak moves from detector column {peak_columns[0]} at "
        f"{first['current_label']} to {peak_columns[-1]} at {last['current_label']} "
        f"(full observed range {min(peak_columns)}..{max(peak_columns)}). Without an energy-axis "
        "calibration this is a detector-coordinate/acquisition drift measurement, not an energy shift.",
        f"- Top/bottom ZLP imbalance grows from {top_bottom[0]:.3f} to {top_bottom[-1]:.3f}; "
        "the high-current data therefore show a detector-half or illumination asymmetry that should "
        "be tracked separately from the summed spectrum.",
        f"- The fixed dynamic-mask threshold increasingly removes coherent signal: folded-ZLP area "
        f"retention falls from {first['dynamic_mask_zlp_area_retained_fraction']:.1%} at "
        f"{first['current_label']} to {last['dynamic_mask_zlp_area_retained_fraction']:.1%} at "
        f"{last['current_label']}, while peak-height retention falls to "
        f"{last['dynamic_mask_zlp_peak_retained_fraction']:.1%}. The +500 threshold should not be "
        "used as a current-independent production setting.",
        f"- Dark-frame blinker counts rise from {dark_blinkers[0]:,} to a maximum of "
        f"{max(dark_blinkers):,} pixels, and source-file dark residual RMS also rises at high current "
        "labels. Matching dark calibration helps, but acquisition-time/configuration drift remains.",
        "- The 250pA CoreLoss spectrum contains broad detector-column plateaus that are consistent "
        "across all four source files and survive the static/dynamic-mask comparison. They are not "
        "transient blinkers; they should be checked against acquisition settings, detector readout "
        "boundaries, and sample position before being interpreted as EELS structure.",
        "- The 500pA result contains only spectrum files 0002..0004, and the 1nA result contains two "
        "dark and two spectrum files. Their temporal statistics are therefore not directly matched "
        "to the four-file lower-current studies.",
        "",
        "## Interpretation Rules",
        "",
        "Features fixed in detector coordinates, repeated modulo 192, present in darks, inconsistent "
        "between detector halves, or nonlinear with nominal current are detector-effect candidates. "
        "Features that align with the ZLP and retain shape after dose normalization are more plausibly "
        "EELS signal. Exposure metadata and an energy calibration are required before quantitative "
        "cross-section or edge-energy interpretation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    current_dirs = []
    for path in (args.study_root / "currents").iterdir():
        if (path / "spectrum" / "final_spectrum.h5").exists():
            current_dirs.append(path)
    current_dirs.sort(
        key=lambda path: json.loads(
            (path / "current_metadata.json").read_text(encoding="utf-8")
        )["current_pa"]
    )
    if not current_dirs:
        raise ValueError("no completed current analyses found")

    studies = [load_current(path, h5py, np) for path in current_dirs]
    output_dir = args.study_root / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "current_metrics.csv"
    save_metrics(metrics_path, studies)
    (output_dir / "current_metrics.json").write_text(
        json.dumps([study["metrics"] for study in studies], indent=2), encoding="utf-8"
    )
    save_comparison_plots(output_dir, studies, plt, np)
    write_report(args.study_root / "report.md", studies, metrics_path)
    print(
        json.dumps(
            {
                "currents": [study["metrics"]["current_label"] for study in studies],
                "metrics": str(metrics_path),
                "output_dir": str(output_dir),
                "report": str(args.study_root / "report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
