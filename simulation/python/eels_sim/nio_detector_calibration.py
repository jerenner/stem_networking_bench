from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

NIO_DETECTOR_CALIBRATION_SCHEMA = "eels-sim-nio-detector-calibration-v1"


def _neighboring_maximum(frame: np.ndarray) -> np.ndarray:
    maximum = np.full_like(frame, -np.inf)
    maximum[1:] = np.maximum(maximum[1:], frame[:-1])
    maximum[:-1] = np.maximum(maximum[:-1], frame[1:])
    maximum[:, 1:] = np.maximum(maximum[:, 1:], frame[:, :-1])
    maximum[:, :-1] = np.maximum(maximum[:, :-1], frame[:, 1:])
    maximum[1:, 1:] = np.maximum(maximum[1:, 1:], frame[:-1, :-1])
    maximum[1:, :-1] = np.maximum(maximum[1:, :-1], frame[:-1, 1:])
    maximum[:-1, 1:] = np.maximum(maximum[:-1, 1:], frame[1:, :-1])
    maximum[:-1, :-1] = np.maximum(maximum[:-1, :-1], frame[1:, 1:])
    return maximum


def _extract_sparse_patches(
    path: str | Path,
    dataset: str,
    pedestal_adu: np.ndarray,
    noise_adu: np.ndarray,
    valid_pixels: np.ndarray,
    frame_start: int,
    frame_count: int | None,
    row_start: int,
    row_stop: int,
    column_start: int,
    storage_lsb: float,
    threshold_sigma: float,
    xray_threshold_adu: float,
    patch_radius: int,
) -> dict[str, object]:
    region = np.s_[row_start:row_stop, column_start:]
    pedestal = pedestal_adu[region]
    noise = noise_adu[region]
    valid = valid_pixels[region]
    global_noise = float(np.median(noise[valid]))
    threshold = np.maximum(threshold_sigma * noise, threshold_sigma * global_noise)
    half_row = (row_stop - row_start) // 2
    patches: list[np.ndarray] = []
    events_per_frame: list[int] = []
    isolated_per_frame: list[int] = []
    with h5py.File(path, "r") as h5:
        frames = h5[dataset]
        stop = len(frames) if frame_count is None else min(len(frames), frame_start + frame_count)
        if not 0 <= frame_start < stop:
            raise ValueError(f"empty frame selection for {path}")
        for frame_index in range(frame_start, stop):
            residual = (
                frames[frame_index, row_start:row_stop, column_start:].astype(np.float32)
                / storage_lsb
                - pedestal
            )
            residual[:half_row] -= np.median(residual[:half_row], axis=0)[None]
            residual[half_row:] -= np.median(residual[half_row:], axis=0)[None]
            candidates = (
                valid
                & (residual > threshold)
                & (residual < xray_threshold_adu)
                & (residual > _neighboring_maximum(residual))
            )
            candidate_rows, candidate_columns = np.nonzero(candidates)
            events_per_frame.append(len(candidate_rows))
            isolated = 0
            for row, column in zip(candidate_rows, candidate_columns):
                if (
                    row < patch_radius
                    or row >= residual.shape[0] - patch_radius
                    or column < patch_radius
                    or column >= residual.shape[1] - patch_radius
                ):
                    continue
                patch_slice = np.s_[
                    row - patch_radius : row + patch_radius + 1,
                    column - patch_radius : column + patch_radius + 1,
                ]
                if not np.all(valid[patch_slice]):
                    continue
                if np.count_nonzero(candidates[patch_slice]) != 1:
                    continue
                patches.append(residual[patch_slice].copy())
                isolated += 1
            isolated_per_frame.append(isolated)
    patch_array = np.asarray(
        patches,
        dtype=np.float32,
    )
    if patch_array.size == 0:
        raise RuntimeError(f"no sparse events found in {path}")
    return {
        "patches": patch_array,
        "events_per_frame": np.asarray(events_per_frame, dtype=np.uint32),
        "isolated_per_frame": np.asarray(isolated_per_frame, dtype=np.uint32),
        "frame_count": len(events_per_frame),
        "global_noise_adu": global_noise,
        "global_threshold_adu": threshold_sigma * global_noise,
    }


def _histogram_subtracted_quantiles(
    spectrum_values: np.ndarray,
    spectrum_frames: int,
    dark_values: np.ndarray,
    dark_frames: int,
    probabilities: np.ndarray,
    value_range: tuple[float, float],
    bins: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum_histogram, edges = np.histogram(spectrum_values, bins=bins, range=value_range)
    dark_histogram = np.histogram(dark_values, bins=edges)[0]
    scale = spectrum_frames / dark_frames
    signal_histogram = np.maximum(
        spectrum_histogram.astype(np.float64) - scale * dark_histogram,
        0.0,
    )
    cumulative = np.cumsum(signal_histogram)
    if cumulative[-1] <= 0.0:
        raise RuntimeError("dark subtraction left no empirical signal events")
    centers = 0.5 * (edges[:-1] + edges[1:])
    quantiles = np.interp(probabilities * cumulative[-1], cumulative, centers)
    return quantiles, centers, signal_histogram


def _fit_response(
    response_templates: np.ndarray,
    measured_peak_quantiles: np.ndarray,
    measured_sum_quantiles: np.ndarray,
    probabilities: np.ndarray,
    noise_sigma_adu: float,
    threshold_adu: float,
    xray_threshold_adu: float,
    random_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(random_seed)
    noise_repeats = 100
    peak_noise = rng.normal(0.0, noise_sigma_adu, size=(len(response_templates), noise_repeats))
    sum_noise = rng.normal(
        0.0,
        np.sqrt(9.0) * noise_sigma_adu,
        size=(len(response_templates), noise_repeats),
    )
    gain_values = np.linspace(0.05, 0.60, 221)
    blur_values = np.linspace(0.0, 1.2, 25)
    candidates: list[dict[str, object]] = []
    response_totals = response_templates.sum(axis=(1, 2))
    for blur_sigma in blur_values:
        blurred = (
            response_templates
            if blur_sigma == 0.0
            else gaussian_filter(
                response_templates,
                sigma=(0.0, blur_sigma, blur_sigma),
                mode="constant",
            )
        )
        if blur_sigma > 0.0:
            blurred_totals = blurred.sum(axis=(1, 2))
            blurred *= np.divide(
                response_totals,
                blurred_totals,
                out=np.ones_like(response_totals),
                where=blurred_totals > 0.0,
            )[:, None, None]
        response_peak = blurred.max(axis=(1, 2))[:, None]
        center_y = blurred.shape[1] // 2
        center_x = blurred.shape[2] // 2
        response_sum = blurred[:, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2].sum(
            axis=(1, 2)
        )[:, None]
        for gain in gain_values:
            simulated_peak = response_peak * gain + peak_noise
            selected = (simulated_peak > threshold_adu) & (simulated_peak < xray_threshold_adu)
            if np.count_nonzero(selected) < 100:
                continue
            simulated_sum = response_sum * gain + sum_noise
            peak_quantiles = np.quantile(simulated_peak[selected], probabilities)
            sum_quantiles = np.quantile(simulated_sum[selected], probabilities)
            objective = float(
                np.mean(np.log(peak_quantiles / measured_peak_quantiles) ** 2)
                + np.mean(
                    np.log(np.maximum(sum_quantiles, 1.0) / np.maximum(measured_sum_quantiles, 1.0))
                    ** 2
                )
            )
            candidates.append(
                {
                    "objective": objective,
                    "gain_adu_per_pair": float(gain),
                    "additional_blur_sigma_pixels": float(blur_sigma),
                    "detection_efficiency_above_threshold": float(selected.mean()),
                    "peak_quantiles_adu": peak_quantiles.tolist(),
                    "patch_sum_quantiles_adu": sum_quantiles.tolist(),
                }
            )
    if not candidates:
        raise RuntimeError("no detector-response fit candidates survived thresholding")
    candidates.sort(key=lambda item: float(item["objective"]))
    best = candidates[0]
    tolerance = float(best["objective"]) + 0.01
    acceptable = [item for item in candidates if float(item["objective"]) <= tolerance]
    best["sensitivity_range"] = {
        "criterion": "objective <= best + 0.01; diagnostic, not a confidence interval",
        "gain_adu_per_pair": [
            min(float(item["gain_adu_per_pair"]) for item in acceptable),
            max(float(item["gain_adu_per_pair"]) for item in acceptable),
        ],
        "additional_blur_sigma_pixels": [
            min(float(item["additional_blur_sigma_pixels"]) for item in acceptable),
            max(float(item["additional_blur_sigma_pixels"]) for item in acceptable),
        ],
    }
    return best


def calibrate_nio_sparse_detector(
    spectrum_hdf5: str | Path,
    dark_hdf5: str | Path,
    readout_calibration_hdf5: str | Path,
    response_kernel_hdf5: str | Path,
    output_hdf5: str | Path,
    output_png: str | Path,
    dataset: str = "frames",
    spectrum_frame_start: int = 0,
    spectrum_frame_count: int | None = None,
    dark_frame_start: int = 256,
    dark_frame_count: int | None = None,
    row_start: int = 32,
    row_stop: int = 928,
    column_start: int = 1986,
    storage_lsb: float = 64.0,
    threshold_sigma: float = 8.0,
    xray_threshold_adu: float = 823.0,
    patch_radius: int = 3,
    random_seed: int = 9173,
) -> dict[str, object]:
    with h5py.File(readout_calibration_hdf5, "r") as calibration:
        pedestal = calibration["maps/pedestal_adu"][:]
        noise = calibration["maps/read_noise_adu"][:]
        valid = calibration["maps/valid_pixel_mask"][:].astype(bool)
    spectrum = _extract_sparse_patches(
        spectrum_hdf5,
        dataset,
        pedestal,
        noise,
        valid,
        spectrum_frame_start,
        spectrum_frame_count,
        row_start,
        row_stop,
        column_start,
        storage_lsb,
        threshold_sigma,
        xray_threshold_adu,
        patch_radius,
    )
    dark = _extract_sparse_patches(
        dark_hdf5,
        dataset,
        pedestal,
        noise,
        valid,
        dark_frame_start,
        dark_frame_count,
        row_start,
        row_stop,
        column_start,
        storage_lsb,
        threshold_sigma,
        xray_threshold_adu,
        patch_radius,
    )
    spectrum_patches = spectrum["patches"]
    dark_patches = dark["patches"]
    center = patch_radius
    spectrum_peaks = spectrum_patches[:, center, center]
    dark_peaks = dark_patches[:, center, center]
    spectrum_sums = spectrum_patches[:, center - 1 : center + 2, center - 1 : center + 2].sum(
        axis=(1, 2)
    )
    dark_sums = dark_patches[:, center - 1 : center + 2, center - 1 : center + 2].sum(axis=(1, 2))
    probabilities = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
    threshold_adu = float(spectrum["global_threshold_adu"])
    peak_quantiles, peak_axis, peak_histogram = _histogram_subtracted_quantiles(
        spectrum_peaks,
        int(spectrum["frame_count"]),
        dark_peaks,
        int(dark["frame_count"]),
        probabilities,
        (threshold_adu, xray_threshold_adu),
    )
    sum_low = float(min(np.quantile(spectrum_sums, 0.001), 0.0))
    sum_high = float(np.quantile(spectrum_sums, 0.9995))
    sum_quantiles, sum_axis, sum_histogram = _histogram_subtracted_quantiles(
        spectrum_sums,
        int(spectrum["frame_count"]),
        dark_sums,
        int(dark["frame_count"]),
        probabilities,
        (sum_low, sum_high),
    )
    with h5py.File(response_kernel_hdf5, "r") as response:
        response_templates = response["event_spatial_pairs"][:].astype(np.float64)
    fit = _fit_response(
        response_templates,
        peak_quantiles,
        sum_quantiles,
        probabilities,
        float(spectrum["global_noise_adu"]),
        threshold_adu,
        xray_threshold_adu,
        random_seed,
    )
    frame_scale = int(spectrum["frame_count"]) / int(dark["frame_count"])
    dark_equivalent_events = frame_scale * len(dark_patches)
    estimated_signal_events = max(len(spectrum_patches) - dark_equivalent_events, 1.0)
    empirical_mean_patch = (
        spectrum_patches.sum(axis=0, dtype=np.float64)
        - frame_scale * dark_patches.sum(axis=0, dtype=np.float64)
    ) / estimated_signal_events
    summary: dict[str, object] = {
        "spectrum_frames": int(spectrum["frame_count"]),
        "dark_frames": int(dark["frame_count"]),
        "spectrum_isolated_events": len(spectrum_patches),
        "dark_isolated_events": len(dark_patches),
        "estimated_dark_fraction": float(dark_equivalent_events / len(spectrum_patches)),
        "region": {
            "rows": [row_start, row_stop],
            "raw_columns": [column_start, pedestal.shape[1]],
            "interpretation": "sparse high-loss CoreLoss tail",
        },
        "threshold": {
            "sigma_multiplier": threshold_sigma,
            "global_noise_adu": float(spectrum["global_noise_adu"]),
            "global_floor_adu": threshold_adu,
            "xray_threshold_adu": xray_threshold_adu,
        },
        "quantile_probabilities": probabilities.tolist(),
        "measured_signal_peak_quantiles_adu": peak_quantiles.tolist(),
        "measured_signal_3x3_sum_quantiles_adu": sum_quantiles.tolist(),
        "fit": fit,
        "recommended_simulation_parameters": {
            "gain_adu_per_electron": fit["gain_adu_per_pair"],
            "raw_doeels.response_blur_sigma_pixels": fit["additional_blur_sigma_pixels"],
        },
        "interpretation": (
            "effective fit conditional on the provisional bare-5-um-silicon "
            "Geant4 charge distribution; sensor thickness and ADC gain remain degenerate"
        ),
        "caveats": [
            "The sparse tail is used only for detector response, not NiO spectral physics.",
            "The DM4 files lack authoritative exposure-time and energy calibration metadata.",
            "Threshold truncation makes the inferred detection efficiency model-dependent.",
            "The fitted extra blur absorbs diffusion, interpixel coupling, and model mismatch.",
        ],
    }
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = NIO_DETECTOR_CALIBRATION_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5.attrs["spectrum_hdf5"] = str(Path(spectrum_hdf5).resolve())
        h5.attrs["dark_hdf5"] = str(Path(dark_hdf5).resolve())
        h5.attrs["response_kernel_hdf5"] = str(Path(response_kernel_hdf5).resolve())
        measured = h5.create_group("measured")
        measured.create_dataset(
            "spectrum_event_patches_adu", data=spectrum_patches, compression="gzip"
        )
        measured.create_dataset("dark_event_patches_adu", data=dark_patches, compression="gzip")
        measured.create_dataset("empirical_signal_mean_patch_adu", data=empirical_mean_patch)
        measured.create_dataset("peak_histogram_axis_adu", data=peak_axis)
        measured.create_dataset("peak_signal_histogram", data=peak_histogram)
        measured.create_dataset("patch_sum_histogram_axis_adu", data=sum_axis)
        measured.create_dataset("patch_sum_signal_histogram", data=sum_histogram)
        measured.create_dataset("spectrum_events_per_frame", data=spectrum["events_per_frame"])
        measured.create_dataset("dark_events_per_frame", data=dark["events_per_frame"])

    _plot_calibration(
        output_png,
        summary,
        empirical_mean_patch,
        peak_axis,
        peak_histogram,
        sum_axis,
        sum_histogram,
    )
    return summary


def _plot_calibration(
    output_png: str | Path,
    summary: dict[str, object],
    empirical_mean_patch: np.ndarray,
    peak_axis: np.ndarray,
    peak_histogram: np.ndarray,
    sum_axis: np.ndarray,
    sum_histogram: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    probabilities = np.asarray(summary["quantile_probabilities"])
    measured_peak = np.asarray(summary["measured_signal_peak_quantiles_adu"])
    measured_sum = np.asarray(summary["measured_signal_3x3_sum_quantiles_adu"])
    fit = summary["fit"]
    fitted_peak = np.asarray(fit["peak_quantiles_adu"])
    fitted_sum = np.asarray(fit["patch_sum_quantiles_adu"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    image = axes[0, 0].imshow(empirical_mean_patch, cmap="magma", origin="lower")
    axes[0, 0].set_title("Dark-subtracted mean isolated-event patch")
    axes[0, 0].set_xlabel("column offset")
    axes[0, 0].set_ylabel("row offset")
    figure.colorbar(image, ax=axes[0, 0], label="ADC code")
    positive_peak = peak_histogram > 0
    axes[0, 1].plot(peak_axis[positive_peak], peak_histogram[positive_peak])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(
        xlabel="local-maximum amplitude (ADC code)",
        ylabel="dark-subtracted events",
        title="Sparse-tail event-amplitude distribution",
    )
    axes[1, 0].plot(measured_peak, probabilities, "o-", label="NiO sparse tail")
    axes[1, 0].plot(fitted_peak, probabilities, "s--", label="fitted Geant4 response")
    axes[1, 0].set(
        xlabel="peak amplitude (ADC code)",
        ylabel="CDF probability",
        title="Peak-response quantiles",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(measured_sum, probabilities, "o-", label="NiO 3x3 sums")
    axes[1, 1].plot(
        fitted_sum,
        probabilities,
        "s--",
        label="fitted Geant4 response",
    )
    axes[1, 1].set(
        xlabel="3x3 patch sum (ADC code)",
        ylabel="CDF probability",
        title="Charge-sharing constraint",
    )
    axes[1, 1].legend()
    figure.suptitle(
        "NiO 15 pA sparse-tail detector calibration\n"
        f"gain={fit['gain_adu_per_pair']:.3f} ADU/e-h pair, "
        f"extra blur={fit['additional_blur_sigma_pixels']:.2f} pixel",
        fontsize=14,
    )
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
