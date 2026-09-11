from __future__ import annotations

import json
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .config import load_config
from .io import read_readout_calibration
from .raw_doeels import (
    _energy_to_raw_column_mapping,
    _fold_raw_columns,
    _prepare_response_kernel,
    _read_response_kernel,
    _sample_raw_columns,
    _stitched_to_energy,
    _transferred_component_profiles,
)
from .readout import select_calibrated_maps

EXPOSURE_SWEEP_SCHEMA = "eels-sim-raw-doeels-exposure-sweep-v1"

DEMO_COLORS = {
    "HAADF": (255, 255, 255),
    "Mn": (65, 215, 100),
    "O": (255, 220, 40),
    "Ti": (255, 75, 65),
}


def _row_probability(rows: int, center: float, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("raw_doeels.detector_row_sigma_pixels must be positive")
    coordinates = np.arange(rows, dtype=np.float64)
    probability = np.exp(-0.5 * ((coordinates - center) / sigma) ** 2)
    probability /= probability.sum()
    return probability


def _projected_response_model(
    row_probability: np.ndarray,
    response_totals: np.ndarray,
    response_templates: np.ndarray | None,
    spatial_kernel: np.ndarray,
    exact_bank_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project single-electron charge clouds through the row-profile estimator."""
    templates = response_templates
    if templates is None:
        templates = response_totals[:, None, None] * spatial_kernel[None, :, :]
    rows = len(row_probability)
    kernel_rows, kernel_columns = templates.shape[1:]
    radius_y = kernel_rows // 2
    centered = row_probability - row_probability.mean()
    denominator = float(np.dot(centered, row_probability))
    if denominator <= 0.0:
        raise ValueError("detector row profile has no reconstructable contrast")

    projection = np.zeros((rows, kernel_rows), dtype=np.float64)
    source_rows = np.arange(rows)
    for kernel_y in range(kernel_rows):
        destination = source_rows + kernel_y - radius_y
        valid = (destination >= 0) & (destination < rows)
        projection[valid, kernel_y] = centered[destination[valid]] / denominator

    mean = np.zeros(kernel_columns, dtype=np.float64)
    second = np.zeros((kernel_columns, kernel_columns), dtype=np.float64)
    template_weight = 1.0 / len(templates)
    for template in templates:
        vectors = projection @ template
        mean += template_weight * (row_probability @ vectors)
        second += template_weight * np.einsum("r,rx,ry->xy", row_probability, vectors, vectors)
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T

    bank_size = max(1, exact_bank_size)
    bank_rows = rng.choice(rows, size=bank_size, p=row_probability)
    bank_templates = rng.integers(0, len(templates), size=bank_size)
    bank = np.zeros((bank_size, kernel_columns), dtype=np.float64)
    for kernel_y in range(kernel_rows):
        destination = bank_rows + kernel_y - radius_y
        valid = (destination >= 0) & (destination < rows)
        if not np.any(valid):
            continue
        weights = centered[destination[valid]] / denominator
        bank[valid] += templates[bank_templates[valid], kernel_y, :] * weights[:, None]
    return mean, covariance, bank


def _add_shifted_columns(
    destination: np.ndarray,
    source: np.ndarray,
    kernel_column: int,
    kernel_columns: int,
) -> None:
    offset = kernel_column - kernel_columns // 2
    source_start = max(0, -offset)
    source_stop = min(len(source), len(source) - offset)
    destination[source_start + offset : source_stop + offset] += source[source_start:source_stop]


def _sample_projected_charge(
    raw_column_hits: np.ndarray,
    response_mean: np.ndarray,
    response_covariance: np.ndarray,
    response_bank: np.ndarray,
    exact_response_max_electrons: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    total_hits = int(raw_column_hits.sum())
    columns = len(raw_column_hits)
    kernel_columns = len(response_mean)
    projected = np.zeros(columns, dtype=np.float64)
    if total_hits == 0:
        return projected, "none"
    if total_hits <= exact_response_max_electrons:
        occupied = np.flatnonzero(raw_column_hits)
        source_columns = np.repeat(occupied, raw_column_hits[occupied].astype(int))
        vectors = response_bank[rng.integers(0, len(response_bank), size=len(source_columns))]
        radius = kernel_columns // 2
        for kernel_column in range(kernel_columns):
            destination = source_columns + kernel_column - radius
            valid = (destination >= 0) & (destination < columns)
            np.add.at(projected, destination[valid], vectors[valid, kernel_column])
        return projected, "exact_projected_event"

    for kernel_column in range(kernel_columns):
        _add_shifted_columns(
            projected,
            raw_column_hits * response_mean[kernel_column],
            kernel_column,
            kernel_columns,
        )
    eigenvalues, eigenvectors = np.linalg.eigh(response_covariance)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    fluctuations = rng.normal(size=(columns, kernel_columns)) @ factor.T
    fluctuations *= np.sqrt(raw_column_hits)[:, None]
    for kernel_column in range(kernel_columns):
        _add_shifted_columns(
            projected,
            fluctuations[:, kernel_column],
            kernel_column,
            kernel_columns,
        )
    return projected, "compound_gaussian_projected_event"


def _projected_readout_noise(
    row_probability: np.ndarray,
    noise_map: np.ndarray | None,
    read_noise_adu: float,
    common_mode_noise_adu: float,
    row_common_mode_noise_adu: float,
    column_common_mode_noise_adu: float,
    use_correlated_noise: bool,
) -> tuple[np.ndarray, float]:
    centered = row_probability - row_probability.mean()
    denominator = float(np.dot(centered, row_probability))
    correlated_variance = 0.0
    if use_correlated_noise:
        correlated_variance = (
            common_mode_noise_adu**2
            + row_common_mode_noise_adu**2
            + column_common_mode_noise_adu**2
        )
    if noise_map is None:
        independent_variance: np.ndarray | float = read_noise_adu**2
    else:
        independent_variance = np.maximum(noise_map**2 - correlated_variance, 0.0)
    # Rounding is independently applied to every ADC pixel in every readout.
    independent_variance = independent_variance + 1.0 / 12.0
    if np.isscalar(independent_variance):
        column_variance = np.full(
            noise_map.shape[1] if noise_map is not None else 1,
            float(independent_variance) * np.dot(centered, centered) / denominator**2,
        )
    else:
        column_variance = (centered[:, None] ** 2 * independent_variance).sum(
            axis=0
        ) / denominator**2
    row_common_variance = 0.0
    if use_correlated_noise and row_common_mode_noise_adu > 0.0:
        row_common_variance = (
            row_common_mode_noise_adu**2 * np.dot(centered, centered) / denominator**2
        )
    return column_variance, float(row_common_variance)


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    candidate = np.asarray(candidate, dtype=np.float64).ravel()
    if reference.std() == 0.0 or candidate.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(reference, candidate)[0, 1])


def simulate_exposure_sweep(
    config_path: str | Path,
    spectrum_image_hdf5: str | Path,
    output_hdf5: str | Path,
    exposures: list[int],
    scan_stride: int | None = None,
    depth_indices: list[int] | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    raw_config = config.raw_doeels
    if raw_config.response_kernel_file is None:
        raise ValueError("raw_doeels.response_kernel_file is required")
    exposure_values = np.unique(np.asarray(exposures, dtype=np.int64))
    if len(exposure_values) == 0 or np.any(exposure_values <= 0):
        raise ValueError("exposures must contain positive integration counts")
    stride = raw_config.scan_stride if scan_stride is None else scan_stride
    if stride <= 0:
        raise ValueError("scan stride must be positive")
    if not 0.0 < raw_config.counting_detection_efficiency <= 1.0:
        raise ValueError("counting detection efficiency must be in (0,1]")
    if raw_config.counting_dark_events_per_frame < 0.0:
        raise ValueError("counting dark-event rate cannot be negative")
    selected_depths = np.asarray(
        raw_config.depth_indices if depth_indices is None else depth_indices,
        dtype=int,
    )

    response_totals, response_templates, spatial_kernel, response_summary = _read_response_kernel(
        raw_config.response_kernel_file
    )
    response_templates, spatial_kernel = _prepare_response_kernel(
        response_totals,
        response_templates,
        spatial_kernel,
        raw_config.response_blur_sigma_pixels,
    )
    detector_shape = (
        config.spectrometer.detector_rows,
        config.spectrometer.detector_columns,
    )
    row_probability = _row_probability(
        detector_shape[0],
        config.spectrometer.zero_y_row,
        raw_config.detector_row_sigma_pixels,
    )
    model_rng = np.random.default_rng(raw_config.random_seed ^ 0x5EED5EED)
    response_mean, response_covariance, response_bank = _projected_response_model(
        row_probability,
        response_totals,
        response_templates,
        spatial_kernel,
        min(max(raw_config.exact_response_max_electrons, 1), 100_000),
        model_rng,
    )

    calibration = (
        None
        if config.readout.calibration_file is None
        else read_readout_calibration(config.readout.calibration_file)
    )
    _, noise_map, signal_efficiency_map = select_calibrated_maps(calibration, config.readout)
    if noise_map is not None and noise_map.shape != detector_shape:
        raise ValueError("readout noise calibration shape does not match detector")
    if signal_efficiency_map is not None:
        raise ValueError("streaming exposure sweeps currently require simulate_known_defects=false")
    column_noise_variance, row_common_variance = _projected_readout_noise(
        row_probability,
        noise_map,
        config.readout.read_noise_adu,
        config.readout.common_mode_noise_adu,
        config.readout.row_common_mode_noise_adu,
        config.readout.column_common_mode_noise_adu,
        config.readout.use_calibrated_correlated_noise,
    )
    if len(column_noise_variance) == 1:
        column_noise_variance = np.full(detector_shape[1], column_noise_variance[0])

    rng = np.random.default_rng(raw_config.random_seed)
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spectrum_image_hdf5, "r") as source:
        if source.attrs.get("schema") != "eels-sim-spectrum-image-v1":
            raise ValueError("Input is not an EELS spectrum image")
        spectra = source["spectrum/counts"]
        component_totals = source["spectrum/component_total_counts"]
        expected_component_source = source.get("spectrum/expected_component_counts")
        component_profiles = source["spectrum/component_profiles"][:]
        component_labels = json.loads(source["spectrum"].attrs["component_labels_json"])
        source_summary = json.loads(source.attrs.get("summary_json", "{}"))
        selected_element_edges = source_summary.get("selected_element_edges", {})
        energy_edges = source["axes/energy_edges_eV"][:]
        input_depths, scan_rows, scan_columns, energy_bins = spectra.shape
        if np.any(selected_depths < 0) or np.any(selected_depths >= input_depths):
            raise ValueError("depth indices are outside the spectrum image")
        y_indices = np.arange(0, scan_rows, stride, dtype=int)
        x_indices = np.arange(0, scan_columns, stride, dtype=int)
        mappings = _energy_to_raw_column_mapping(energy_edges, config)
        fit_profiles = _transferred_component_profiles(
            component_profiles, mappings, energy_edges, config
        )
        profile_pseudoinverse = np.linalg.pinv(fit_profiles.T, rcond=1.0e-5)
        energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
        counting_energy_mask = energy_centers >= raw_config.counting_min_energy_eV
        if not np.any(counting_energy_mask):
            raise ValueError("counting energy boundary leaves no spectrum bins")
        counting_pseudoinverse = np.linalg.pinv(
            fit_profiles[:, counting_energy_mask].T, rcond=1.0e-5
        )
        if not 0 <= raw_config.counting_first_raw_column < detector_shape[1]:
            raise ValueError("counting first raw column is outside the detector")
        if not 0 <= raw_config.counting_dark_first_raw_column < detector_shape[1]:
            raise ValueError("counting dark first raw column is outside the detector")
        shape = (
            len(exposure_values),
            len(selected_depths),
            len(y_indices),
            len(x_indices),
        )
        energy_shape = (*shape, energy_bins)
        component_shape = (*shape, len(component_labels))

        with h5py.File(output_path, "w") as output:
            output.attrs["schema"] = EXPOSURE_SWEEP_SCHEMA
            output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["config_file"] = str(Path(config_path).resolve())
            output.attrs["source_spectrum_image"] = str(Path(spectrum_image_hdf5).resolve())
            output.attrs["response_kernel"] = str(Path(raw_config.response_kernel_file).resolve())
            output.attrs["component_labels_json"] = json.dumps(component_labels)
            axes = output.create_group("axes")
            axes.create_dataset("integrations_per_position", data=exposure_values)
            axes.create_dataset(
                "dwell_ms",
                data=1.0e3 * exposure_values / config.experiment.frame_rate_hz,
            )
            axes.create_dataset("energy_edges_eV", data=energy_edges)
            axes.create_dataset(
                "energy_centers_eV", data=0.5 * (energy_edges[:-1] + energy_edges[1:])
            )
            axes.create_dataset("depth_index", data=selected_depths)
            axes.create_dataset("y_index", data=y_indices)
            axes.create_dataset("x_index", data=x_indices)
            for source_name, selection in (
                ("focal_depth_A", selected_depths),
                ("scan_y_A", y_indices),
                ("scan_x_A", x_indices),
            ):
                if source_name in source["axes"]:
                    axes.create_dataset(source_name, data=source["axes"][source_name][selection])

            truth = output.create_group("truth")
            incident = truth.create_dataset(
                "incident_spectrum_counts", energy_shape, "u8", compression="gzip"
            )
            accepted = truth.create_dataset("accepted_electrons", shape, "u8")
            rejected = truth.create_dataset("rejected_electrons", shape, "u8")
            expected_components = truth.create_dataset(
                "expected_component_counts", component_shape, "f4", compression="gzip"
            )
            reconstruction = output.create_group("reconstruction")
            pre_energy = reconstruction.create_dataset(
                "pre_readout_energy_counts", energy_shape, "f4", compression="gzip"
            )
            detector_energy = reconstruction.create_dataset(
                "energy_counts", energy_shape, "f4", compression="gzip"
            )
            counted_energy = reconstruction.create_dataset(
                "counted_energy_counts", energy_shape, "f4", compression="gzip"
            )
            pre_components = reconstruction.create_dataset(
                "pre_readout_component_counts",
                component_shape,
                "f4",
                compression="gzip",
            )
            detector_components = reconstruction.create_dataset(
                "component_counts", component_shape, "f4", compression="gzip"
            )
            counted_components = reconstruction.create_dataset(
                "counted_component_counts",
                component_shape,
                "f4",
                compression="gzip",
            )
            haadf_expected = reconstruction.create_dataset(
                "haadf_expected_counts", shape, "f4", compression="gzip"
            )
            haadf_counts = reconstruction.create_dataset(
                "haadf_counts", shape, "u8", compression="gzip"
            )
            reconstruction.attrs["accumulation"] = (
                "nested checkpoints; projected compound response and readout-noise "
                "sufficient statistics; no raw detector frames retained"
            )
            response_modes: set[str] = set()
            rejected_total = 0
            source_haadf = source.get("truth/haadf_expected_counts")
            if source_haadf is None:
                source_haadf = source["reconstruction/haadf_counts"]

            for depth_output, depth_index in enumerate(selected_depths):
                for y_output, y_index in enumerate(y_indices):
                    for x_output, x_index in enumerate(x_indices):
                        if expected_component_source is None:
                            source_components = component_totals[
                                depth_index, y_index, x_index
                            ].astype(np.float64)
                            probability = spectra[depth_index, y_index, x_index].astype(np.float64)
                        else:
                            source_components = expected_component_source[
                                depth_index, y_index, x_index
                            ].astype(np.float64)
                            probability = source_components @ component_profiles
                        source_electrons = float(probability.sum())
                        probability /= source_electrons
                        component_per_integration = (
                            source_components
                            * raw_config.electrons_per_integration
                            / source_electrons
                        )
                        haadf_per_integration = (
                            float(source_haadf[depth_index, y_index, x_index])
                            * raw_config.electrons_per_integration
                            / source_electrons
                        )
                        cumulative_incident = np.zeros(energy_bins, dtype=np.uint64)
                        cumulative_raw_columns = np.zeros(detector_shape[1], dtype=np.uint64)
                        cumulative_projected_pairs = np.zeros(detector_shape[1], dtype=np.float64)
                        cumulative_noise_adu = np.zeros(detector_shape[1], dtype=np.float64)
                        cumulative_counted_columns = np.zeros(detector_shape[1], dtype=np.uint64)
                        cumulative_rejected = 0
                        cumulative_haadf = 0
                        previous_exposure = 0
                        for exposure_output, exposure in enumerate(exposure_values):
                            delta = int(exposure - previous_exposure)
                            electron_count = delta * raw_config.electrons_per_integration
                            energy_increment = rng.multinomial(electron_count, probability).astype(
                                np.uint64
                            )
                            raw_increment, rejected_increment = _sample_raw_columns(
                                energy_increment,
                                mappings,
                                detector_shape[1],
                                rng,
                            )
                            projected_increment, mode = _sample_projected_charge(
                                raw_increment,
                                response_mean,
                                response_covariance,
                                response_bank,
                                raw_config.exact_response_max_electrons,
                                rng,
                            )
                            response_modes.add(mode)
                            noise_increment = rng.normal(
                                0.0,
                                np.sqrt(delta * column_noise_variance),
                            )
                            if row_common_variance > 0.0:
                                noise_increment += rng.normal(
                                    0.0, np.sqrt(delta * row_common_variance)
                                )
                            cumulative_incident += energy_increment
                            cumulative_raw_columns += raw_increment.astype(np.uint64)
                            cumulative_projected_pairs += projected_increment
                            cumulative_noise_adu += noise_increment
                            countable_increment = np.zeros(detector_shape[1], dtype=np.uint64)
                            countable_slice = np.s_[raw_config.counting_first_raw_column :]
                            countable_increment[countable_slice] = rng.binomial(
                                raw_increment[countable_slice],
                                raw_config.counting_detection_efficiency,
                            ).astype(np.uint64)
                            false_events = int(
                                rng.poisson(delta * raw_config.counting_dark_events_per_frame)
                            )
                            if false_events:
                                false_width = (
                                    detector_shape[1] - raw_config.counting_dark_first_raw_column
                                )
                                countable_increment[
                                    raw_config.counting_dark_first_raw_column :
                                ] += rng.multinomial(
                                    false_events,
                                    np.full(false_width, 1.0 / false_width),
                                ).astype(
                                    np.uint64
                                )
                            cumulative_counted_columns += countable_increment
                            cumulative_rejected += rejected_increment
                            cumulative_haadf += int(
                                rng.poisson(max(delta * haadf_per_integration, 0.0))
                            )
                            index = (
                                exposure_output,
                                depth_output,
                                y_output,
                                x_output,
                            )
                            incident[index] = cumulative_incident
                            accepted[index] = int(cumulative_raw_columns.sum())
                            rejected[index] = cumulative_rejected
                            expected_components[index] = component_per_integration * exposure
                            folded = _fold_raw_columns(
                                cumulative_raw_columns.astype(np.float64), config
                            )
                            pre_spectrum = _stitched_to_energy(folded, energy_edges, config)
                            pre_energy[index] = pre_spectrum
                            pre_components[index] = profile_pseudoinverse @ pre_spectrum
                            collapsed_adu = (
                                config.readout.gain_adu_per_electron * cumulative_projected_pairs
                                + cumulative_noise_adu
                            )
                            estimated_hits = collapsed_adu / (
                                config.readout.gain_adu_per_electron * float(response_totals.mean())
                            )
                            detector_spectrum = _stitched_to_energy(
                                _fold_raw_columns(estimated_hits, config),
                                energy_edges,
                                config,
                            )
                            detector_energy[index] = detector_spectrum
                            detector_components[index] = profile_pseudoinverse @ detector_spectrum
                            expected_false_columns = np.zeros(detector_shape[1], dtype=np.float64)
                            false_width = (
                                detector_shape[1] - raw_config.counting_dark_first_raw_column
                            )
                            expected_false_columns[raw_config.counting_dark_first_raw_column :] = (
                                exposure * raw_config.counting_dark_events_per_frame / false_width
                            )
                            corrected_counted_columns = (
                                cumulative_counted_columns.astype(np.float64)
                                - expected_false_columns
                            ) / raw_config.counting_detection_efficiency
                            counted_spectrum = _stitched_to_energy(
                                _fold_raw_columns(corrected_counted_columns, config),
                                energy_edges,
                                config,
                            )
                            counted_energy[index] = counted_spectrum
                            counted_components[index] = counting_pseudoinverse @ (
                                counted_spectrum[counting_energy_mask]
                            )
                            haadf_expected[index] = haadf_per_integration * exposure
                            haadf_counts[index] = cumulative_haadf
                            rejected_total += rejected_increment
                            previous_exposure = int(exposure)

            elements = reconstruction.create_group("elements")
            metric_summary: dict[str, list[dict[str, float]]] = {}
            for element, edge_labels in selected_element_edges.items():
                indices = [
                    component_labels.index(label)
                    for label in edge_labels
                    if label in component_labels
                ]
                if not indices:
                    continue
                group = elements.create_group(element)
                group.attrs["edge_labels_json"] = json.dumps(edge_labels)
                expected_map = expected_components[..., indices].sum(axis=-1)
                pre_map = pre_components[..., indices].sum(axis=-1)
                detector_map = detector_components[..., indices].sum(axis=-1)
                counted_map = counted_components[..., indices].sum(axis=-1)
                group.create_dataset("expected_counts", data=expected_map, compression="gzip")
                group.create_dataset("pre_readout_counts", data=pre_map, compression="gzip")
                group.create_dataset("detector_counts", data=detector_map, compression="gzip")
                group.create_dataset("counted_counts", data=counted_map, compression="gzip")
                pre_correlation = np.empty((len(exposure_values), len(selected_depths)))
                detector_correlation = np.empty_like(pre_correlation)
                counted_correlation = np.empty_like(pre_correlation)
                for exposure_index in range(len(exposure_values)):
                    for depth_index in range(len(selected_depths)):
                        reference = expected_map[exposure_index, depth_index]
                        pre_correlation[exposure_index, depth_index] = _correlation(
                            reference, pre_map[exposure_index, depth_index]
                        )
                        detector_correlation[exposure_index, depth_index] = _correlation(
                            reference, detector_map[exposure_index, depth_index]
                        )
                        counted_correlation[exposure_index, depth_index] = _correlation(
                            reference, counted_map[exposure_index, depth_index]
                        )
                group.create_dataset("pre_readout_correlation", data=pre_correlation)
                group.create_dataset("detector_correlation", data=detector_correlation)
                group.create_dataset("counted_correlation", data=counted_correlation)
                metric_summary[element] = [
                    {
                        "integrations": int(exposure),
                        "dwell_ms": float(1.0e3 * exposure / config.experiment.frame_rate_hz),
                        "pre_readout_correlation": float(pre_correlation[i, 0]),
                        "detector_correlation": float(detector_correlation[i, 0]),
                        "counted_correlation": float(counted_correlation[i, 0]),
                    }
                    for i, exposure in enumerate(exposure_values)
                ]

            summary: dict[str, object] = {
                "scan_shape": [
                    len(selected_depths),
                    len(y_indices),
                    len(x_indices),
                ],
                "exposures_integrations_per_position": exposure_values.tolist(),
                "dwell_ms": (1.0e3 * exposure_values / config.experiment.frame_rate_hz).tolist(),
                "electrons_per_integration": raw_config.electrons_per_integration,
                "maximum_electrons_per_position": int(
                    exposure_values[-1] * raw_config.electrons_per_integration
                ),
                "ideal_scan_time_s": float(
                    len(selected_depths)
                    * len(y_indices)
                    * len(x_indices)
                    * exposure_values[-1]
                    / config.experiment.frame_rate_hz
                ),
                "ideal_scan_time_minutes": float(
                    len(selected_depths)
                    * len(y_indices)
                    * len(x_indices)
                    * exposure_values[-1]
                    / (60.0 * config.experiment.frame_rate_hz)
                ),
                "total_integrations_at_maximum_exposure": int(
                    len(selected_depths) * len(y_indices) * len(x_indices) * exposure_values[-1]
                ),
                "raw_frames_retained": 0,
                "streaming_accumulation": True,
                "response_modes": sorted(response_modes),
                "response_mean_pairs": float(response_totals.mean()),
                "response_blur_sigma_pixels": raw_config.response_blur_sigma_pixels,
                "counting": {
                    "first_raw_column": raw_config.counting_first_raw_column,
                    "minimum_energy_eV": raw_config.counting_min_energy_eV,
                    "detection_efficiency": raw_config.counting_detection_efficiency,
                    "dark_events_per_frame": raw_config.counting_dark_events_per_frame,
                    "dark_first_raw_column": raw_config.counting_dark_first_raw_column,
                    "method": "hybrid: counted at/above validity boundary; analog channel retained below",
                },
                "spectrometer_rejected_electrons_across_increments": rejected_total,
                "element_metrics": metric_summary,
                "response_summary": response_summary,
                "approximations": [
                    "Per-readout ADC saturation is neglected; the calibrated one-frame scan showed one saturated pixel in approximately 944 million pixel reads.",
                    "High-occupancy response increments use a compound-Gaussian projection of the measured Geant4 event-template moments.",
                    "Independent ADC rounding noise is represented by variance 1/12 per pixel per readout.",
                    "Pedestal is assumed to be removed with the calibrated map before accumulation.",
                    "Sparse counting applies independent efficiency thinning and a uniform-in-tail dark false-event model; spatial pile-up is neglected at the modeled 1.8 countable events per readout.",
                ],
            }
            output.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
    return summary


def plot_exposure_sweep(
    input_hdf5: str | Path,
    output_png: str | Path,
    depth_selection_index: int = 0,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    with h5py.File(input_hdf5, "r") as h5:
        if h5.attrs.get("schema") != EXPOSURE_SWEEP_SCHEMA:
            raise ValueError("Input is not a raw DOEELS exposure sweep")
        dwell_ms = h5["axes/dwell_ms"][:]
        exposures = h5["axes/integrations_per_position"][:]
        elements = sorted(h5["reconstruction/elements"].keys())
        if not elements:
            raise ValueError("Exposure sweep contains no element maps")
        depth_count = h5["axes/depth_index"].shape[0]
        if not 0 <= depth_selection_index < depth_count:
            raise ValueError("depth selection index is outside the sweep")
        selected = np.unique(
            np.asarray(
                [
                    int(np.argmin(np.abs(exposures - target)))
                    for target in (300, 3000, exposures[-1])
                ]
            )
        )
        while len(selected) < 3 and len(selected) < len(exposures):
            candidates = [i for i in range(len(exposures)) if i not in selected]
            selected = np.sort(np.append(selected, candidates[-1]))

        figure = plt.figure(figsize=(4.2 * 4, 3.3 * (len(elements) + 1)), constrained_layout=True)
        grid = figure.add_gridspec(len(elements) + 1, 4)
        correlation_axis = figure.add_subplot(grid[0, :])
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(elements)))
        metrics: dict[str, dict[str, list[float]]] = {}
        for color, element in zip(colors, elements):
            group = h5[f"reconstruction/elements/{element}"]
            pre = group["pre_readout_correlation"][:, depth_selection_index]
            detector = group["detector_correlation"][:, depth_selection_index]
            counted = group["counted_correlation"][:, depth_selection_index]
            correlation_axis.plot(dwell_ms, counted, "o-", color=color, label=f"{element} counted")
            correlation_axis.plot(
                dwell_ms, pre, "--", color=color, alpha=0.65, label=f"{element} ideal"
            )
            correlation_axis.plot(dwell_ms, detector, ":", color=color, alpha=0.45)
            metrics[element] = {
                "counted": counted.tolist(),
                "integrating_raw_adc": detector.tolist(),
                "pre_readout": pre.tolist(),
            }
        correlation_axis.set_xscale("log")
        correlation_axis.set_ylim(-0.15, 1.05)
        correlation_axis.set(
            xlabel="dwell per probe position (ms)",
            ylabel="correlation with expected element map",
            title="Nested 16x16 LMTO exposure sweep",
        )
        correlation_axis.grid(alpha=0.25)
        correlation_axis.legend(ncol=min(4, len(elements)), fontsize=9)

        for row, element in enumerate(elements, start=1):
            group = h5[f"reconstruction/elements/{element}"]
            expected = group["expected_counts"][:, depth_selection_index]
            counted = group["counted_counts"][:, depth_selection_index]
            reference = expected[-1] / exposures[-1]
            reference = (reference - reference.mean()) / max(reference.std(), 1.0e-12)
            panels = [reference]
            titles = [f"{element} expected"]
            correlations = group["counted_correlation"][:, depth_selection_index]
            for exposure_index in selected[-3:]:
                image = counted[exposure_index] / exposures[exposure_index]
                image = (image - image.mean()) / max(image.std(), 1.0e-12)
                panels.append(image)
                titles.append(
                    f"{dwell_ms[exposure_index]:.2f} ms, r={correlations[exposure_index]:.2f}"
                )
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axis = figure.add_subplot(grid[row, column])
                axis.imshow(panel, cmap="coolwarm", vmin=-2.5, vmax=2.5, origin="lower")
                axis.set_title(title)
                axis.set_xticks([])
                axis.set_yticks([])
        output_path = Path(output_png)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170)
        plt.close(figure)
    return {
        "element_correlations": metrics,
        "selected_exposures": exposures[selected].tolist(),
    }


def _periodic_display_image(image: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Smooth and independently contrast-scale a reconstruction for display."""
    values = np.asarray(image, dtype=np.float64)
    if sigma_pixels > 0.0:
        rows, columns = values.shape
        frequency_y = np.fft.fftfreq(rows)
        frequency_x = np.fft.fftfreq(columns)
        transfer = np.exp(
            -2.0
            * np.pi**2
            * sigma_pixels**2
            * (frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2)
        )
        values = np.fft.ifft2(np.fft.fft2(values) * transfer).real
    lower, upper = np.quantile(values, (0.02, 0.995))
    if upper <= lower:
        upper = lower + 1.0
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0) ** 0.75


def write_exposure_slider_demo(
    input_hdf5: str | Path,
    output_html: str | Path,
    output_png: str | Path,
    elements: tuple[str, ...] = ("Mn", "O", "Ti"),
    exposure_index: int = -1,
    depth_selection_index: int = 0,
    display_sigma_pixels: float = 0.6,
) -> dict[str, object]:
    """Write a slider viewer and montage from detector-reconstructed maps."""
    from PIL import Image, ImageDraw

    if not elements:
        raise ValueError("at least one element channel is required")
    if display_sigma_pixels < 0.0:
        raise ValueError("display smoothing cannot be negative")
    with h5py.File(input_hdf5, "r") as h5:
        if h5.attrs.get("schema") != EXPOSURE_SWEEP_SCHEMA:
            raise ValueError("Input is not a raw DOEELS exposure sweep")
        exposure_count = len(h5["axes/integrations_per_position"])
        resolved_exposure = exposure_index % exposure_count
        depth_count = len(h5["axes/depth_index"])
        if not 0 <= depth_selection_index < depth_count:
            raise ValueError("depth selection index is outside the sweep")
        available = set(h5["reconstruction/elements"].keys())
        missing = set(elements) - available
        if missing:
            raise ValueError(f"Exposure sweep has no element map(s): {sorted(missing)}")

        dwell_ms = float(h5["axes/dwell_ms"][resolved_exposure])
        integrations = int(h5["axes/integrations_per_position"][resolved_exposure])
        source_depth_index = int(h5["axes/depth_index"][depth_selection_index])
        focal_depth_A = (
            float(h5["axes/focal_depth_A"][depth_selection_index])
            if "axes/focal_depth_A" in h5
            else float("nan")
        )
        scan_x_A = h5["axes/scan_x_A"][:]
        scan_y_A = h5["axes/scan_y_A"][:]
        step_x_A = float(np.median(np.diff(scan_x_A))) if len(scan_x_A) > 1 else 0.0
        step_y_A = float(np.median(np.diff(scan_y_A))) if len(scan_y_A) > 1 else 0.0
        extent_x_A = float(scan_x_A[-1] - scan_x_A[0] + step_x_A)
        extent_y_A = float(scan_y_A[-1] - scan_y_A[0] + step_y_A)

        raw_maps: dict[str, np.ndarray] = {
            "HAADF": h5["reconstruction/haadf_counts"][resolved_exposure, depth_selection_index]
        }
        correlations: dict[str, float] = {}
        for element in elements:
            group = h5[f"reconstruction/elements/{element}"]
            raw_maps[element] = group["counted_counts"][resolved_exposure, depth_selection_index]
            correlations[element] = float(
                group["counted_correlation"][resolved_exposure, depth_selection_index]
            )

    display_maps = {
        name: _periodic_display_image(values, display_sigma_pixels)
        for name, values in raw_maps.items()
    }
    size_y, size_x = display_maps["HAADF"].shape
    if size_y != size_x:
        raise ValueError("slider demo currently requires a square scan")
    channels = ["HAADF", *elements]
    encoded = {
        name: b64encode(
            np.rint(255.0 * display_maps[name][::-1]).astype(np.uint8).tobytes()
        ).decode("ascii")
        for name in channels
    }
    payload = {
        "channels": channels,
        "images": encoded,
        "colors": {name: list(DEMO_COLORS.get(name, (90, 170, 255))) for name in channels},
        "correlations": correlations,
        "size": size_x,
        "dwell_ms": dwell_ms,
        "integrations": integrations,
        "extent_x_nm": extent_x_A / 10.0,
        "extent_y_nm": extent_y_A / 10.0,
        "source_depth_index": source_depth_index,
        "focal_depth_A": focal_depth_A,
        "display_sigma_pixels": display_sigma_pixels,
    }
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LMTO detector reconstruction</title><style>
body{margin:0;background:#0d1118;color:#edf2f8;font:16px system-ui,sans-serif}.app{max-width:820px;margin:24px auto;padding:0 20px}
.panel{background:#171d27;border:1px solid #303949;border-radius:14px;padding:22px;box-shadow:0 14px 40px #0007}
h1{font-size:24px;margin:0 0 5px}.sub{color:#aeb9c8;margin-bottom:20px}.control{margin:0 auto 16px;max-width:640px}
label{display:flex;justify-content:space-between;color:#d9e1eb;margin-bottom:7px}input{width:100%}.ticks{display:flex;justify-content:space-between;color:#93a0b2;font-size:12px}
canvas{display:block;width:min(100%,640px);aspect-ratio:1;margin:auto;background:#000;border-radius:8px;image-rendering:auto}
.meta{display:flex;justify-content:space-between;gap:12px;color:#aeb9c8;margin:10px auto 0;max-width:640px}.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px}
.notice{margin-top:17px;padding:12px;background:#222a36;border-left:4px solid #efb34b;color:#dbe2eb;font-size:14px}
</style></head><body><div class="app"><div class="panel"><h1>LMTO STEM–EELS detector reconstruction</h1>
<div class="sub"><span id="scanSize"></span> probe positions · <span id="acq"></span></div><div class="control">
<label for="channel"><span>Display channel</span><b id="channelName"></b></label><input id="channel" type="range" min="0" step="1">
<div class="ticks" id="ticks"></div></div><canvas id="view"></canvas><div class="meta"><span id="legend"></span><span id="fov"></span></div>
<div class="notice"><b>Simulation note:</b> HAADF is a Poisson-sampled annular-detector signal. Mn, O, and Ti are fitted from spectra accumulated after spectrometer transfer, Monte Carlo silicon charge-cloud response, calibrated readout noise, and sparse-counting corrections. Each channel uses independent display contrast and <span id="smoothing"></span>-pixel smoothing.</div>
</div></div><script>const DATA=__PAYLOAD__;const slider=document.getElementById('channel');slider.max=DATA.channels.length-1;
document.getElementById('ticks').innerHTML=DATA.channels.map(x=>'<span>'+x+'</span>').join('');document.getElementById('scanSize').textContent=DATA.size+' × '+DATA.size;document.getElementById('acq').textContent=DATA.dwell_ms.toFixed(2)+' ms per position · '+DATA.integrations+' integrations';document.getElementById('smoothing').textContent=DATA.display_sigma_pixels.toFixed(2);
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),off=document.createElement('canvas');off.width=off.height=DATA.size;canvas.width=canvas.height=DATA.size;
function render(){const name=DATA.channels[Number(slider.value)],raw=atob(DATA.images[name]),bytes=new Uint8Array(raw.length),rgb=DATA.colors[name],image=ctx.createImageData(DATA.size,DATA.size);
for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);for(let i=0;i<bytes.length;i++){const j=4*i,v=bytes[i]/255;image.data[j]=rgb[0]*v;image.data[j+1]=rgb[1]*v;image.data[j+2]=rgb[2]*v;image.data[j+3]=255}
off.getContext('2d').putImageData(image,0,0);ctx.imageSmoothingEnabled=true;ctx.clearRect(0,0,DATA.size,DATA.size);ctx.drawImage(off,0,0);
document.getElementById('channelName').textContent=name;const metric=name==='HAADF'?'annular counts':'counted fit · r='+DATA.correlations[name].toFixed(3);
document.getElementById('legend').innerHTML='<span class="swatch" style="background:rgb('+rgb.join(',')+')"></span>'+name+' · '+metric;
document.getElementById('fov').textContent=DATA.extent_x_nm.toFixed(2)+' × '+DATA.extent_y_nm.toFixed(2)+' nm'}slider.oninput=render;render();</script></body></html>""".replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":"))
    )
    html_path = Path(output_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    panel_pixels = 360
    gap_pixels = 18
    header_pixels = 74
    canvas = Image.new(
        "RGB",
        (
            len(channels) * panel_pixels + (len(channels) - 1) * gap_pixels,
            panel_pixels + header_pixels,
        ),
        (13, 17, 24),
    )
    draw = ImageDraw.Draw(canvas)
    subtitle = (
        f"Detector-reconstructed LMTO scan | {dwell_ms:.2f} ms/position | "
        f"{extent_x_A / 10.0:.2f} nm field"
    )
    draw.text((8, 8), subtitle, fill=(235, 240, 247))
    for panel_index, name in enumerate(channels):
        color = DEMO_COLORS.get(name, (90, 170, 255))
        rgb = np.rint(
            display_maps[name][::-1, :, None] * np.asarray(color, dtype=np.float64)[None, None, :]
        ).astype(np.uint8)
        panel = Image.fromarray(rgb, mode="RGB").resize(
            (panel_pixels, panel_pixels), resample=Image.Resampling.BICUBIC
        )
        x_offset = panel_index * (panel_pixels + gap_pixels)
        canvas.paste(panel, (x_offset, header_pixels))
        title = name
        if name in correlations:
            title += f"  (r={correlations[name]:.3f})"
        draw.text((x_offset + 8, 42), title, fill=color)
    png_path = Path(output_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    return payload
