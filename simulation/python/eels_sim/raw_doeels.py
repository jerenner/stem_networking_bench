from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import SimulationConfig, load_config
from .io import read_readout_calibration, read_transport_output
from .readout import digitize, select_calibrated_maps
from .transport import geometry_from_run_info, transport_deposits

RESPONSE_KERNEL_SCHEMA = "eels-sim-monte-carlo-response-kernel-v1"
RAW_DOEELS_SCHEMA = "eels-sim-raw-doeels-scan-v1"


def build_monte_carlo_response_kernel(
    config_path: str | Path,
    geant4_base: str | Path,
    output_hdf5: str | Path,
    radius_pixels: int = 3,
) -> dict[str, object]:
    if radius_pixels < 1:
        raise ValueError("response-kernel radius must be positive")
    config = load_config(config_path)
    tables = read_transport_output(geant4_base)
    geometry = geometry_from_run_info(tables["run_info"])
    charge, transport_summary = transport_deposits(
        tables["deposits"],
        geometry,
        config.transport,
        primaries_per_frame=1,
        random_seed=config.framing.random_seed,
    )
    primaries = tables["primaries"]
    event_count = len(primaries["event_id"])
    if charge.shape[0] != event_count:
        raise ValueError("Monte Carlo response frames do not match primary count")
    size = 2 * radius_pixels + 1
    accumulated = np.zeros((size, size), dtype=np.float64)
    event_spatial_pairs = np.zeros((event_count, size, size), dtype=np.uint32)
    captured_pairs = np.zeros(event_count, dtype=np.uint32)
    total_pairs = charge.sum(axis=(1, 2), dtype=np.uint64).astype(np.uint32)
    extent_x_um = geometry.columns * geometry.pixel_pitch_um
    extent_y_um = geometry.rows * geometry.pixel_pitch_um
    impact_columns = np.floor(
        (primaries["x_um"] + extent_x_um / 2.0) / geometry.pixel_pitch_um
    ).astype(int)
    impact_rows = np.floor(
        (primaries["y_um"] + extent_y_um / 2.0) / geometry.pixel_pitch_um
    ).astype(int)
    phase_x = (
        np.mod(primaries["x_um"] + extent_x_um / 2.0, geometry.pixel_pitch_um)
        / geometry.pixel_pitch_um
    )
    phase_y = (
        np.mod(primaries["y_um"] + extent_y_um / 2.0, geometry.pixel_pitch_um)
        / geometry.pixel_pitch_um
    )
    phase_x_histogram = np.histogram(phase_x, bins=10, range=(0.0, 1.0))[0]
    phase_y_histogram = np.histogram(phase_y, bins=10, range=(0.0, 1.0))[0]
    for index, (row, column) in enumerate(zip(impact_rows, impact_columns)):
        row_start = max(0, row - radius_pixels)
        row_stop = min(geometry.rows, row + radius_pixels + 1)
        column_start = max(0, column - radius_pixels)
        column_stop = min(geometry.columns, column + radius_pixels + 1)
        kernel_row_start = radius_pixels - (row - row_start)
        kernel_column_start = radius_pixels - (column - column_start)
        patch = charge[index, row_start:row_stop, column_start:column_stop]
        accumulated[
            kernel_row_start : kernel_row_start + patch.shape[0],
            kernel_column_start : kernel_column_start + patch.shape[1],
        ] += patch
        event_spatial_pairs[
            index,
            kernel_row_start : kernel_row_start + patch.shape[0],
            kernel_column_start : kernel_column_start + patch.shape[1],
        ] = patch
        captured_pairs[index] = int(patch.sum())
    if accumulated.sum() <= 0.0:
        raise RuntimeError("Monte Carlo response kernel contains no collected charge")
    mean_spatial_fraction = accumulated / accumulated.sum()
    capture_fraction = float(captured_pairs.sum() / max(total_pairs.sum(), 1))
    summary: dict[str, object] = {
        "event_count": event_count,
        "radius_pixels": radius_pixels,
        "mean_pairs": float(total_pairs.mean()),
        "median_pairs": float(np.median(total_pairs)),
        "std_pairs": float(total_pairs.std(ddof=1)),
        "p10_pairs": float(np.quantile(total_pairs, 0.1)),
        "p90_pairs": float(np.quantile(total_pairs, 0.9)),
        "kernel_capture_fraction": capture_fraction,
        "subpixel_phase_histogram_x": phase_x_histogram.tolist(),
        "subpixel_phase_histogram_y": phase_y_histogram.tolist(),
        "subpixel_phase_histogram_bins": 10,
        "geometry": asdict(geometry),
        "transport_summary": transport_summary,
        "approximation": (
            "aligned per-electron Monte Carlo charge templates; renderers may use "
            "the stored ensemble mean only for explicitly bounded high-dose runs"
        ),
    }
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = RESPONSE_KERNEL_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["geant4_base"] = str(Path(geant4_base).resolve())
        h5.attrs["config_file"] = str(Path(config_path).resolve())
        h5.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5.create_dataset("event_total_pairs", data=total_pairs, compression="gzip")
        h5.create_dataset("event_captured_pairs", data=captured_pairs, compression="gzip")
        h5.create_dataset("event_spatial_pairs", data=event_spatial_pairs, compression="gzip")
        h5.create_dataset("mean_spatial_fraction", data=mean_spatial_fraction)
    return summary


def _read_response_kernel(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict]:
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != RESPONSE_KERNEL_SCHEMA:
            raise ValueError(f"Unsupported detector response kernel in {path}")
        return (
            h5["event_total_pairs"][:].astype(np.float64),
            (
                h5["event_spatial_pairs"][:].astype(np.float64)
                if "event_spatial_pairs" in h5
                else None
            ),
            h5["mean_spatial_fraction"][:].astype(np.float64),
            json.loads(h5.attrs["summary_json"]),
        )


def _prepare_response_kernel(
    response_totals: np.ndarray,
    response_templates: np.ndarray | None,
    spatial_kernel: np.ndarray,
    blur_sigma_pixels: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    if blur_sigma_pixels < 0.0:
        raise ValueError("raw_doeels.response_blur_sigma_pixels cannot be negative")
    if blur_sigma_pixels == 0.0:
        return response_templates, spatial_kernel
    if response_templates is not None:
        blurred_templates = gaussian_filter(
            response_templates,
            sigma=(0.0, blur_sigma_pixels, blur_sigma_pixels),
            mode="constant",
        )
        blurred_totals = blurred_templates.sum(axis=(1, 2))
        blurred_templates *= np.divide(
            response_totals,
            blurred_totals,
            out=np.ones_like(response_totals),
            where=blurred_totals > 0.0,
        )[:, None, None]
        spatial_kernel = blurred_templates.sum(axis=0)
        spatial_kernel /= spatial_kernel.sum()
        return blurred_templates, spatial_kernel
    spatial_kernel = gaussian_filter(
        spatial_kernel,
        sigma=blur_sigma_pixels,
        mode="constant",
    )
    spatial_kernel /= spatial_kernel.sum()
    return None, spatial_kernel


def _energy_to_raw_column_mapping(
    energy_edges_eV: np.ndarray,
    config: SimulationConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    spectrometer = config.spectrometer
    subsamples = config.raw_doeels.energy_mapping_subsamples
    if subsamples <= 0:
        raise ValueError("raw_doeels.energy_mapping_subsamples must be positive")
    blur_nodes, blur_weights = np.polynomial.hermite.hermgauss(7)
    blur_weights = blur_weights / np.sqrt(np.pi)
    mappings: list[tuple[np.ndarray, np.ndarray]] = []
    lane_offset = (spectrometer.zlp_repeats - 1) * spectrometer.zlp_lane_width_columns
    for lower, upper in zip(energy_edges_eV[:-1], energy_edges_eV[1:]):
        losses = lower + (np.arange(subsamples) + 0.5) / subsamples * (upper - lower)
        stitched_centers = (
            spectrometer.zero_loss_stitched_column
            + losses / spectrometer.dispersion_eV_per_column
            + spectrometer.dispersion_quadratic_columns_per_eV2 * losses**2
        )
        weights: dict[int, float] = {}
        dispersion_derivative = (
            1.0 / spectrometer.dispersion_eV_per_column
            + 2.0 * spectrometer.dispersion_quadratic_columns_per_eV2 * losses
        )
        blur_sigma_columns = np.sqrt(
            (dispersion_derivative * spectrometer.energy_blur_sigma_eV) ** 2
            + (spectrometer.point_spread_sigma_x_um / spectrometer.pixel_pitch_um) ** 2
        )
        for center, sigma_columns in zip(stitched_centers, blur_sigma_columns):
            if sigma_columns > 0.0:
                coordinates = center + np.sqrt(2.0) * sigma_columns * blur_nodes
                coordinate_weights = blur_weights
            else:
                coordinates = np.asarray([center])
                coordinate_weights = np.asarray([1.0])
            for coordinate, blur_weight in zip(coordinates, coordinate_weights):
                if coordinate < spectrometer.zlp_lane_width_columns - 0.5:
                    raw_coordinates = [
                        coordinate + lane * spectrometer.zlp_lane_width_columns
                        for lane in range(spectrometer.zlp_repeats)
                    ]
                    base_weight = float(blur_weight) / (subsamples * spectrometer.zlp_repeats)
                else:
                    raw_coordinates = [coordinate + lane_offset]
                    base_weight = float(blur_weight) / subsamples
                for raw_coordinate in raw_coordinates:
                    left = int(np.floor(raw_coordinate))
                    fraction = raw_coordinate - left
                    for column, weight in (
                        (left, 1.0 - fraction),
                        (left + 1, fraction),
                    ):
                        if 0 <= column < spectrometer.detector_columns and weight > 0.0:
                            weights[column] = weights.get(column, 0.0) + base_weight * weight
        columns = np.asarray(sorted(weights), dtype=np.int32)
        probabilities = np.asarray([weights[column] for column in columns])
        mappings.append((columns, probabilities))
    return mappings


def _sample_raw_columns(
    energy_counts: np.ndarray,
    mappings: list[tuple[np.ndarray, np.ndarray]],
    detector_columns: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    raw_columns = np.zeros(detector_columns, dtype=np.uint32)
    rejected = 0
    for count, (columns, probabilities) in zip(energy_counts, mappings):
        if count == 0:
            continue
        lost_probability = max(0.0, 1.0 - float(probabilities.sum()))
        p = np.append(probabilities, lost_probability)
        p /= p.sum()
        sampled = rng.multinomial(int(count), p)
        raw_columns[columns] += sampled[:-1].astype(np.uint32)
        rejected += int(sampled[-1])
    return raw_columns, rejected


def _transferred_component_profiles(
    component_profiles: np.ndarray,
    mappings: list[tuple[np.ndarray, np.ndarray]],
    energy_edges_eV: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Apply the deterministic spectrometer/folding transfer to fit templates."""
    transferred = np.zeros_like(component_profiles, dtype=np.float64)
    for component_index, profile in enumerate(component_profiles):
        raw_columns = np.zeros(config.spectrometer.detector_columns, dtype=np.float64)
        for weight, (columns, probabilities) in zip(profile, mappings):
            if weight != 0.0:
                raw_columns[columns] += weight * probabilities
        transferred[component_index] = _stitched_to_energy(
            _fold_raw_columns(raw_columns, config), energy_edges_eV, config
        )
    return transferred


def _shift_convolve(source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rows, columns = source.shape
    radius_y = kernel.shape[0] // 2
    radius_x = kernel.shape[1] // 2
    output = np.zeros_like(source, dtype=np.float64)
    for kernel_y, kernel_x in np.argwhere(kernel > 1.0e-8):
        dy = int(kernel_y - radius_y)
        dx = int(kernel_x - radius_x)
        source_y0, source_y1 = max(0, -dy), min(rows, rows - dy)
        source_x0, source_x1 = max(0, -dx), min(columns, columns - dx)
        output[
            source_y0 + dy : source_y1 + dy,
            source_x0 + dx : source_x1 + dx,
        ] += (
            source[source_y0:source_y1, source_x0:source_x1] * kernel[kernel_y, kernel_x]
        )
    return output


def _sample_detector_charge(
    repeated_pixels: np.ndarray,
    detector_shape: tuple[int, int],
    response_totals: np.ndarray,
    response_templates: np.ndarray | None,
    mean_spatial_kernel: np.ndarray,
    exact_response_max_electrons: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if exact_response_max_electrons < 0:
        raise ValueError("raw_doeels.exact_response_max_electrons cannot be negative")
    use_exact = (
        response_templates is not None and len(repeated_pixels) <= exact_response_max_electrons
    )
    if not use_exact:
        sampled_pairs = rng.choice(response_totals, size=len(repeated_pixels), replace=True)
        source_charge = np.bincount(
            repeated_pixels,
            weights=sampled_pairs,
            minlength=int(np.prod(detector_shape)),
        ).reshape(detector_shape)
        return (
            _shift_convolve(source_charge, mean_spatial_kernel),
            "sampled_total_mean_template",
        )

    template_indices = rng.integers(0, response_templates.shape[0], size=len(repeated_pixels))
    rows, columns = detector_shape
    radius_y = response_templates.shape[1] // 2
    radius_x = response_templates.shape[2] // 2
    charge = np.zeros(detector_shape, dtype=np.float64)
    source_rows, source_columns = np.divmod(repeated_pixels, columns)
    for kernel_y in range(response_templates.shape[1]):
        for kernel_x in range(response_templates.shape[2]):
            weights = response_templates[template_indices, kernel_y, kernel_x]
            if not np.any(weights):
                continue
            dy = kernel_y - radius_y
            dx = kernel_x - radius_x
            destination_rows = source_rows + dy
            destination_columns = source_columns + dx
            valid = (
                (weights != 0.0)
                & (destination_rows >= 0)
                & (destination_rows < rows)
                & (destination_columns >= 0)
                & (destination_columns < columns)
            )
            np.add.at(
                charge,
                (destination_rows[valid], destination_columns[valid]),
                weights[valid],
            )
    return charge, "exact_event_template"


def _fold_raw_columns(raw_columns: np.ndarray, config: SimulationConfig) -> np.ndarray:
    spectrometer = config.spectrometer
    lane_width = spectrometer.zlp_lane_width_columns
    lane_count = spectrometer.zlp_repeats
    lane_offset = (lane_count - 1) * lane_width
    stitched_count = spectrometer.detector_columns - lane_offset
    stitched = np.zeros(stitched_count, dtype=np.float64)
    for lane in range(lane_count):
        start = lane * lane_width
        stitched[:lane_width] += raw_columns[start : start + lane_width]
    stitched[lane_width:] = raw_columns[lane_count * lane_width :]
    return stitched


def _stitched_to_energy(
    stitched: np.ndarray,
    energy_edges_eV: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    columns = np.arange(len(stitched), dtype=np.float64)
    # The current raw demonstrator uses the configured linear dispersion; the
    # quadratic inverse will be added when a nonzero coefficient is calibrated.
    losses = (
        columns - config.spectrometer.zero_loss_stitched_column
    ) * config.spectrometer.dispersion_eV_per_column
    indices = np.searchsorted(energy_edges_eV, losses, side="right") - 1
    valid = (indices >= 0) & (indices < len(energy_edges_eV) - 1)
    return np.bincount(indices[valid], weights=stitched[valid], minlength=len(energy_edges_eV) - 1)


def simulate_raw_doeels_scan(
    config_path: str | Path,
    spectrum_image_hdf5: str | Path,
    output_hdf5: str | Path,
    scan_stride: int | None = None,
    depth_indices: list[int] | None = None,
    integrations_per_position: int | None = None,
    electrons_per_integration: int | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    raw_config = config.raw_doeels
    raw_config = replace(
        raw_config,
        scan_stride=(raw_config.scan_stride if scan_stride is None else scan_stride),
        depth_indices=(raw_config.depth_indices if depth_indices is None else depth_indices),
        integrations_per_position=(
            raw_config.integrations_per_position
            if integrations_per_position is None
            else integrations_per_position
        ),
        electrons_per_integration=(
            raw_config.electrons_per_integration
            if electrons_per_integration is None
            else electrons_per_integration
        ),
    )
    if raw_config.response_kernel_file is None:
        raise ValueError("raw_doeels.response_kernel_file is required")
    if raw_config.electrons_per_integration <= 0:
        raise ValueError("raw_doeels.electrons_per_integration must be positive")
    if raw_config.integrations_per_position <= 0 or raw_config.scan_stride <= 0:
        raise ValueError("raw DOEELS integration count and scan stride must be positive")
    (
        response_totals,
        response_templates,
        spatial_kernel,
        response_summary,
    ) = _read_response_kernel(raw_config.response_kernel_file)
    response_templates, spatial_kernel = _prepare_response_kernel(
        response_totals,
        response_templates,
        spatial_kernel,
        raw_config.response_blur_sigma_pixels,
    )
    calibration = (
        None
        if config.readout.calibration_file is None
        else read_readout_calibration(config.readout.calibration_file)
    )
    pedestal_map, noise_map, signal_efficiency_map = select_calibrated_maps(
        calibration, config.readout
    )
    detector_shape = (
        config.spectrometer.detector_rows,
        config.spectrometer.detector_columns,
    )
    for name, array in (("pedestal", pedestal_map), ("noise", noise_map)):
        if array is not None and array.shape != detector_shape:
            raise ValueError(f"{name} calibration shape does not match DOEELS detector")
    pedestal_for_reconstruction = (
        np.full(detector_shape, config.readout.pedestal_adu, dtype=np.float64)
        if pedestal_map is None
        else pedestal_map.astype(np.float64)
    )
    rows = np.arange(config.spectrometer.detector_rows, dtype=np.float64)
    row_probability = np.exp(
        -0.5 * ((rows - config.spectrometer.zero_y_row) / raw_config.detector_row_sigma_pixels) ** 2
    )
    row_probability /= row_probability.sum()

    with h5py.File(spectrum_image_hdf5, "r") as source:
        if source.attrs.get("schema") != "eels-sim-spectrum-image-v1":
            raise ValueError("Input is not an EELS spectrum image")
        spectra = source["spectrum/counts"]
        energy_edges_eV = source["axes/energy_edges_eV"][:]
        component_profiles = source["spectrum/component_profiles"][:]
        component_labels = json.loads(source["spectrum"].attrs["component_labels_json"])
        source_summary = json.loads(source.attrs.get("summary_json", "{}"))
        selected_element_edges = source_summary.get("selected_element_edges", {})
        source_component_totals = source.get("spectrum/component_total_counts")
        if source_component_totals is None:
            raise ValueError("Spectrum image is missing spectrum/component_total_counts")
        source_expected_components = source.get("spectrum/expected_component_counts")
        input_depths, scan_rows, scan_columns, energy_bins = spectra.shape
        depth_indices = np.asarray(raw_config.depth_indices, dtype=int)
        if np.any(depth_indices < 0) or np.any(depth_indices >= input_depths):
            raise ValueError("raw_doeels.depth_indices are outside the spectrum image")
        y_indices = np.arange(0, scan_rows, raw_config.scan_stride, dtype=int)
        x_indices = np.arange(0, scan_columns, raw_config.scan_stride, dtype=int)
        frame_count = (
            len(depth_indices)
            * len(y_indices)
            * len(x_indices)
            * raw_config.integrations_per_position
        )
        mappings = _energy_to_raw_column_mapping(energy_edges_eV, config)
        output_path = Path(output_hdf5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(raw_config.random_seed)
        with h5py.File(output_path, "w") as output:
            output.attrs["schema"] = RAW_DOEELS_SCHEMA
            output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["config_file"] = str(Path(config_path).resolve())
            output.attrs["source_spectrum_image"] = str(Path(spectrum_image_hdf5).resolve())
            output.attrs["response_kernel"] = str(Path(raw_config.response_kernel_file).resolve())
            output.attrs["component_labels_json"] = json.dumps(component_labels)
            output.attrs["raw_config_json"] = json.dumps(asdict(raw_config), sort_keys=True)
            axes_group = output.create_group("axes")
            axes_group.create_dataset("energy_edges_eV", data=energy_edges_eV)
            axes_group.create_dataset(
                "energy_centers_eV",
                data=0.5 * (energy_edges_eV[:-1] + energy_edges_eV[1:]),
            )
            for source_name, output_name, selection in (
                ("focal_depth_A", "focal_depth_A", depth_indices),
                ("scan_y_A", "scan_y_A", y_indices),
                ("scan_x_A", "scan_x_A", x_indices),
            ):
                if source_name in source["axes"]:
                    axes_group.create_dataset(
                        output_name, data=source["axes"][source_name][selection]
                    )
            frames_group = output.create_group("frames")
            frames_group.attrs["axis_order"] = "frame,row,column"
            raw_dataset = frames_group.create_dataset(
                "raw",
                shape=(frame_count, *detector_shape),
                dtype=np.uint16,
                chunks=(
                    1,
                    min(120, detector_shape[0]),
                    min(480, detector_shape[1]),
                ),
                compression="gzip",
                compression_opts=raw_config.compression_level,
                shuffle=True,
            )
            raw_dataset.attrs["units"] = "ADU"
            scan_group = output.create_group("scan")
            frame_depth = scan_group.create_dataset("frame_depth_index", (frame_count,), "u2")
            frame_y = scan_group.create_dataset("frame_y_index", (frame_count,), "u2")
            frame_x = scan_group.create_dataset("frame_x_index", (frame_count,), "u2")
            frame_integration = scan_group.create_dataset(
                "frame_integration_index", (frame_count,), "u2"
            )
            truth = output.create_group("truth")
            incident_dataset = truth.create_dataset(
                "incident_spectrum_counts",
                (frame_count, energy_bins),
                "u4",
                compression="gzip",
            )
            column_dataset = (
                truth.create_dataset(
                    "raw_column_hit_counts",
                    (frame_count, detector_shape[1]),
                    "u4",
                    compression="gzip",
                )
                if raw_config.keep_hit_counts
                else None
            )
            accepted_dataset = truth.create_dataset("accepted_electrons", (frame_count,), "u4")
            charge_dataset = truth.create_dataset("generated_charge_pairs", (frame_count,), "u8")
            saturation_dataset = truth.create_dataset("saturated_pixels", (frame_count,), "u4")
            expected_components = truth.create_dataset(
                "expected_component_counts",
                (frame_count, len(component_labels)),
                "f4",
                compression="gzip",
            )
            reconstruction = output.create_group("reconstruction")
            pre_readout_energy = reconstruction.create_dataset(
                "pre_readout_energy_counts",
                (frame_count, energy_bins),
                "f4",
                compression="gzip",
            )
            pre_readout_components = reconstruction.create_dataset(
                "pre_readout_component_counts",
                (frame_count, len(component_labels)),
                "f4",
                compression="gzip",
            )
            reconstructed_energy = reconstruction.create_dataset(
                "energy_counts", (frame_count, energy_bins), "f4", compression="gzip"
            )
            reconstructed_components = reconstruction.create_dataset(
                "component_counts",
                (frame_count, len(component_labels)),
                "f4",
                compression="gzip",
            )
            collapsed_dataset = reconstruction.create_dataset(
                "collapsed_signal_adu",
                (frame_count, detector_shape[1]),
                "f4",
                compression="gzip",
            )
            baseline_dataset = reconstruction.create_dataset(
                "column_baseline_adu",
                (frame_count, detector_shape[1]),
                "f4",
                compression="gzip",
            )
            reconstruction.attrs["baseline_method"] = (
                "per-column least-squares fit of the configured Gaussian row "
                "profile plus a constant baseline"
            )
            reconstruction.attrs["component_fit"] = (
                "signed linear least-squares coefficients against the source "
                "component profiles; negative values are retained to expose noise"
            )
            fit_profiles = _transferred_component_profiles(
                component_profiles, mappings, energy_edges_eV, config
            )
            profile_pseudoinverse = np.linalg.pinv(fit_profiles.T, rcond=1.0e-5)
            map_shape = (
                len(depth_indices),
                len(y_indices),
                len(x_indices),
                len(component_labels),
            )
            expected_component_maps = np.zeros(map_shape, dtype=np.float64)
            pre_readout_component_maps = np.zeros(map_shape, dtype=np.float64)
            detector_component_maps = np.zeros(map_shape, dtype=np.float64)
            frame_index = 0
            rejected_total = 0
            detector_response_modes: set[str] = set()
            for map_depth_index, depth_index in enumerate(depth_indices):
                for map_y_index, y_index in enumerate(y_indices):
                    for map_x_index, x_index in enumerate(x_indices):
                        if source_expected_components is None:
                            probability = spectra[depth_index, y_index, x_index].astype(np.float64)
                            source_components = source_component_totals[
                                depth_index, y_index, x_index
                            ].astype(np.float64)
                        else:
                            source_components = source_expected_components[
                                depth_index, y_index, x_index
                            ].astype(np.float64)
                            probability = source_components @ component_profiles
                        source_electrons = float(probability.sum())
                        probability /= source_electrons
                        component_expectation = (
                            source_components
                            * raw_config.electrons_per_integration
                            / source_electrons
                        )
                        for integration_index in range(raw_config.integrations_per_position):
                            energy_counts = rng.multinomial(
                                raw_config.electrons_per_integration, probability
                            ).astype(np.uint32)
                            raw_column_hits, rejected = _sample_raw_columns(
                                energy_counts,
                                mappings,
                                detector_shape[1],
                                rng,
                            )
                            rejected_total += rejected
                            hits = rng.multinomial(raw_column_hits, row_probability).T.astype(
                                np.uint32
                            )
                            flat_hits = hits.ravel()
                            occupied = np.flatnonzero(flat_hits)
                            repeated_pixels = np.repeat(occupied, flat_hits[occupied])
                            charge, response_mode = _sample_detector_charge(
                                repeated_pixels,
                                detector_shape,
                                response_totals,
                                response_templates,
                                spatial_kernel,
                                raw_config.exact_response_max_electrons,
                                rng,
                            )
                            detector_response_modes.add(response_mode)
                            _, raw = digitize(
                                charge[None],
                                config.readout,
                                raw_config.random_seed + frame_index + 1,
                                pedestal_map,
                                noise_map,
                                signal_efficiency_map,
                            )
                            raw_frame = raw[0]
                            raw_dataset[frame_index] = raw_frame
                            frame_depth[frame_index] = depth_index
                            frame_y[frame_index] = y_index
                            frame_x[frame_index] = x_index
                            frame_integration[frame_index] = integration_index
                            incident_dataset[frame_index] = energy_counts
                            if column_dataset is not None:
                                column_dataset[frame_index] = raw_column_hits
                            accepted_dataset[frame_index] = int(raw_column_hits.sum())
                            charge_dataset[frame_index] = int(round(charge.sum()))
                            saturation_dataset[frame_index] = int(
                                np.count_nonzero(raw_frame == (1 << config.readout.adc_bits) - 1)
                            )
                            expected_components[frame_index] = component_expectation
                            pre_energy = _stitched_to_energy(
                                _fold_raw_columns(raw_column_hits.astype(np.float64), config),
                                energy_edges_eV,
                                config,
                            )
                            pre_coefficients = profile_pseudoinverse @ pre_energy
                            pre_readout_energy[frame_index] = pre_energy
                            pre_readout_components[frame_index] = pre_coefficients
                            residual = raw_frame.astype(np.float64) - pedestal_for_reconstruction
                            centered_profile = row_probability - row_probability.mean()
                            profile_denominator = float(np.dot(centered_profile, row_probability))
                            collapsed = centered_profile @ residual / profile_denominator
                            column_baseline = residual.mean(axis=0) - (
                                collapsed / detector_shape[0]
                            )
                            collapsed_dataset[frame_index] = collapsed
                            baseline_dataset[frame_index] = column_baseline
                            estimated_raw_hits = collapsed / (
                                config.readout.gain_adu_per_electron * float(response_totals.mean())
                            )
                            energy_reconstruction = _stitched_to_energy(
                                _fold_raw_columns(estimated_raw_hits, config),
                                energy_edges_eV,
                                config,
                            )
                            reconstructed_energy[frame_index] = energy_reconstruction
                            coefficients = profile_pseudoinverse @ energy_reconstruction
                            reconstructed_components[frame_index] = coefficients
                            map_index = (
                                map_depth_index,
                                map_y_index,
                                map_x_index,
                            )
                            expected_component_maps[map_index] += component_expectation
                            pre_readout_component_maps[map_index] += pre_coefficients
                            detector_component_maps[map_index] += coefficients
                            frame_index += 1
            maps = reconstruction.create_group("scan_maps")
            maps.attrs["axis_order"] = "depth,y,x,component"
            maps.attrs["component_labels_json"] = json.dumps(component_labels)
            maps.attrs["selected_element_edges_json"] = json.dumps(
                selected_element_edges, sort_keys=True
            )
            maps.create_dataset("depth_index", data=depth_indices.astype(np.uint16))
            maps.create_dataset("y_index", data=y_indices.astype(np.uint16))
            maps.create_dataset("x_index", data=x_indices.astype(np.uint16))
            maps.create_dataset(
                "expected_component_counts",
                data=expected_component_maps.astype(np.float32),
            )
            maps.create_dataset(
                "pre_readout_component_counts",
                data=pre_readout_component_maps.astype(np.float32),
            )
            maps.create_dataset(
                "detector_component_counts",
                data=detector_component_maps.astype(np.float32),
            )
            element_maps = maps.create_group("elements")
            for element, edge_labels in selected_element_edges.items():
                indices = [
                    component_labels.index(label)
                    for label in edge_labels
                    if label in component_labels
                ]
                if not indices:
                    continue
                group = element_maps.create_group(element)
                group.attrs["edge_labels_json"] = json.dumps(edge_labels)
                group.create_dataset(
                    "expected_counts",
                    data=expected_component_maps[..., indices].sum(axis=-1).astype(np.float32),
                )
                group.create_dataset(
                    "pre_readout_counts",
                    data=pre_readout_component_maps[..., indices].sum(axis=-1).astype(np.float32),
                )
                group.create_dataset(
                    "detector_counts",
                    data=detector_component_maps[..., indices].sum(axis=-1).astype(np.float32),
                )
            summary = {
                "frame_count": frame_count,
                "detector_shape": list(detector_shape),
                "scan_depth_indices": depth_indices.tolist(),
                "scan_y_indices": y_indices.tolist(),
                "scan_x_indices": x_indices.tolist(),
                "integrations_per_position": raw_config.integrations_per_position,
                "electrons_per_integration": raw_config.electrons_per_integration,
                "total_incident_electrons": frame_count * raw_config.electrons_per_integration,
                "spectrometer_rejected_electrons": rejected_total,
                "response_mean_pairs": float(response_totals.mean()),
                "response_blur_sigma_pixels": raw_config.response_blur_sigma_pixels,
                "detector_response_modes": sorted(detector_response_modes),
                "equivalent_87khz_integrations_per_frame": (
                    raw_config.electrons_per_integration
                    / config.experiment.expected_primaries_per_frame
                ),
                "response_summary": response_summary,
                "warning": (
                    "effective gain and residual spreading are fitted from the NiO "
                    "15 pA sparse tail; detector geometry, row profile, and "
                    "spectrometer calibration remain provisional"
                ),
            }
            output.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
    return summary


def _pearson_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    candidate = np.asarray(candidate, dtype=np.float64).ravel()
    if reference.std() == 0.0 or candidate.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(reference, candidate)[0, 1])


def plot_raw_doeels_diagnostics(
    raw_hdf5: str | Path,
    output_png: str | Path,
    depth_selection_index: int = 0,
    frame_index: int = 0,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    with h5py.File(raw_hdf5, "r") as h5:
        if h5.attrs.get("schema") != RAW_DOEELS_SCHEMA:
            raise ValueError("Input is not a raw DOEELS scan")
        maps = h5["reconstruction/scan_maps/elements"]
        elements = sorted(maps.keys())
        if not elements:
            raise ValueError("Raw DOEELS scan has no selected element maps")
        depth_count = h5["reconstruction/scan_maps/depth_index"].shape[0]
        if not 0 <= depth_selection_index < depth_count:
            raise ValueError("depth selection index is outside the raw scan")
        if not 0 <= frame_index < h5["frames/raw"].shape[0]:
            raise ValueError("frame index is outside the raw scan")
        energy = h5["axes/energy_centers_eV"][:]
        incident = h5["truth/incident_spectrum_counts"][:].sum(axis=0)
        pre_readout = h5["reconstruction/pre_readout_energy_counts"][:].sum(axis=0)
        detector = h5["reconstruction/energy_counts"][:].sum(axis=0)
        raw_frame = h5["frames/raw"][frame_index].astype(np.float64)
        correlations: dict[str, dict[str, float]] = {}
        element_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for element in elements:
            group = maps[element]
            expected = group["expected_counts"][depth_selection_index]
            pre_map = group["pre_readout_counts"][depth_selection_index]
            detector_map = group["detector_counts"][depth_selection_index]
            element_data[element] = (expected, pre_map, detector_map)
            correlations[element] = {
                "pre_readout": _pearson_correlation(expected, pre_map),
                "detector": _pearson_correlation(expected, detector_map),
            }

    figure = plt.figure(figsize=(13.5, 3.0 + 2.7 * len(elements)), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=(1.1, len(elements)))
    top = outer[0].subgridspec(1, 2, width_ratios=(1.35, 1.0))
    raw_axis = figure.add_subplot(top[0])
    spectrum_axis = figure.add_subplot(top[1])
    raw_centered = raw_frame - np.median(raw_frame, axis=0, keepdims=True)
    raw_limit = max(float(np.quantile(np.abs(raw_centered), 0.995)), 1.0)
    raw_image = raw_axis.imshow(
        raw_centered,
        aspect="auto",
        cmap="coolwarm",
        vmin=-raw_limit,
        vmax=raw_limit,
        interpolation="nearest",
    )
    raw_axis.set_title(f"Raw DOEELS frame {frame_index} (column pedestal removed)")
    raw_axis.set_xlabel("detector column")
    raw_axis.set_ylabel("detector row")
    figure.colorbar(raw_image, ax=raw_axis, label="ADU")
    spectrum_axis.plot(energy, incident, label="specimen output", linewidth=1.3)
    spectrum_axis.plot(energy, pre_readout, label="after spectrometer", linewidth=1.0)
    spectrum_axis.plot(energy, detector, label="from raw ADC", linewidth=0.9)
    spectrum_axis.set_yscale("symlog", linthresh=max(1.0, incident.max() * 1.0e-7))
    spectrum_axis.set_xlim(float(energy.min()), float(energy.max()))
    spectrum_axis.set_xlabel("energy loss (eV)")
    spectrum_axis.set_ylabel("summed counts")
    spectrum_axis.set_title("Spectrum reconstructed from all scan frames")
    spectrum_axis.legend(fontsize=8)
    spectrum_axis.grid(alpha=0.2)

    map_grid = outer[1].subgridspec(len(elements), 3)
    column_titles = ("Expected edge signal", "After spectrometer", "From raw ADC")
    for row_index, element in enumerate(elements):
        expected, pre_map, detector_map = element_data[element]
        for column_index, image_data in enumerate((expected, pre_map, detector_map)):
            axis = figure.add_subplot(map_grid[row_index, column_index])
            if column_index < 2:
                low, high = np.quantile(image_data, (0.01, 0.99))
                if high <= low:
                    high = low + 1.0
                image_artist = axis.imshow(
                    image_data,
                    origin="lower",
                    cmap="magma",
                    vmin=low,
                    vmax=high,
                    interpolation="nearest",
                )
            else:
                center = float(np.median(image_data))
                limit = max(float(np.quantile(np.abs(image_data - center), 0.99)), 1.0)
                image_artist = axis.imshow(
                    image_data - center,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="nearest",
                )
            if row_index == 0:
                axis.set_title(column_titles[column_index])
            correlation = (
                1.0
                if column_index == 0
                else correlations[element]["pre_readout" if column_index == 1 else "detector"]
            )
            axis.text(
                0.02,
                0.98,
                f"{element}  r={correlation:.2f}",
                transform=axis.transAxes,
                va="top",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(
        "LMTO raw DOEELS simulation: information retained at each detector stage",
        fontsize=15,
    )
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "frame_index": frame_index,
        "depth_selection_index": depth_selection_index,
        "element_correlations": correlations,
        "output_png": str(output_path.resolve()),
    }
