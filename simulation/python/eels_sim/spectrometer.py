from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .config import SpectrometerConfig
from .phase_space import validate_phase_space

DETECTOR_ENTRY_SCHEMA = "eels-sim-detector-entry-v1"
REJECTION_CODES = {
    0: "accepted",
    1: "backward_or_transverse",
    2: "outside_collection_aperture",
    3: "outside_energy_window",
    4: "outside_detector_columns",
    5: "outside_detector_rows",
}


def _reject(reason: np.ndarray, condition: np.ndarray, code: int) -> None:
    reason[(reason == 0) & condition] = code


def transfer_phase_space(
    electrons: Mapping[str, np.ndarray],
    config: SpectrometerConfig,
    reference_energy_eV: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Apply a configurable first-order paraxial spectrometer transfer."""
    count = validate_phase_space(electrons)
    if config.dispersion_eV_per_column <= 0.0:
        raise ValueError("dispersion_eV_per_column must be positive")
    if config.detector_rows <= 0 or config.detector_columns <= 0:
        raise ValueError("Detector dimensions must be positive")
    if config.pixel_pitch_um <= 0.0:
        raise ValueError("pixel_pitch_um must be positive")
    if config.zlp_repeats < 1:
        raise ValueError("zlp_repeats must be at least one")
    if config.zlp_repeats > 1 and config.zlp_lane_width_columns <= 0:
        raise ValueError("Repeated ZLP layout requires a positive lane width")
    if (
        config.zlp_repeats > 1
        and config.zlp_repeats * config.zlp_lane_width_columns > config.detector_columns
    ):
        raise ValueError("Repeated ZLP lanes do not fit within the detector")
    weights = np.asarray(electrons["weight"], dtype=np.float64)
    if config.require_unit_weight and not np.allclose(weights, 1.0):
        raise ValueError(
            "The individual-electron transfer requires weight=1; materialize "
            "weighted specimen records before detector transport"
        )

    rng = np.random.default_rng(config.random_seed)
    dir_x = np.asarray(electrons["dir_x"], dtype=np.float64)
    dir_y = np.asarray(electrons["dir_y"], dtype=np.float64)
    dir_z = np.asarray(electrons["dir_z"], dtype=np.float64)
    forward = dir_z > 0.0
    safe_dir_z = np.where(forward, dir_z, np.nan)
    theta_x_mrad = 1.0e3 * dir_x / safe_dir_z
    theta_y_mrad = 1.0e3 * dir_y / safe_dir_z
    radial_angle_mrad = np.hypot(theta_x_mrad, theta_y_mrad)

    kinetic_energy = np.asarray(electrons["kinetic_energy_eV"], dtype=np.float64)
    energy_loss = reference_energy_eV - kinetic_energy
    mapped_loss = energy_loss.copy()
    if config.energy_blur_sigma_eV > 0.0:
        mapped_loss += rng.normal(0.0, config.energy_blur_sigma_eV, count)

    x_um = np.asarray(electrons["x_um"], dtype=np.float64)
    y_um = np.asarray(electrons["y_um"], dtype=np.float64)
    stitched_column = (
        config.zero_loss_stitched_column
        + mapped_loss / config.dispersion_eV_per_column
        + config.dispersion_quadratic_columns_per_eV2 * mapped_loss**2
        + (config.x_magnification * x_um + config.x_angle_to_position_um_per_mrad * theta_x_mrad)
        / config.pixel_pitch_um
    )
    if config.point_spread_sigma_x_um > 0.0:
        stitched_column += rng.normal(
            0.0,
            config.point_spread_sigma_x_um / config.pixel_pitch_um,
            count,
        )

    raw_column = stitched_column.copy()
    zlp_lane = np.full(count, -1, dtype=np.int16)
    if config.zlp_repeats > 1:
        zlp_boundary = config.zlp_lane_width_columns - 0.5
        in_zlp_read = stitched_column < zlp_boundary
        zlp_lane[in_zlp_read] = rng.integers(
            0, config.zlp_repeats, np.count_nonzero(in_zlp_read), dtype=np.int16
        )
        raw_column[in_zlp_read] += zlp_lane[in_zlp_read] * config.zlp_lane_width_columns
        raw_column[~in_zlp_read] += (config.zlp_repeats - 1) * config.zlp_lane_width_columns

    detector_y_um = (
        config.y_magnification * y_um + config.y_angle_to_position_um_per_mrad * theta_y_mrad
    )
    if config.point_spread_sigma_y_um > 0.0:
        detector_y_um += rng.normal(0.0, config.point_spread_sigma_y_um, count)
    raw_row = config.zero_y_row + detector_y_um / config.pixel_pitch_um

    output_theta_x_mrad = (
        config.x_position_to_angle_mrad_per_um * x_um
        + config.x_angle_magnification * theta_x_mrad
        + config.energy_to_output_angle_x_mrad_per_eV * energy_loss
    )
    output_theta_y_mrad = (
        config.y_position_to_angle_mrad_per_um * y_um
        + config.y_angle_magnification * theta_y_mrad
        + config.energy_to_output_angle_y_mrad_per_eV * energy_loss
    )
    output_directions = np.column_stack(
        (
            output_theta_x_mrad * 1.0e-3,
            output_theta_y_mrad * 1.0e-3,
            np.ones(count),
        )
    )
    output_directions /= np.linalg.norm(output_directions, axis=1)[:, None]

    reason = np.zeros(count, dtype=np.int8)
    _reject(reason, ~forward, 1)
    _reject(reason, radial_angle_mrad > config.collection_semiangle_mrad, 2)
    _reject(
        reason,
        (energy_loss < config.minimum_energy_loss_eV)
        | (energy_loss > config.maximum_energy_loss_eV),
        3,
    )
    _reject(
        reason,
        (stitched_column < -0.5)
        | (raw_column < -0.5)
        | (raw_column >= config.detector_columns - 0.5),
        4,
    )
    _reject(
        reason,
        (raw_row < -0.5) | (raw_row >= config.detector_rows - 0.5),
        5,
    )
    accepted = reason == 0

    detector_x_um = (raw_column + 0.5 - config.detector_columns / 2.0) * config.pixel_pitch_um
    detector_y_um = (raw_row + 0.5 - config.detector_rows / 2.0) * config.pixel_pitch_um
    source_indices = np.arange(count, dtype=np.uint64)
    entries = {
        "event_id": np.arange(np.count_nonzero(accepted), dtype=np.uint64),
        "frame_id": np.asarray(electrons["frame_id"], dtype=np.uint64)[accepted],
        "electron_id": np.asarray(electrons["electron_id"], dtype=np.uint64)[accepted],
        "source_index": source_indices[accepted],
        "x_um": detector_x_um[accepted],
        "y_um": detector_y_um[accepted],
        "z_um": np.full(np.count_nonzero(accepted), config.detector_source_z_um),
        "dir_x": output_directions[accepted, 0],
        "dir_y": output_directions[accepted, 1],
        "dir_z": output_directions[accepted, 2],
        "kinetic_energy_eV": kinetic_energy[accepted],
        "time_ns": np.asarray(electrons["time_ns"], dtype=np.float64)[accepted],
        "weight": weights[accepted],
        "loss_channel": np.asarray(electrons["loss_channel"], dtype=np.int32)[accepted],
        "energy_loss_eV": energy_loss[accepted],
        "mapped_energy_loss_eV": mapped_loss[accepted],
        "stitched_column": stitched_column[accepted],
        "raw_column": raw_column[accepted],
        "raw_row": raw_row[accepted],
        "zlp_lane": zlp_lane[accepted],
    }
    for lineage_field in (
        "parent_electron_id",
        "branch_id",
        "source_record_id",
        "spectral_component_id",
        "plural_order",
    ):
        if lineage_field in electrons:
            entries[lineage_field] = np.asarray(electrons[lineage_field])[accepted]
    diagnostics = {
        "frame_id": np.asarray(electrons["frame_id"], dtype=np.uint64),
        "electron_id": np.asarray(electrons["electron_id"], dtype=np.uint64),
        "rejection_code": reason,
    }
    rejection_counts = {
        REJECTION_CODES[code]: int(np.count_nonzero(reason == code)) for code in REJECTION_CODES
    }
    summary: dict[str, object] = {
        "input_electrons": count,
        "accepted_electrons": int(np.count_nonzero(accepted)),
        "acceptance_fraction": float(np.mean(accepted)),
        "rejection_counts": rejection_counts,
        "reference_energy_eV": float(reference_energy_eV),
    }
    return entries, diagnostics, summary


def write_detector_entries_hdf5(
    path: str | Path,
    entries: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    summary: Mapping[str, object],
    config: SpectrometerConfig,
    source_path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = DETECTOR_ENTRY_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["source_phase_space"] = str(Path(source_path))
        h5.attrs["spectrometer_config_json"] = json.dumps(asdict(config), sort_keys=True)
        h5.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5.attrs["rejection_code_json"] = json.dumps(REJECTION_CODES, sort_keys=True)
        electron_group = h5.create_group("electrons")
        electron_group.attrs["position_units"] = "um"
        electron_group.attrs["energy_units"] = "eV"
        electron_group.attrs["time_units"] = "ns"
        electron_group.attrs["pixel_coordinate_convention"] = "integer values are pixel centers"
        for name, values in entries.items():
            electron_group.create_dataset(name, data=values, compression="gzip", shuffle=True)
        diagnostic_group = h5.create_group("diagnostics")
        for name, values in diagnostics.items():
            diagnostic_group.create_dataset(name, data=values, compression="gzip", shuffle=True)


def write_geant4_source_csv(path: str | Path, entries: Mapping[str, np.ndarray]) -> None:
    weights = np.asarray(entries["weight"])
    if not np.allclose(weights, 1.0):
        raise ValueError("Geant4 source CSV requires individual electrons with weight=1")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "event_id",
        "frame_id",
        "electron_id",
        "x_um",
        "y_um",
        "z_um",
        "dir_x",
        "dir_y",
        "dir_z",
        "kinetic_energy_eV",
        "time_ns",
        "weight",
        "loss_channel",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(len(entries["event_id"])):
            writer.writerow([entries[name][index] for name in fields])
