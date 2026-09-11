from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .config import TransportConfig

K_BOLTZMANN_J_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class SensorGeometry:
    rows: int
    columns: int
    pixel_pitch_um: float
    sensor_thickness_um: float


def geometry_from_run_info(run_info: dict[str, np.ndarray]) -> SensorGeometry:
    return SensorGeometry(
        rows=int(run_info["rows"][0]),
        columns=int(run_info["pixel_columns"][0]),
        pixel_pitch_um=float(run_info["pixel_pitch_um"][0]),
        sensor_thickness_um=float(run_info["sensor_thickness_um"][0]),
    )


def diffusion_sigma_um(z_um: float, sensor_thickness_um: float, config: TransportConfig) -> float:
    """RMS transverse cloud width for collection at the +z sensor face."""
    collection_z_um = sensor_thickness_um / 2.0
    drift_distance_cm = max(0.0, collection_z_um - z_um) * 1.0e-4
    thickness_cm = sensor_thickness_um * 1.0e-4
    field_v_cm = config.bias_V / thickness_cm
    unsaturated_velocity = config.electron_mobility_cm2_Vs * field_v_cm
    drift_velocity = unsaturated_velocity / (
        1.0 + unsaturated_velocity / config.saturation_velocity_cm_s
    )
    drift_time_s = drift_distance_cm / drift_velocity
    diffusion_cm2_s = (
        config.electron_mobility_cm2_Vs
        * K_BOLTZMANN_J_K
        * config.temperature_K
        / ELEMENTARY_CHARGE_C
    )
    sigma_um = sqrt(2.0 * diffusion_cm2_s * drift_time_s) * 1.0e4
    return sqrt(sigma_um * sigma_um + config.sigma_floor_um**2)


def sample_pair_count(edep_eV: float, config: TransportConfig, rng: np.random.Generator) -> int:
    mean_pairs = edep_eV / config.pair_creation_energy_eV
    if mean_pairs <= 0.0:
        return 0
    sigma_pairs = sqrt(config.fano_factor * mean_pairs)
    return max(0, int(round(rng.normal(mean_pairs, sigma_pairs))))


def _axis_weights(
    coordinate_um: float,
    sigma_um: float,
    count: int,
    pitch_um: float,
    radius_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    extent_um = count * pitch_um
    center = int(np.floor((coordinate_um + extent_um / 2.0) / pitch_um))
    radius = max(1, int(np.ceil(radius_sigma * sigma_um / pitch_um)) + 1)
    indices = np.arange(max(0, center - radius), min(count, center + radius + 1))
    if len(indices) == 0:
        return indices, np.empty(0, dtype=np.float64)
    lower = -extent_um / 2.0 + indices * pitch_um
    upper = lower + pitch_um
    scale = sqrt(2.0) * sigma_um
    weights = np.array(
        [
            0.5 * (erf((hi - coordinate_um) / scale) - erf((lo - coordinate_um) / scale))
            for lo, hi in zip(lower, upper)
        ],
        dtype=np.float64,
    )
    return indices, weights


def spread_pairs(
    charge: np.ndarray,
    frame: int,
    x_um: float,
    y_um: float,
    sigma_um: float,
    pair_count: int,
    geometry: SensorGeometry,
    config: TransportConfig,
    rng: np.random.Generator,
) -> tuple[int, int]:
    columns, wx = _axis_weights(
        x_um,
        sigma_um,
        geometry.columns,
        geometry.pixel_pitch_um,
        config.cloud_radius_sigma,
    )
    rows, wy = _axis_weights(
        y_um,
        sigma_um,
        geometry.rows,
        geometry.pixel_pitch_um,
        config.cloud_radius_sigma,
    )
    if pair_count == 0 or len(rows) == 0 or len(columns) == 0:
        return 0, pair_count

    weights = np.outer(wy, wx).ravel()
    probability_sum = float(weights.sum())
    if probability_sum > 1.0:
        weights /= probability_sum
        probability_sum = 1.0
    probabilities = np.append(weights, max(0.0, 1.0 - probability_sum))
    probabilities /= probabilities.sum()
    samples = rng.multinomial(pair_count, probabilities)
    collected = samples[:-1].reshape(len(rows), len(columns))
    charge[frame][np.ix_(rows, columns)] += collected.astype(charge.dtype)
    return int(collected.sum()), int(samples[-1])


def transport_deposits(
    deposits: dict[str, np.ndarray],
    geometry: SensorGeometry,
    config: TransportConfig,
    primaries_per_frame: int,
    random_seed: int,
    max_events: int | None = None,
    frame_ids_by_event: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    event_ids = deposits["event_id"]
    if max_events is not None:
        selected = event_ids < max_events
    else:
        selected = np.ones(len(event_ids), dtype=bool)
    if not np.any(selected):
        raise ValueError("No energy deposits selected")

    selected_events = event_ids[selected]
    if frame_ids_by_event is None:
        frame_count = int(selected_events.max()) // primaries_per_frame + 1
    else:
        frame_ids_by_event = np.asarray(frame_ids_by_event, dtype=np.int64)
        if int(selected_events.max()) >= len(frame_ids_by_event):
            raise ValueError("Frame-ID lookup does not cover all selected events")
        relevant_events = len(frame_ids_by_event)
        if max_events is not None:
            relevant_events = min(relevant_events, max_events)
        relevant_frames = frame_ids_by_event[:relevant_events]
        if np.any(relevant_frames < 0):
            raise ValueError("Upstream frame IDs must be nonnegative")
        frame_count = int(relevant_frames.max()) + 1
    charge = np.zeros((frame_count, geometry.rows, geometry.columns), dtype=np.uint32)
    rng = np.random.default_rng(random_seed)
    generated_pairs = collected_pairs = lost_pairs = 0

    fields = ("event_id", "x_um", "y_um", "z_um", "edep_eV")
    rows = zip(*(deposits[field][selected] for field in fields))
    for event_id, x_um, y_um, z_um, edep_eV in rows:
        pair_count = sample_pair_count(float(edep_eV), config, rng)
        sigma_um = diffusion_sigma_um(float(z_um), geometry.sensor_thickness_um, config)
        frame_index = (
            int(event_id) // primaries_per_frame
            if frame_ids_by_event is None
            else int(frame_ids_by_event[int(event_id)])
        )
        collected, lost = spread_pairs(
            charge,
            frame_index,
            float(x_um),
            float(y_um),
            sigma_um,
            pair_count,
            geometry,
            config,
            rng,
        )
        generated_pairs += pair_count
        collected_pairs += collected
        lost_pairs += lost

    return charge, {
        "generated_pairs": generated_pairs,
        "collected_pairs": collected_pairs,
        "lost_pairs": lost_pairs,
        "frame_count": frame_count,
    }
