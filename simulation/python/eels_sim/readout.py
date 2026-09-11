from __future__ import annotations

import numpy as np

from .config import ReadoutConfig


def select_calibrated_maps(
    calibration: dict[str, np.ndarray] | None,
    config: ReadoutConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Select independent calibration effects for one digitization run."""
    if calibration is None:
        return None, None, None
    map_prefix = "" if config.simulate_known_defects else "healthy_"
    pedestal = calibration[f"{map_prefix}pedestal_adu"] if config.use_calibrated_pedestal else None
    noise = calibration[f"{map_prefix}read_noise_adu"] if config.use_calibrated_noise else None
    signal_efficiency = calibration["signal_efficiency"] if config.simulate_known_defects else None
    return pedestal, noise, signal_efficiency


def digitize(
    charge_electrons: np.ndarray,
    config: ReadoutConfig,
    random_seed: int,
    pedestal_map: np.ndarray | None = None,
    noise_map: np.ndarray | None = None,
    signal_efficiency_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    spatial_shape = charge_electrons.shape[1:]
    if pedestal_map is not None and pedestal_map.shape != spatial_shape:
        raise ValueError(f"Pedestal map {pedestal_map.shape} does not match frames {spatial_shape}")
    if noise_map is not None and noise_map.shape != spatial_shape:
        raise ValueError(f"Noise map {noise_map.shape} does not match frames {spatial_shape}")
    if signal_efficiency_map is not None and signal_efficiency_map.shape != spatial_shape:
        raise ValueError(
            f"Signal-efficiency map {signal_efficiency_map.shape} does not match "
            f"frames {spatial_shape}"
        )
    pedestal = config.pedestal_adu if pedestal_map is None else pedestal_map[None, :, :]
    signal = charge_electrons.astype(np.float64) * config.gain_adu_per_electron
    if signal_efficiency_map is not None:
        signal *= signal_efficiency_map[None, :, :]
    analog = pedestal + signal
    use_correlated = config.use_calibrated_correlated_noise
    correlated_variance = 0.0
    if use_correlated:
        correlated_variance = (
            config.common_mode_noise_adu**2
            + config.row_common_mode_noise_adu**2
            + config.column_common_mode_noise_adu**2
        )
    if noise_map is not None:
        independent_noise = np.sqrt(
            np.maximum(noise_map.astype(np.float64) ** 2 - correlated_variance, 0.0)
        )
        analog += rng.normal(0.0, 1.0, size=analog.shape) * independent_noise[None, :, :]
    elif config.read_noise_adu > 0.0:
        analog += rng.normal(0.0, config.read_noise_adu, size=analog.shape)
    if use_correlated and config.common_mode_noise_adu > 0.0:
        common = rng.normal(0.0, config.common_mode_noise_adu, size=(analog.shape[0], 1, 1))
        analog += common
    if use_correlated and config.row_common_mode_noise_adu > 0.0:
        row_common = rng.normal(
            0.0,
            config.row_common_mode_noise_adu,
            size=(analog.shape[0], analog.shape[1], 1),
        )
        analog += row_common
    if use_correlated and config.column_common_mode_noise_adu > 0.0:
        column_common = rng.normal(
            0.0,
            config.column_common_mode_noise_adu,
            size=(analog.shape[0], 1, analog.shape[2]),
        )
        analog += column_common
    maximum = (1 << config.adc_bits) - 1
    raw = np.rint(np.clip(analog, 0.0, maximum)).astype(
        np.uint16 if config.adc_bits <= 16 else np.uint32
    )
    return analog.astype(np.float32), raw
