from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class ExperimentConfig:
    beam_energy_keV: float
    frame_rate_hz: float
    beam_current_pA: float

    @property
    def integration_time_us(self) -> float:
        return 1.0e6 / self.frame_rate_hz

    @property
    def expected_primaries_per_frame(self) -> float:
        current_a = self.beam_current_pA * 1.0e-12
        return current_a / (ELEMENTARY_CHARGE_C * self.frame_rate_hz)


@dataclass(frozen=True)
class TransportConfig:
    pair_creation_energy_eV: float = 3.64
    fano_factor: float = 0.115
    temperature_K: float = 300.0
    bias_V: float = 20.0
    electron_mobility_cm2_Vs: float = 1350.0
    saturation_velocity_cm_s: float = 1.0e7
    sigma_floor_um: float = 0.05
    cloud_radius_sigma: float = 6.0


@dataclass(frozen=True)
class ReadoutConfig:
    gain_adu_per_electron: float = 0.0036
    pedestal_adu: float = 100.0
    read_noise_adu: float = 1.0
    common_mode_noise_adu: float = 0.0
    row_common_mode_noise_adu: float = 0.0
    column_common_mode_noise_adu: float = 0.0
    adc_bits: int = 16
    calibration_file: str | None = None
    use_calibrated_pedestal: bool = True
    use_calibrated_noise: bool = True
    use_calibrated_correlated_noise: bool = True
    simulate_known_defects: bool = False


@dataclass(frozen=True)
class SpectrometerConfig:
    reference_energy_eV: float | None = None
    dispersion_eV_per_column: float = 1.0
    dispersion_quadratic_columns_per_eV2: float = 0.0
    zero_loss_stitched_column: float = 0.0
    detector_rows: int = 960
    detector_columns: int = 3840
    pixel_pitch_um: float = 10.0
    detector_source_z_um: float = -5.0
    zero_y_row: float = 479.5
    x_magnification: float = 1.0
    y_magnification: float = 1.0
    x_angle_to_position_um_per_mrad: float = 0.0
    y_angle_to_position_um_per_mrad: float = 0.0
    x_position_to_angle_mrad_per_um: float = 0.0
    y_position_to_angle_mrad_per_um: float = 0.0
    x_angle_magnification: float = 1.0
    y_angle_magnification: float = 1.0
    energy_to_output_angle_x_mrad_per_eV: float = 0.0
    energy_to_output_angle_y_mrad_per_eV: float = 0.0
    collection_semiangle_mrad: float = 50.0
    minimum_energy_loss_eV: float = -10.0
    maximum_energy_loss_eV: float = 1000.0
    energy_blur_sigma_eV: float = 0.0
    point_spread_sigma_x_um: float = 0.0
    point_spread_sigma_y_um: float = 0.0
    zlp_repeats: int = 1
    zlp_lane_width_columns: int = 0
    require_unit_weight: bool = True
    random_seed: int = 24680


@dataclass(frozen=True)
class SpecimenConfig:
    kernel: str = "identity"
    random_seed: int = 97531
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializationConfig:
    mode: str = "auto"
    random_seed: int = 86420
    max_electrons: int = 10_000_000


@dataclass(frozen=True)
class FramingConfig:
    primaries_per_frame: int = 10
    random_seed: int = 12345


@dataclass(frozen=True)
class ScanConfig:
    model: str = "synthetic_disordered_rocksalt"
    spatial_provider: str = "analytic"
    spatial_response_file: str | None = None
    scan_pixels: int = 64
    lateral_cells: int = 8
    depth_atomic_planes: int = 32
    depth_sections: int = 8
    lattice_constant_A: float = 4.15
    probe_sigma_A: float = 0.55
    axial_sigma_A: float = 8.0
    electrons_per_probe: int = 200_000
    energy_min_eV: float = -2.0
    energy_max_eV: float = 790.0
    energy_step_eV: float = 1.0
    haadf_peak_counts: float = 5_000.0
    reconstruction_sigma_pixels: float = 0.6
    random_seed: int = 314159
    element_edges: dict[str, list[str]] = field(default_factory=dict)
    abtem: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RawDOEELSConfig:
    response_kernel_file: str | None = None
    electrons_per_integration: int = 2152
    integrations_per_position: int = 1
    depth_indices: list[int] = field(default_factory=lambda: [0])
    scan_stride: int = 1
    detector_row_sigma_pixels: float = 187.0
    energy_mapping_subsamples: int = 16
    exact_response_max_electrons: int = 100_000
    response_blur_sigma_pixels: float = 0.0
    counting_first_raw_column: int = 0
    counting_min_energy_eV: float = 0.0
    counting_detection_efficiency: float = 1.0
    counting_dark_events_per_frame: float = 0.0
    counting_dark_first_raw_column: int = 0
    compression_level: int = 1
    keep_hit_counts: bool = False
    random_seed: int = 424242


@dataclass(frozen=True)
class SimulationConfig:
    experiment: ExperimentConfig
    transport: TransportConfig
    readout: ReadoutConfig
    spectrometer: SpectrometerConfig
    specimen: SpecimenConfig
    materialization: MaterializationConfig
    framing: FramingConfig
    scan: ScanConfig
    raw_doeels: RawDOEELSConfig
    source_text: str


def load_config(path: str | Path) -> SimulationConfig:
    config_path = Path(path)
    source_text = config_path.read_text(encoding="utf-8")
    values = tomllib.loads(source_text)
    readout_values = dict(values.get("readout", {}))
    calibration_file = readout_values.get("calibration_file")
    if calibration_file is not None:
        calibration_path = Path(calibration_file)
        if not calibration_path.is_absolute():
            calibration_path = (config_path.parent / calibration_path).resolve()
        readout_values["calibration_file"] = str(calibration_path)
    specimen_values = dict(values.get("specimen", {}))
    specimen_parameters = dict(specimen_values.get("parameters", {}))
    library_file = specimen_parameters.get("library_file")
    if library_file is not None:
        library_path = Path(str(library_file))
        if not library_path.is_absolute():
            library_path = (config_path.parent / library_path).resolve()
        specimen_parameters["library_file"] = str(library_path)
    specimen_values["parameters"] = specimen_parameters
    scan_values = dict(values.get("scan", {}))
    spatial_response_file = scan_values.get("spatial_response_file")
    if spatial_response_file is not None:
        response_path = Path(str(spatial_response_file))
        if not response_path.is_absolute():
            response_path = (config_path.parent / response_path).resolve()
        scan_values["spatial_response_file"] = str(response_path)
    raw_doeels_values = dict(values.get("raw_doeels", {}))
    response_kernel_file = raw_doeels_values.get("response_kernel_file")
    if response_kernel_file is not None:
        kernel_path = Path(str(response_kernel_file))
        if not kernel_path.is_absolute():
            kernel_path = (config_path.parent / kernel_path).resolve()
        raw_doeels_values["response_kernel_file"] = str(kernel_path)
    return SimulationConfig(
        experiment=ExperimentConfig(**values["experiment"]),
        transport=TransportConfig(**values.get("transport", {})),
        readout=ReadoutConfig(**readout_values),
        spectrometer=SpectrometerConfig(**values.get("spectrometer", {})),
        specimen=SpecimenConfig(**specimen_values),
        materialization=MaterializationConfig(**values.get("materialization", {})),
        framing=FramingConfig(**values.get("framing", {})),
        scan=ScanConfig(**scan_values),
        raw_doeels=RawDOEELSConfig(**raw_doeels_values),
        source_text=source_text,
    )
