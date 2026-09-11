from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .io import read_readout_calibration, read_transport_output, write_digitized_hdf5
from .materialize import materialize_phase_space
from .phase_space import (
    generate_diagnostic_phase_space,
    read_phase_space_hdf5,
    read_phase_space_metadata,
    write_phase_space_hdf5,
)
from .readout import digitize, select_calibrated_maps
from .specimen import (
    available_specimen_kernels,
    load_specimen_kernel,
    run_specimen_kernel,
)
from .spectrometer import (
    transfer_phase_space,
    write_detector_entries_hdf5,
    write_geant4_source_csv,
)
from .transport import geometry_from_run_info, transport_deposits


def _frame_ids_by_event(primaries: dict[str, np.ndarray]) -> np.ndarray | None:
    if "frame_id" not in primaries:
        return None
    frame_ids = primaries["frame_id"].astype(np.int64)
    if np.any(frame_ids < 0):
        return None
    event_ids = primaries["event_id"].astype(np.int64)
    lookup = np.full(int(event_ids.max()) + 1, -1, dtype=np.int64)
    lookup[event_ids] = frame_ids
    if np.any(lookup < 0):
        raise ValueError("Primary event IDs must be contiguous for frame propagation")
    return lookup


def _inspect(base: Path) -> int:
    tables = read_transport_output(base)
    geometry = geometry_from_run_info(tables["run_info"])
    event_energy = tables["events"]["total_edep_eV"]
    print(
        f"sensor={geometry.rows}x{geometry.columns}, pitch={geometry.pixel_pitch_um:g} um, "
        f"thickness={geometry.sensor_thickness_um:g} um"
    )
    print(
        f"primaries={len(tables['primaries']['event_id'])}, "
        f"deposits={len(tables['deposits']['event_id'])}"
    )
    print(
        "deposited energy per primary [eV]: "
        f"median={np.median(event_energy):.1f}, "
        f"p10={np.quantile(event_energy, 0.1):.1f}, "
        f"p90={np.quantile(event_energy, 0.9):.1f}"
    )
    return 0


def _digitize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tables = read_transport_output(args.input)
    geometry = geometry_from_run_info(tables["run_info"])
    charge, summary = transport_deposits(
        tables["deposits"],
        geometry,
        config.transport,
        config.framing.primaries_per_frame,
        config.framing.random_seed,
        args.max_events,
        _frame_ids_by_event(tables["primaries"]),
    )
    calibration = None
    if config.readout.calibration_file is not None:
        calibration = read_readout_calibration(config.readout.calibration_file)
    pedestal_map, noise_map, signal_efficiency_map = select_calibrated_maps(
        calibration, config.readout
    )
    analog, raw = digitize(
        charge,
        config.readout,
        config.framing.random_seed + 1,
        pedestal_map,
        noise_map,
        signal_efficiency_map,
    )
    write_digitized_hdf5(
        args.output,
        charge,
        analog,
        raw,
        config,
        args.input,
        summary,
        args.keep_intermediate,
    )
    print(
        f"wrote {raw.shape[0]} frame(s) of {raw.shape[1]}x{raw.shape[2]} uint{raw.dtype.itemsize * 8} "
        f"to {args.output}"
    )
    print(
        f"charge pairs: generated={summary['generated_pairs']}, "
        f"collected={summary['collected_pairs']}, lost={summary['lost_pairs']}"
    )
    return 0


def _occupancy(config_path: Path) -> int:
    config = load_config(config_path)
    print(f"integration time: {config.experiment.integration_time_us:.6g} us")
    print(
        "expected incident electrons/frame: "
        f"{config.experiment.expected_primaries_per_frame:.6g}"
    )
    return 0


def _generate_phase_space(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    electrons = generate_diagnostic_phase_space(
        electron_count=args.electrons,
        frame_count=args.frames,
        beam_energy_eV=config.experiment.beam_energy_keV * 1.0e3,
        losses_eV=np.asarray(args.losses_eV),
        fractions=np.asarray(args.fractions),
        position_sigma_um=args.position_sigma_um,
        angular_sigma_mrad=args.angular_sigma_mrad,
        integration_time_ns=config.experiment.integration_time_us * 1.0e3,
        random_seed=args.random_seed,
    )
    write_phase_space_hdf5(
        args.output,
        electrons,
        metadata={
            "generator": "diagnostic_discrete_loss_mixture",
            "warning": "This source is not a specimen-scattering model",
            "weight_semantics": "individual_electrons",
            "losses_eV": args.losses_eV,
            "fractions": args.fractions,
        },
    )
    print(
        f"wrote {args.electrons} diagnostic electron(s) in {args.frames} frame(s) "
        f"to {args.output}"
    )
    return 0


def _inspect_phase_space(path: Path) -> int:
    electrons = read_phase_space_hdf5(path)
    directions = np.column_stack((electrons["dir_x"], electrons["dir_y"], electrons["dir_z"]))
    angles_mrad = 1.0e3 * np.hypot(directions[:, 0], directions[:, 1]) / np.abs(directions[:, 2])
    print(
        f"electrons={len(electrons['electron_id'])}, "
        f"frames={len(np.unique(electrons['frame_id']))}, "
        f"unit_weight={bool(np.allclose(electrons['weight'], 1.0))}"
    )
    print(
        "kinetic energy [eV]: "
        f"min={np.min(electrons['kinetic_energy_eV']):.6g}, "
        f"median={np.median(electrons['kinetic_energy_eV']):.6g}, "
        f"max={np.max(electrons['kinetic_energy_eV']):.6g}"
    )
    print(
        f"radial angle [mrad]: median={np.median(angles_mrad):.6g}, "
        f"max={np.max(angles_mrad):.6g}"
    )
    return 0


def _transfer_spectrometer(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    electrons = read_phase_space_hdf5(args.input)
    reference_energy_eV = config.spectrometer.reference_energy_eV
    if reference_energy_eV is None:
        reference_energy_eV = config.experiment.beam_energy_keV * 1.0e3
    entries, diagnostics, summary = transfer_phase_space(
        electrons, config.spectrometer, reference_energy_eV
    )
    write_detector_entries_hdf5(
        args.output,
        entries,
        diagnostics,
        summary,
        config.spectrometer,
        args.input,
    )
    if args.geant4_csv is not None:
        write_geant4_source_csv(args.geant4_csv, entries)
    print(
        f"accepted {summary['accepted_electrons']}/{summary['input_electrons']} "
        f"electron(s); wrote detector entries to {args.output}"
    )
    if args.geant4_csv is not None:
        print(f"wrote Geant4 individual-electron source to {args.geant4_csv}")
    rejected = summary["rejection_counts"]
    rejection_text = ", ".join(
        f"{name}={count}" for name, count in rejected.items() if name != "accepted" and count
    )
    if rejection_text:
        print(f"rejected: {rejection_text}")
    return 0


def _list_specimen_kernels() -> int:
    for name, source in available_specimen_kernels().items():
        print(f"{name}\t{source}")
    return 0


def _run_specimen(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    incident = read_phase_space_hdf5(args.input)
    kernel_name = args.kernel or config.specimen.kernel
    kernel = load_specimen_kernel(kernel_name)
    result = run_specimen_kernel(
        kernel,
        incident,
        reference_energy_eV=config.experiment.beam_energy_keV * 1.0e3,
        parameters=config.specimen.parameters,
        random_seed=config.specimen.random_seed,
    )
    write_phase_space_hdf5(
        args.output,
        result.electrons,
        metadata={
            "producer": "specimen_kernel",
            "kernel": kernel.name,
            "weight_semantics": result.weight_semantics,
            "kernel_parameters": config.specimen.parameters,
            "kernel_metadata": result.metadata,
            "source_phase_space": str(args.input),
        },
    )
    print(
        f"kernel={kernel.name}: wrote {len(result.electrons['electron_id'])} "
        f"weighted outgoing record(s) to {args.output}"
    )
    print(f"weight semantics: {result.weight_semantics}")
    return 0


def _materialize_phase_space(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    weighted = read_phase_space_hdf5(args.input)
    metadata = read_phase_space_metadata(args.input)
    requested_mode = args.mode or config.materialization.mode
    if requested_mode == "auto":
        semantics = metadata.get("weight_semantics")
        if semantics == "branch_probability":
            mode = "categorical"
        elif semantics == "expected_electrons":
            mode = "poisson"
        elif semantics == "individual_electrons":
            mode = "passthrough"
        else:
            raise ValueError(
                "Automatic materialization requires weight_semantics metadata; "
                "pass --mode explicitly"
            )
    else:
        mode = requested_mode
    expected_mode = {
        "branch_probability": "categorical",
        "expected_electrons": "poisson",
        "individual_electrons": "passthrough",
    }.get(metadata.get("weight_semantics"))
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
            f"Input weight semantics require {expected_mode} materialization, " f"not {mode}"
        )
    random_seed = (
        config.materialization.random_seed if args.random_seed is None else args.random_seed
    )
    max_electrons = (
        config.materialization.max_electrons if args.max_electrons is None else args.max_electrons
    )
    individual, summary = materialize_phase_space(
        weighted,
        mode=mode,
        random_seed=random_seed,
        max_electrons=max_electrons,
    )
    write_phase_space_hdf5(
        args.output,
        individual,
        metadata={
            "producer": "weighted_phase_space_materializer",
            "weight_semantics": "individual_electrons",
            "source_phase_space": str(args.input),
            "materialization_summary": summary,
        },
    )
    print(
        f"materialized {summary['output_electrons']} individual electron(s) "
        f"from {summary['input_records']} weighted record(s) using {mode} sampling"
    )
    print(f"wrote unit-weight phase space to {args.output}")
    return 0


def _calibrate_dark(args: argparse.Namespace) -> int:
    from .dark_calibration import calibrate_dark_stack

    summary = calibrate_dark_stack(
        args.input,
        args.output,
        dataset_name=args.dataset,
        calibration_frames=args.calibration_frames,
        validation_frames=args.validation_frames,
        storage_lsb=args.storage_lsb,
        max_noise_raw=args.max_noise_raw,
        batch_frames=args.batch_frames,
        row_stride=args.row_stride,
        column_stride=args.column_stride,
        known_defect_regions=args.known_defect,
    )
    print(f"wrote readout calibration to {args.output}")
    print(
        f"storage LSB={summary['storage_lsb_raw_units_per_adu']} raw units/code, "
        f"valid={100.0 * summary['valid_fraction']:.3f}%, "
        f"median noise={summary['read_noise_adu']['median']:.4g} codes"
    )
    print(
        f"held-out bias RMS={summary['validation_bias_rms_adu']:.4g} codes, "
        f"global common-mode RMS={summary['global_common_mode_rms_adu']:.4g} codes"
    )
    return 0


def _pixel_region(value: str) -> tuple[int, int, int, int]:
    try:
        row_text, column_text = value.split(",")
        row_start, row_stop = (int(item) for item in row_text.split(":"))
        column_start, column_stop = (int(item) for item in column_text.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "region must have form ROW_START:ROW_STOP,COL_START:COL_STOP"
        ) from error
    return row_start, row_stop, column_start, column_stop


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated numeric list") from error
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def _compare_response(args: argparse.Namespace) -> int:
    from .response_calibration import compare_response_libraries

    config = load_config(args.config)
    result = compare_response_libraries(
        args.legacy_100,
        args.legacy_300,
        args.new_100,
        args.new_300,
        args.output,
        config.transport,
        config.framing.random_seed,
    )
    ratios = result["median_pair_ratio_new_to_legacy"]
    print(f"wrote response comparison to {args.output}")
    print(
        f"new/legacy median charge ratio: 100 keV={ratios['100keV']:.4g}, "
        f"300 keV={ratios['300keV']:.4g}"
    )
    return 0


def _run_abtem_convergence(args: argparse.Namespace) -> int:
    from .abtem_convergence import run_abtem_convergence

    report = run_abtem_convergence(
        args.config,
        args.output_json,
        args.output_csv,
        args.output_plot,
        args.output_combined_plot,
    )
    summary = report["summary"]
    print(
        f"completed {summary['successful_cases']}/{summary['case_count']} " f"convergence case(s)"
    )
    print(f"wrote convergence report to {args.output_json}")
    if args.output_csv is not None:
        print(f"wrote convergence table to {args.output_csv}")
    if args.output_plot is not None:
        print(f"wrote convergence plot to {args.output_plot}")
    if args.output_combined_plot is not None:
        print(f"wrote combined-check plot to {args.output_combined_plot}")
    return 0 if summary["failed_cases"] == 0 else 1


def _simulate_1d_spectrum(args: argparse.Namespace) -> int:
    from .spectrum_1d import simulate_1d_spectrum

    summary = simulate_1d_spectrum(
        config_path=args.config,
        electron_count=args.electrons,
        frame_count=args.frames,
        output_hdf5=args.output_hdf5,
        output_plot=args.output_plot,
        energy_min_eV=args.energy_min_eV,
        energy_max_eV=args.energy_max_eV,
        energy_step_eV=args.energy_step_eV,
        output_phase_space=args.output_phase_space,
        output_detector_entries=args.output_detector_entries,
    )
    print(
        f"simulated {summary['incident_electrons']} incident electron(s) for "
        f"{summary['material']}"
    )
    print(
        f"spectrometer accepted {summary['accepted_electrons']} "
        f"({100.0 * summary['acceptance_fraction']:.3f}%)"
    )
    print(f"wrote 1D spectrum to {args.output_hdf5}")
    if args.output_plot is not None:
        print(f"wrote spectrum plot to {args.output_plot}")
    return 0


def _simulate_lmto_scan(args: argparse.Namespace) -> int:
    from .lmto_scan import simulate_lmto_scan

    summary = simulate_lmto_scan(
        config_path=args.config,
        output_hdf5=args.output_hdf5,
        output_montage=args.output_montage,
        output_html=args.output_html,
    )
    depth, rows, columns = summary["scan_shape"]
    print(
        f"simulated {depth} focal section(s) on a {rows}x{columns} LMTO scan; "
        f"{summary['energy_bins']} energy bin(s) per probe position"
    )
    print(
        f"field of view={summary['field_of_view_A'] / 10.0:.3f} nm, "
        f"model thickness={summary['sample_thickness_A'] / 10.0:.3f} nm"
    )
    print(f"wrote spectrum image and reconstructions to {args.output_hdf5}")
    if args.output_montage is not None:
        print(f"wrote reconstruction montage to {args.output_montage}")
    if args.output_html is not None:
        print(f"wrote interactive element/depth viewer to {args.output_html}")
    return 0


def _generate_lmto_abtem_spatial(args: argparse.Namespace) -> int:
    from .lmto_abtem_scan import generate_lmto_abtem_spatial_response

    summary = generate_lmto_abtem_spatial_response(
        config_path=args.config,
        output_hdf5=args.output_hdf5,
    )
    depth, rows, columns = summary["scan_shape"]
    print(
        f"calculated abTEM LMTO spatial response: {depth} through-focus "
        f"section(s), {rows}x{columns} probe positions"
    )
    print(
        "transition support: "
        + ", ".join(f"{name}={status}" for name, status in summary["transition_status"].items())
    )
    print(f"wrote reusable spatial response to {args.output_hdf5}")
    return 0


def _compare_lmto_abtem_spatial(args: argparse.Namespace) -> int:
    from .lmto_abtem_scan import compare_lmto_abtem_spatial_responses

    comparison = compare_lmto_abtem_spatial_responses(
        args.reference, args.confirmation, args.output_json
    )
    haadf = comparison["haadf"]
    print(
        "HAADF confirmation: "
        f"correlation={haadf['pearson_correlation']:.6f}, "
        f"relative RMS={haadf['relative_rms_difference']:.6g}"
    )
    for name, metrics in comparison["edges"].items():
        print(
            f"{name}: correlation={metrics['pearson_correlation']:.6f}, "
            f"relative RMS={metrics['relative_rms_difference']:.6g}"
        )
    print(f"wrote spatial comparison to {args.output_json}")
    return 0


def _tile_lmto_abtem_spatial(args: argparse.Namespace) -> int:
    from .lmto_abtem_scan import tile_lmto_abtem_spatial_response

    summary = tile_lmto_abtem_spatial_response(args.input_hdf5, args.output_hdf5, args.tile_factor)
    depth, rows, columns = summary["scan_shape"]
    print(
        f"tiled abTEM response {args.tile_factor}x{args.tile_factor}: "
        f"{depth} section(s), {rows}x{columns} probe positions"
    )
    print(
        f"field of view={summary['field_of_view_A'] / 10.0:.3f} nm; "
        "original probe sampling retained"
    )
    print(f"wrote periodic spatial response to {args.output_hdf5}")
    return 0


def _refine_lmto_abtem_haadf(args: argparse.Namespace) -> int:
    from .lmto_abtem_scan import refine_lmto_abtem_haadf

    summary = refine_lmto_abtem_haadf(
        args.config,
        args.input_hdf5,
        args.output_hdf5,
        args.potential_sampling_A,
    )
    print(
        f"recalculated {summary['scan_shape'][0]} HAADF section(s) at actual "
        f"sampling {summary['potential_sampling_A']} A"
    )
    print(f"wrote refined combined spatial response to {args.output_hdf5}")
    return 0


def _build_detector_response_kernel(args: argparse.Namespace) -> int:
    from .raw_doeels import build_monte_carlo_response_kernel

    summary = build_monte_carlo_response_kernel(
        args.config, args.geant4_base, args.output_hdf5, args.radius_pixels
    )
    print(
        f"built {2 * args.radius_pixels + 1}x{2 * args.radius_pixels + 1} "
        f"response kernel from {summary['event_count']} Geant4 electron(s)"
    )
    print(
        f"single-electron charge: mean={summary['mean_pairs']:.3f}, "
        f"median={summary['median_pairs']:.3f} e-h pairs"
    )
    print(f"wrote Monte Carlo response kernel to {args.output_hdf5}")
    return 0


def _simulate_raw_doeels(args: argparse.Namespace) -> int:
    from .raw_doeels import simulate_raw_doeels_scan

    summary = simulate_raw_doeels_scan(
        args.config,
        args.spectrum_image,
        args.output_hdf5,
        scan_stride=args.scan_stride,
        depth_indices=args.depth_index,
        integrations_per_position=args.integrations_per_position,
        electrons_per_integration=args.electrons_per_integration,
    )
    print(
        f"wrote {summary['frame_count']} raw DOEELS frame(s) of "
        f"{summary['detector_shape'][0]}x{summary['detector_shape'][1]} pixels"
    )
    print(
        f"incident electrons={summary['total_incident_electrons']}, "
        f"spectrometer-rejected={summary['spectrometer_rejected_electrons']}"
    )
    print(f"wrote raw scan and reconstructed spectra to {args.output_hdf5}")
    return 0


def _plot_raw_doeels(args: argparse.Namespace) -> int:
    from .raw_doeels import plot_raw_doeels_diagnostics

    summary = plot_raw_doeels_diagnostics(
        args.input_hdf5,
        args.output_png,
        depth_selection_index=args.depth_selection_index,
        frame_index=args.frame_index,
    )
    for element, metrics in summary["element_correlations"].items():
        print(
            f"{element}: pre-readout r={metrics['pre_readout']:.3f}, "
            f"raw-ADC r={metrics['detector']:.3f}"
        )
    print(f"wrote raw DOEELS diagnostics to {args.output_png}")
    return 0


def _simulate_exposure_sweep(args: argparse.Namespace) -> int:
    from .exposure_sweep import simulate_exposure_sweep

    summary = simulate_exposure_sweep(
        args.config,
        args.spectrum_image,
        args.output_hdf5,
        exposures=args.exposure,
        scan_stride=args.scan_stride,
        depth_indices=args.depth_index,
    )
    print(
        f"wrote {summary['scan_shape'][1]}x{summary['scan_shape'][2]} nested "
        f"exposure sweep with {len(summary['exposures_integrations_per_position'])} "
        "checkpoint(s)"
    )
    print(
        f"maximum dwell={summary['dwell_ms'][-1]:.3f} ms, "
        f"raw frames retained={summary['raw_frames_retained']}"
    )
    print(f"wrote compact exposure sweep to {args.output_hdf5}")
    return 0


def _plot_exposure_sweep(args: argparse.Namespace) -> int:
    from .exposure_sweep import plot_exposure_sweep

    summary = plot_exposure_sweep(
        args.input_hdf5,
        args.output_png,
        depth_selection_index=args.depth_selection_index,
    )
    for element, correlations in summary["element_correlations"].items():
        print(
            f"{element}: maximum-exposure counted r={correlations['counted'][-1]:.3f}, "
            f"integrating raw-ADC r={correlations['integrating_raw_adc'][-1]:.3f}"
        )
    print(f"wrote exposure-sweep diagnostics to {args.output_png}")
    return 0


def _build_exposure_slider_demo(args: argparse.Namespace) -> int:
    from .exposure_sweep import write_exposure_slider_demo

    elements = tuple(args.element or ("Mn", "O", "Ti"))
    summary = write_exposure_slider_demo(
        args.input_hdf5,
        args.output_html,
        args.output_png,
        elements=elements,
        exposure_index=args.exposure_index,
        depth_selection_index=args.depth_selection_index,
        display_sigma_pixels=args.display_sigma_pixels,
    )
    print(
        f"built {'/'.join(summary['channels'])} slider at "
        f"{summary['dwell_ms']:.3f} ms per position"
    )
    print(f"wrote interactive viewer to {args.output_html}")
    print(f"wrote static montage to {args.output_png}")
    return 0


def _calibrate_nio_sparse_detector(args: argparse.Namespace) -> int:
    from .nio_detector_calibration import calibrate_nio_sparse_detector

    summary = calibrate_nio_sparse_detector(
        args.spectrum_hdf5,
        args.dark_hdf5,
        args.readout_calibration,
        args.response_kernel,
        args.output_hdf5,
        args.output_png,
        dataset=args.dataset,
        spectrum_frame_start=args.spectrum_frame_start,
        spectrum_frame_count=args.spectrum_frame_count,
        dark_frame_start=args.dark_frame_start,
        dark_frame_count=args.dark_frame_count,
        row_start=args.row_start,
        row_stop=args.row_stop,
        column_start=args.column_start,
        storage_lsb=args.storage_lsb,
        threshold_sigma=args.threshold_sigma,
        xray_threshold_adu=args.xray_threshold_adu,
        patch_radius=args.patch_radius,
        random_seed=args.random_seed,
    )
    fit = summary["fit"]
    print(
        f"isolated events: spectrum={summary['spectrum_isolated_events']}, "
        f"dark={summary['dark_isolated_events']}"
    )
    print(
        f"effective gain={fit['gain_adu_per_pair']:.4f} ADU/e-h pair, "
        f"additional blur={fit['additional_blur_sigma_pixels']:.3f} pixel"
    )
    print(f"wrote NiO detector calibration to {args.output_hdf5}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eels-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="summarize Geant4 CSV output")
    inspect_parser.add_argument("input", type=Path, help="output base before _nt_*.csv")

    digitize_parser = subparsers.add_parser(
        "digitize", help="transport charge, pixelize, and simulate readout"
    )
    digitize_parser.add_argument("input", type=Path, help="output base before _nt_*.csv")
    digitize_parser.add_argument("--config", "-c", type=Path, required=True)
    digitize_parser.add_argument("--output", "-o", type=Path, required=True)
    digitize_parser.add_argument("--max-events", type=int)
    digitize_parser.add_argument("--keep-intermediate", action="store_true")

    occupancy_parser = subparsers.add_parser(
        "occupancy", help="calculate source timing and mean frame occupancy"
    )
    occupancy_parser.add_argument("--config", "-c", type=Path, required=True)

    phase_parser = subparsers.add_parser(
        "generate-phase-space",
        help="write a diagnostic individual-electron phase-space mixture",
    )
    phase_parser.add_argument("--config", "-c", type=Path, required=True)
    phase_parser.add_argument("--output", "-o", type=Path, required=True)
    phase_parser.add_argument("--electrons", type=int, default=1000)
    phase_parser.add_argument("--frames", type=int, default=1)
    phase_parser.add_argument("--losses-eV", type=_float_list, default=[0.0])
    phase_parser.add_argument("--fractions", type=_float_list, default=[1.0])
    phase_parser.add_argument("--position-sigma-um", type=float, default=0.0)
    phase_parser.add_argument("--angular-sigma-mrad", type=float, default=0.0)
    phase_parser.add_argument("--random-seed", type=int, default=13579)

    phase_inspect_parser = subparsers.add_parser(
        "inspect-phase-space", help="validate and summarize a phase-space HDF5 file"
    )
    phase_inspect_parser.add_argument("input", type=Path)

    transfer_parser = subparsers.add_parser(
        "transfer-spectrometer",
        help="map specimen-exit phase space onto the detector plane",
    )
    transfer_parser.add_argument("input", type=Path)
    transfer_parser.add_argument("--config", "-c", type=Path, required=True)
    transfer_parser.add_argument("--output", "-o", type=Path, required=True)
    transfer_parser.add_argument(
        "--geant4-csv",
        type=Path,
        help="also write an analog individual-electron source for Geant4",
    )

    subparsers.add_parser(
        "list-specimen-kernels", help="list built-in and installed specimen plugins"
    )

    specimen_parser = subparsers.add_parser(
        "run-specimen", help="run a specimen kernel on incident phase space"
    )
    specimen_parser.add_argument("input", type=Path)
    specimen_parser.add_argument("--config", "-c", type=Path, required=True)
    specimen_parser.add_argument("--output", "-o", type=Path, required=True)
    specimen_parser.add_argument(
        "--kernel",
        help="override [specimen].kernel with a registered name or module:object",
    )

    materialize_parser = subparsers.add_parser(
        "materialize-phase-space",
        help="sample weighted records into unit-weight individual electrons",
    )
    materialize_parser.add_argument("input", type=Path)
    materialize_parser.add_argument("--config", "-c", type=Path, required=True)
    materialize_parser.add_argument("--output", "-o", type=Path, required=True)
    materialize_parser.add_argument(
        "--mode", choices=("auto", "poisson", "categorical", "passthrough")
    )
    materialize_parser.add_argument("--random-seed", type=int)
    materialize_parser.add_argument("--max-electrons", type=int)

    dark_parser = subparsers.add_parser(
        "calibrate-dark", help="derive readout maps from a dark-frame HDF5 stack"
    )
    dark_parser.add_argument("input", type=Path)
    dark_parser.add_argument("--output", "-o", type=Path, required=True)
    dark_parser.add_argument("--dataset", default="frames")
    dark_parser.add_argument("--calibration-frames", type=int)
    dark_parser.add_argument("--validation-frames", type=int)
    dark_parser.add_argument("--storage-lsb", type=int)
    dark_parser.add_argument("--max-noise-raw", type=float, default=500.0)
    dark_parser.add_argument("--batch-frames", type=int, default=4)
    dark_parser.add_argument("--row-stride", type=int, default=8)
    dark_parser.add_argument("--column-stride", type=int, default=8)
    dark_parser.add_argument(
        "--known-defect",
        action="append",
        type=_pixel_region,
        default=[],
        metavar="ROWS,COLUMNS",
        help=("repeatable half-open pixel region, for example " "0:480,2272:2288"),
    )

    response_parser = subparsers.add_parser(
        "compare-response",
        help="compare legacy and new single-electron response libraries",
    )
    response_parser.add_argument("--legacy-100", type=Path, required=True)
    response_parser.add_argument("--legacy-300", type=Path, required=True)
    response_parser.add_argument("--new-100", type=Path, required=True)
    response_parser.add_argument("--new-300", type=Path, required=True)
    response_parser.add_argument("--config", "-c", type=Path, required=True)
    response_parser.add_argument("--output", "-o", type=Path, required=True)

    convergence_parser = subparsers.add_parser(
        "run-abtem-convergence",
        help="run the configured one-at-a-time abTEM convergence matrix",
    )
    convergence_parser.add_argument("--config", "-c", type=Path, required=True)
    convergence_parser.add_argument("--output-json", type=Path, required=True)
    convergence_parser.add_argument("--output-csv", type=Path)
    convergence_parser.add_argument("--output-plot", type=Path)
    convergence_parser.add_argument("--output-combined-plot", type=Path)

    spectrum_parser = subparsers.add_parser(
        "simulate-1d-spectrum",
        help="sample an individual-electron specimen kernel and histogram its spectrum",
    )
    spectrum_parser.add_argument("--config", "-c", type=Path, required=True)
    spectrum_parser.add_argument("--electrons", type=int, default=100000)
    spectrum_parser.add_argument("--frames", type=int, default=1)
    spectrum_parser.add_argument("--output-hdf5", type=Path, required=True)
    spectrum_parser.add_argument("--output-plot", type=Path)
    spectrum_parser.add_argument("--output-phase-space", type=Path)
    spectrum_parser.add_argument("--output-detector-entries", type=Path)
    spectrum_parser.add_argument("--energy-min-eV", type=float, default=-2.0)
    spectrum_parser.add_argument("--energy-max-eV", type=float, default=800.0)
    spectrum_parser.add_argument("--energy-step-eV", type=float, default=0.25)

    scan_parser = subparsers.add_parser(
        "simulate-lmto-scan",
        help="generate a synthetic LMTO HAADF and EELS spectrum image",
    )
    scan_parser.add_argument("--config", "-c", type=Path, required=True)
    scan_parser.add_argument("--output-hdf5", type=Path, required=True)
    scan_parser.add_argument("--output-montage", type=Path)
    scan_parser.add_argument("--output-html", type=Path)

    abtem_scan_parser = subparsers.add_parser(
        "generate-lmto-abtem-spatial",
        help="run abTEM LMTO HAADF and edge-specific raster scans",
    )
    abtem_scan_parser.add_argument("--config", "-c", type=Path, required=True)
    abtem_scan_parser.add_argument("--output-hdf5", type=Path, required=True)

    abtem_compare_parser = subparsers.add_parser(
        "compare-lmto-abtem-spatial",
        help="compare two cached LMTO abTEM spatial calculations",
    )
    abtem_compare_parser.add_argument("--reference", type=Path, required=True)
    abtem_compare_parser.add_argument("--confirmation", type=Path, required=True)
    abtem_compare_parser.add_argument("--output-json", type=Path, required=True)

    abtem_tile_parser = subparsers.add_parser(
        "tile-lmto-abtem-spatial",
        help="periodically expand a cached LMTO abTEM spatial response",
    )
    abtem_tile_parser.add_argument("--input-hdf5", type=Path, required=True)
    abtem_tile_parser.add_argument("--output-hdf5", type=Path, required=True)
    abtem_tile_parser.add_argument("--tile-factor", type=int, required=True)

    haadf_refine_parser = subparsers.add_parser(
        "refine-lmto-abtem-haadf",
        help="replace cached HAADF with a finer-grid abTEM calculation",
    )
    haadf_refine_parser.add_argument("--config", "-c", type=Path, required=True)
    haadf_refine_parser.add_argument("--input-hdf5", type=Path, required=True)
    haadf_refine_parser.add_argument("--output-hdf5", type=Path, required=True)
    haadf_refine_parser.add_argument("--potential-sampling-A", type=float, required=True)

    response_kernel_parser = subparsers.add_parser(
        "build-detector-response-kernel",
        help="compress Geant4 single-electron deposits into a reusable kernel",
    )
    response_kernel_parser.add_argument("--config", "-c", type=Path, required=True)
    response_kernel_parser.add_argument("--geant4-base", type=Path, required=True)
    response_kernel_parser.add_argument("--output-hdf5", type=Path, required=True)
    response_kernel_parser.add_argument("--radius-pixels", type=int, default=3)

    raw_doeels_parser = subparsers.add_parser(
        "simulate-raw-doeels",
        help="stream Monte Carlo-response raw DOEELS frames over a spectrum image",
    )
    raw_doeels_parser.add_argument("--config", "-c", type=Path, required=True)
    raw_doeels_parser.add_argument("--spectrum-image", type=Path, required=True)
    raw_doeels_parser.add_argument("--output-hdf5", type=Path, required=True)
    raw_doeels_parser.add_argument("--scan-stride", type=int)
    raw_doeels_parser.add_argument(
        "--depth-index",
        type=int,
        action="append",
        help="repeat to render selected focal depths; defaults to config",
    )
    raw_doeels_parser.add_argument("--integrations-per-position", type=int)
    raw_doeels_parser.add_argument("--electrons-per-integration", type=int)

    raw_plot_parser = subparsers.add_parser(
        "plot-raw-doeels",
        help="plot raw frames, reconstructed spectra, and stage-by-stage element maps",
    )
    raw_plot_parser.add_argument("--input-hdf5", type=Path, required=True)
    raw_plot_parser.add_argument("--output-png", type=Path, required=True)
    raw_plot_parser.add_argument("--depth-selection-index", type=int, default=0)
    raw_plot_parser.add_argument("--frame-index", type=int, default=0)

    exposure_parser = subparsers.add_parser(
        "simulate-raw-doeels-exposure-sweep",
        help="stream nested detector exposures into compact spectra and maps",
    )
    exposure_parser.add_argument("--config", "-c", type=Path, required=True)
    exposure_parser.add_argument("--spectrum-image", type=Path, required=True)
    exposure_parser.add_argument("--output-hdf5", type=Path, required=True)
    exposure_parser.add_argument(
        "--exposure",
        type=int,
        action="append",
        required=True,
        help="87 kHz integrations per position; repeat for nested checkpoints",
    )
    exposure_parser.add_argument("--scan-stride", type=int)
    exposure_parser.add_argument(
        "--depth-index",
        type=int,
        action="append",
        help="repeat to select focal depths; defaults to config",
    )

    exposure_plot_parser = subparsers.add_parser(
        "plot-raw-doeels-exposure-sweep",
        help="plot dwell-dependent correlations and reconstructed element maps",
    )
    exposure_plot_parser.add_argument("--input-hdf5", type=Path, required=True)
    exposure_plot_parser.add_argument("--output-png", type=Path, required=True)
    exposure_plot_parser.add_argument("--depth-selection-index", type=int, default=0)

    exposure_demo_parser = subparsers.add_parser(
        "build-raw-doeels-slider-demo",
        help="build a channel slider and montage from detector-reconstructed maps",
    )
    exposure_demo_parser.add_argument("--input-hdf5", type=Path, required=True)
    exposure_demo_parser.add_argument("--output-html", type=Path, required=True)
    exposure_demo_parser.add_argument("--output-png", type=Path, required=True)
    exposure_demo_parser.add_argument(
        "--element", action="append", help="repeat to choose element channels"
    )
    exposure_demo_parser.add_argument("--exposure-index", type=int, default=-1)
    exposure_demo_parser.add_argument("--depth-selection-index", type=int, default=0)
    exposure_demo_parser.add_argument("--display-sigma-pixels", type=float, default=0.6)

    nio_calibration_parser = subparsers.add_parser(
        "calibrate-nio-sparse-detector",
        help="fit detector gain and residual charge spreading from sparse NiO tail events",
    )
    nio_calibration_parser.add_argument("--spectrum-hdf5", type=Path, required=True)
    nio_calibration_parser.add_argument("--dark-hdf5", type=Path, required=True)
    nio_calibration_parser.add_argument("--readout-calibration", type=Path, required=True)
    nio_calibration_parser.add_argument("--response-kernel", type=Path, required=True)
    nio_calibration_parser.add_argument("--output-hdf5", type=Path, required=True)
    nio_calibration_parser.add_argument("--output-png", type=Path, required=True)
    nio_calibration_parser.add_argument("--dataset", default="frames")
    nio_calibration_parser.add_argument("--spectrum-frame-start", type=int, default=0)
    nio_calibration_parser.add_argument("--spectrum-frame-count", type=int)
    nio_calibration_parser.add_argument("--dark-frame-start", type=int, default=256)
    nio_calibration_parser.add_argument("--dark-frame-count", type=int)
    nio_calibration_parser.add_argument("--row-start", type=int, default=32)
    nio_calibration_parser.add_argument("--row-stop", type=int, default=928)
    nio_calibration_parser.add_argument("--column-start", type=int, default=1986)
    nio_calibration_parser.add_argument("--storage-lsb", type=float, default=64.0)
    nio_calibration_parser.add_argument("--threshold-sigma", type=float, default=8.0)
    nio_calibration_parser.add_argument("--xray-threshold-adu", type=float, default=823.0)
    nio_calibration_parser.add_argument("--patch-radius", type=int, default=3)
    nio_calibration_parser.add_argument("--random-seed", type=int, default=9173)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.input)
    if args.command == "digitize":
        return _digitize(args)
    if args.command == "occupancy":
        return _occupancy(args.config)
    if args.command == "generate-phase-space":
        return _generate_phase_space(args)
    if args.command == "inspect-phase-space":
        return _inspect_phase_space(args.input)
    if args.command == "transfer-spectrometer":
        return _transfer_spectrometer(args)
    if args.command == "list-specimen-kernels":
        return _list_specimen_kernels()
    if args.command == "run-specimen":
        return _run_specimen(args)
    if args.command == "materialize-phase-space":
        return _materialize_phase_space(args)
    if args.command == "calibrate-dark":
        return _calibrate_dark(args)
    if args.command == "run-abtem-convergence":
        return _run_abtem_convergence(args)
    if args.command == "simulate-1d-spectrum":
        return _simulate_1d_spectrum(args)
    if args.command == "simulate-lmto-scan":
        return _simulate_lmto_scan(args)
    if args.command == "generate-lmto-abtem-spatial":
        return _generate_lmto_abtem_spatial(args)
    if args.command == "compare-lmto-abtem-spatial":
        return _compare_lmto_abtem_spatial(args)
    if args.command == "tile-lmto-abtem-spatial":
        return _tile_lmto_abtem_spatial(args)
    if args.command == "refine-lmto-abtem-haadf":
        return _refine_lmto_abtem_haadf(args)
    if args.command == "build-detector-response-kernel":
        return _build_detector_response_kernel(args)
    if args.command == "simulate-raw-doeels":
        return _simulate_raw_doeels(args)
    if args.command == "plot-raw-doeels":
        return _plot_raw_doeels(args)
    if args.command == "simulate-raw-doeels-exposure-sweep":
        return _simulate_exposure_sweep(args)
    if args.command == "plot-raw-doeels-exposure-sweep":
        return _plot_exposure_sweep(args)
    if args.command == "build-raw-doeels-slider-demo":
        return _build_exposure_slider_demo(args)
    if args.command == "calibrate-nio-sparse-detector":
        return _calibrate_nio_sparse_detector(args)
    return _compare_response(args)
