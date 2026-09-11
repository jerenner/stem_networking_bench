from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .config import load_config
from .phase_space import generate_diagnostic_phase_space, write_phase_space_hdf5
from .specimen import load_specimen_kernel, run_specimen_kernel
from .spectrometer import transfer_phase_space, write_detector_entries_hdf5

SPECTRUM_1D_SCHEMA = "eels-sim-spectrum-1d-v1"


def simulate_1d_spectrum(
    config_path: str | Path,
    electron_count: int,
    frame_count: int,
    output_hdf5: str | Path,
    output_plot: str | Path | None,
    energy_min_eV: float,
    energy_max_eV: float,
    energy_step_eV: float,
    output_phase_space: str | Path | None = None,
    output_detector_entries: str | Path | None = None,
) -> dict[str, object]:
    if electron_count <= 0 or frame_count <= 0:
        raise ValueError("electron_count and frame_count must be positive")
    if energy_step_eV <= 0.0 or energy_max_eV <= energy_min_eV:
        raise ValueError("invalid spectrum energy range or step")

    config = load_config(config_path)
    reference_energy_eV = config.spectrometer.reference_energy_eV
    if reference_energy_eV is None:
        reference_energy_eV = config.experiment.beam_energy_keV * 1.0e3
    incident = generate_diagnostic_phase_space(
        electron_count=electron_count,
        frame_count=frame_count,
        beam_energy_eV=reference_energy_eV,
        losses_eV=np.array([0.0]),
        fractions=np.array([1.0]),
        position_sigma_um=0.0,
        angular_sigma_mrad=0.0,
        integration_time_ns=config.experiment.integration_time_us * 1.0e3,
        random_seed=config.specimen.random_seed - 1,
    )
    kernel = load_specimen_kernel(config.specimen.kernel)
    result = run_specimen_kernel(
        kernel,
        incident,
        reference_energy_eV=reference_energy_eV,
        parameters=config.specimen.parameters,
        random_seed=config.specimen.random_seed,
    )
    if result.weight_semantics != "individual_electrons":
        raise ValueError(
            "simulate-1d-spectrum requires a kernel that directly emits " "individual electrons"
        )
    if output_phase_space is not None:
        write_phase_space_hdf5(
            output_phase_space,
            result.electrons,
            metadata={
                "producer": "simulate_1d_spectrum",
                "kernel": kernel.name,
                "weight_semantics": result.weight_semantics,
                "kernel_metadata": result.metadata,
            },
        )

    entries, diagnostics, transfer_summary = transfer_phase_space(
        result.electrons, config.spectrometer, reference_energy_eV
    )
    if output_detector_entries is not None:
        write_detector_entries_hdf5(
            output_detector_entries,
            entries,
            diagnostics,
            transfer_summary,
            config.spectrometer,
            output_phase_space or "in-memory specimen phase space",
        )

    edges_eV = np.arange(
        energy_min_eV,
        energy_max_eV + energy_step_eV,
        energy_step_eV,
        dtype=np.float64,
    )
    if edges_eV[-1] < energy_max_eV:
        edges_eV = np.append(edges_eV, energy_max_eV)
    centers_eV = 0.5 * (edges_eV[:-1] + edges_eV[1:])
    truth_loss_eV = reference_energy_eV - np.asarray(result.electrons["kinetic_energy_eV"])
    measured_loss_eV = np.asarray(entries["mapped_energy_loss_eV"])
    truth_counts = np.histogram(truth_loss_eV, edges_eV)[0]
    accepted_counts = np.histogram(measured_loss_eV, edges_eV)[0]

    component_labels = {
        int(key): str(value)
        for key, value in result.metadata.get("component_labels", {"0": "all"}).items()
    }
    accepted_component_id = np.asarray(
        entries.get(
            "spectral_component_id",
            np.zeros(len(measured_loss_eV), dtype=np.uint16),
        )
    )
    component_counts = {
        component_id: np.histogram(
            measured_loss_eV[accepted_component_id == component_id], edges_eV
        )[0]
        for component_id in sorted(component_labels)
    }

    summary: dict[str, object] = {
        "material": str(result.metadata.get("material", "unspecified")),
        "kernel": kernel.name,
        "incident_electrons": electron_count,
        "accepted_electrons": int(transfer_summary["accepted_electrons"]),
        "acceptance_fraction": float(transfer_summary["acceptance_fraction"]),
        "energy_range_eV": [energy_min_eV, energy_max_eV],
        "energy_step_eV": energy_step_eV,
        "component_labels": component_labels,
    }
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = SPECTRUM_1D_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["config_file"] = str(Path(config_path).resolve())
        h5.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5.attrs["kernel_metadata_json"] = json.dumps(result.metadata, sort_keys=True)
        spectrum = h5.create_group("spectrum")
        spectrum.attrs["energy_units"] = "eV"
        spectrum.attrs["intensity_units"] = "electron counts per bin"
        spectrum.create_dataset("energy_edges_eV", data=edges_eV)
        spectrum.create_dataset("energy_centers_eV", data=centers_eV)
        spectrum.create_dataset("truth_counts", data=truth_counts)
        spectrum.create_dataset("accepted_counts", data=accepted_counts)
        components = spectrum.create_group("accepted_components")
        for component_id, counts in component_counts.items():
            dataset = components.create_dataset(str(component_id), data=counts)
            dataset.attrs["label"] = component_labels[component_id]
        events = h5.create_group("accepted_events")
        events.create_dataset(
            "energy_loss_eV", data=measured_loss_eV, compression="gzip", shuffle=True
        )
        events.create_dataset(
            "spectral_component_id",
            data=accepted_component_id,
            compression="gzip",
            shuffle=True,
        )
        if "plural_order" in entries:
            events.create_dataset(
                "plural_order",
                data=entries["plural_order"],
                compression="gzip",
                shuffle=True,
            )
        events.create_dataset(
            "raw_column", data=entries["raw_column"], compression="gzip", shuffle=True
        )
        events.create_dataset("raw_row", data=entries["raw_row"], compression="gzip", shuffle=True)

    if output_plot is not None:
        plot_1d_spectrum(
            output_plot,
            centers_eV,
            accepted_counts,
            component_counts,
            component_labels,
            summary["material"],
            energy_max_eV,
        )
    return summary


def plot_1d_spectrum(
    output_path: str | Path,
    energy_eV: np.ndarray,
    total_counts: np.ndarray,
    component_counts: dict[int, np.ndarray],
    component_labels: dict[int, str],
    material: str,
    energy_max_eV: float,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for axis in axes:
        axis.step(
            energy_eV,
            total_counts,
            where="mid",
            color="black",
            lw=1.1,
            label="total",
        )
        for color_index, component_id in enumerate(sorted(component_counts)):
            if component_id == 0 or not np.any(component_counts[component_id]):
                continue
            axis.step(
                energy_eV,
                component_counts[component_id],
                where="mid",
                lw=0.9,
                color=colors(color_index % 10),
                label=component_labels[component_id],
            )
        axis.set_ylabel("electrons / bin")
        axis.grid(alpha=0.2)
    axes[0].set_xlim(0.0, energy_max_eV)
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.8)
    axes[0].set_title(f"Simulated energy-resolved EELS — {material}")
    axes[0].legend(ncols=4, fontsize=8)
    axes[1].set_xlim(min(3.0, energy_max_eV * 0.05), min(80.0, energy_max_eV))
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[1].set_xlabel("energy loss (eV)")
    axes[1].set_title("Low-loss and shallow-core region (ZLP excluded)")
    figure.savefig(path, dpi=180)
    plt.close(figure)
