from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .phase_space import scatter_directions, validate_phase_space
from .specimen import SpecimenContext, SpecimenResult
from .spectral_library import read_spectral_library


def _float_parameter(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = float(parameters.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


class EnergyResolvedSpecimenKernel:
    """Sample a tabulated low/core-loss library into individual electrons.

    Core components are mutually exclusive per incident electron. Low-loss events
    are an independent Poisson process, so a core event can be accompanied by
    zero, one, or several low-loss excitations (plural scattering).
    """

    name = "energy-resolved"

    def simulate(
        self,
        incident_electrons: Mapping[str, np.ndarray],
        context: SpecimenContext,
    ) -> SpecimenResult:
        count = validate_phase_space(incident_electrons)
        if not np.allclose(incident_electrons["weight"], 1.0):
            raise ValueError("The energy-resolved kernel requires unit-weight input")

        library_value = context.parameters.get("library_file")
        if library_value is None:
            raise ValueError("energy-resolved kernel requires parameters.library_file")
        library_path = Path(str(library_value))
        library = read_spectral_library(library_path)
        energy_tolerance_eV = _float_parameter(context.parameters, "beam_energy_tolerance_eV", 1.0)
        if abs(library.beam_energy_eV - context.reference_energy_eV) > energy_tolerance_eV:
            raise ValueError(
                f"Spectral library beam energy {library.beam_energy_eV:g} eV does "
                f"not match simulation reference energy {context.reference_energy_eV:g} eV"
            )

        low_loss_scale = _float_parameter(context.parameters, "low_loss_scale", 1.0)
        core_probability_scale = _float_parameter(context.parameters, "core_probability_scale", 1.0)
        angular_scale = _float_parameter(context.parameters, "angular_scale", 1.0)
        maximum_plural_order = int(context.parameters.get("maximum_plural_order", 20))
        if maximum_plural_order < 1 or maximum_plural_order > np.iinfo(np.uint16).max:
            raise ValueError("maximum_plural_order must be in [1, 65535]")

        core_probabilities = np.asarray(
            [
                component.integrated_probability * core_probability_scale
                for component in library.core_loss
            ],
            dtype=np.float64,
        )
        total_core_probability = float(core_probabilities.sum())
        if total_core_probability > 1.0 + 1.0e-12:
            raise ValueError(
                "Scaled core-loss probabilities sum above one; reduce " "core_probability_scale"
            )
        outcome_probability = np.concatenate(
            ([max(0.0, 1.0 - total_core_probability)], core_probabilities)
        )
        outcome_probability /= outcome_probability.sum()
        # 0 is no core event; 1..N identifies a library core component.
        core_choice = context.rng.choice(
            len(outcome_probability), size=count, p=outcome_probability
        )

        low_count = np.zeros(count, dtype=np.int64)
        low_energy_eV = np.zeros(count, dtype=np.float64)
        clipped_plural_events = 0
        if library.low_loss is not None:
            mean_events = library.low_loss.mean_events * low_loss_scale
            sampled_count = context.rng.poisson(mean_events, count)
            low_count = np.minimum(sampled_count, maximum_plural_order)
            clipped_plural_events = int(np.count_nonzero(sampled_count > low_count))
            event_count = int(low_count.sum())
            if event_count:
                sampled_energy = library.low_loss.distribution.sample(event_count, context.rng)
                owners = np.repeat(np.arange(count, dtype=np.int64), low_count)
                low_energy_eV = np.bincount(owners, weights=sampled_energy, minlength=count)

        core_energy_eV = np.zeros(count, dtype=np.float64)
        core_sigma_mrad = np.zeros(count, dtype=np.float64)
        component_id = np.zeros(count, dtype=np.uint16)
        component_counts: dict[str, int] = {}
        for core_index, component in enumerate(library.core_loss, start=1):
            selected = core_choice == core_index
            selected_count = int(np.count_nonzero(selected))
            component_counts[component.name] = selected_count
            if selected_count:
                core_energy_eV[selected] = component.distribution.sample(
                    selected_count, context.rng
                )
                core_sigma_mrad[selected] = component.angular_sigma_mrad
                # 0=ZLP, 1=low loss, 2..=core components.
                component_id[selected] = core_index + 1
        only_low_loss = (core_choice == 0) & (low_count > 0)
        component_id[only_low_loss] = 1

        energy_loss_eV = low_energy_eV + core_energy_eV
        incident_energy_eV = np.asarray(incident_electrons["kinetic_energy_eV"], dtype=np.float64)
        outgoing_energy_eV = incident_energy_eV - energy_loss_eV
        if np.any(outgoing_energy_eV <= 0.0):
            raise ValueError("A sampled loss exhausted the incident kinetic energy")

        low_sigma_mrad = np.zeros(count, dtype=np.float64)
        if library.low_loss is not None:
            low_sigma_mrad = library.low_loss.angular_sigma_mrad * np.sqrt(low_count)
        sigma_mrad = angular_scale * np.sqrt(
            library.elastic_angular_sigma_mrad**2 + low_sigma_mrad**2 + core_sigma_mrad**2
        )
        theta_x_mrad = context.rng.normal(0.0, sigma_mrad)
        theta_y_mrad = context.rng.normal(0.0, sigma_mrad)
        incident_directions = np.column_stack(
            (
                incident_electrons["dir_x"],
                incident_electrons["dir_y"],
                incident_electrons["dir_z"],
            )
        )
        outgoing_directions = scatter_directions(incident_directions, theta_x_mrad, theta_y_mrad)

        outgoing = {name: np.asarray(values).copy() for name, values in incident_electrons.items()}
        outgoing["z_um"] = (
            np.asarray(incident_electrons["z_um"], dtype=np.float64)
            + library.specimen_thickness_A * 1.0e-4
        )
        outgoing["dir_x"] = outgoing_directions[:, 0]
        outgoing["dir_y"] = outgoing_directions[:, 1]
        outgoing["dir_z"] = outgoing_directions[:, 2]
        outgoing["kinetic_energy_eV"] = outgoing_energy_eV
        outgoing["weight"] = np.ones(count, dtype=np.float64)
        outgoing["loss_channel"] = np.where(
            core_choice > 0, 3, np.where(low_count > 0, 2, 0)
        ).astype(np.int32)
        outgoing["parent_electron_id"] = np.asarray(
            incident_electrons["electron_id"], dtype=np.uint64
        ).copy()
        outgoing["branch_id"] = component_id.astype(np.uint32)
        outgoing["spectral_component_id"] = component_id
        plural_order = low_count + (core_choice > 0)
        outgoing["plural_order"] = plural_order.astype(np.uint16)

        component_labels = {"0": "zero_loss", "1": "low_loss"}
        component_labels.update(
            {str(index + 2): component.name for index, component in enumerate(library.core_loss)}
        )
        metadata = {
            "model": "tabulated energy-resolved single-electron kernel",
            "material": library.material,
            "library_file": str(library_path),
            "library_metadata": dict(library.metadata),
            "component_labels": component_labels,
            "component_counts": {
                "zero_loss": int(np.count_nonzero(component_id == 0)),
                "low_loss": int(np.count_nonzero(component_id == 1)),
                **component_counts,
            },
            "mean_low_loss_events": (
                0.0
                if library.low_loss is None
                else float(library.low_loss.mean_events * low_loss_scale)
            ),
            "total_core_probability": total_core_probability,
            "plural_scattering_fraction": float(np.mean(plural_order > 1)),
            "clipped_plural_events": clipped_plural_events,
            "random_seed": context.random_seed,
        }
        return SpecimenResult(
            electrons=outgoing,
            weight_semantics="individual_electrons",
            metadata=metadata,
        )
