from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .phase_space import validate_phase_space

SPECIMEN_ENTRY_POINT_GROUP = "eels_sim.specimen_kernels"
WEIGHT_SEMANTICS = {
    "expected_electrons",
    "branch_probability",
    "individual_electrons",
}


@dataclass(frozen=True)
class SpecimenContext:
    reference_energy_eV: float
    random_seed: int
    parameters: Mapping[str, object]
    rng: np.random.Generator = field(repr=False, compare=False)


@dataclass(frozen=True)
class SpecimenResult:
    electrons: Mapping[str, np.ndarray]
    weight_semantics: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class SpecimenKernel(Protocol):
    """Interface implemented by built-in and third-party specimen kernels."""

    name: str

    def simulate(
        self,
        incident_electrons: Mapping[str, np.ndarray],
        context: SpecimenContext,
    ) -> SpecimenResult:
        """Return weighted outgoing-electron phase space."""


class IdentitySpecimenKernel:
    """Diagnostic no-sample kernel used to validate plugin plumbing."""

    name = "identity"

    def simulate(
        self,
        incident_electrons: Mapping[str, np.ndarray],
        context: SpecimenContext,
    ) -> SpecimenResult:
        del context
        count = validate_phase_space(incident_electrons)
        if not np.allclose(incident_electrons["weight"], 1.0):
            raise ValueError("The identity kernel requires unit-weight incident electrons")
        outgoing = {name: np.asarray(values).copy() for name, values in incident_electrons.items()}
        outgoing.setdefault(
            "parent_electron_id",
            np.asarray(incident_electrons["electron_id"], dtype=np.uint64).copy(),
        )
        outgoing.setdefault("branch_id", np.zeros(count, dtype=np.uint32))
        return SpecimenResult(
            electrons=outgoing,
            weight_semantics="branch_probability",
            metadata={"model": "identity", "physics": "no specimen interaction"},
        )


def _load_abtem_kernel() -> SpecimenKernel:
    from .abtem_adapter import AbTEMSpecimenKernel

    return AbTEMSpecimenKernel()


def _load_energy_resolved_kernel() -> SpecimenKernel:
    from .energy_resolved import EnergyResolvedSpecimenKernel

    return EnergyResolvedSpecimenKernel()


_BUILTIN_KERNELS: dict[str, Callable[[], SpecimenKernel]] = {
    IdentitySpecimenKernel.name: IdentitySpecimenKernel,
    "abtem": _load_abtem_kernel,
    "energy-resolved": _load_energy_resolved_kernel,
}


def _entry_points() -> list[metadata.EntryPoint]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=SPECIMEN_ENTRY_POINT_GROUP))
    return list(discovered.get(SPECIMEN_ENTRY_POINT_GROUP, ()))


def available_specimen_kernels() -> dict[str, str]:
    kernels = {name: "built-in" for name in _BUILTIN_KERNELS}
    for entry_point in _entry_points():
        kernels.setdefault(entry_point.name, f"entry point {entry_point.value}")
    return dict(sorted(kernels.items()))


def _instantiate_kernel(candidate: Any, requested_name: str) -> SpecimenKernel:
    if isinstance(candidate, type):
        candidate = candidate()
    elif not hasattr(candidate, "simulate") and callable(candidate):
        candidate = candidate()
    if not hasattr(candidate, "simulate") or not callable(candidate.simulate):
        raise TypeError(f"Specimen kernel {requested_name!r} has no simulate method")
    if not hasattr(candidate, "name"):
        raise TypeError(f"Specimen kernel {requested_name!r} has no name attribute")
    return candidate


def load_specimen_kernel(name: str) -> SpecimenKernel:
    if name in _BUILTIN_KERNELS:
        return _BUILTIN_KERNELS[name]()
    if ":" in name:
        module_name, object_name = name.split(":", 1)
        candidate = getattr(import_module(module_name), object_name)
        return _instantiate_kernel(candidate, name)
    matches = [entry_point for entry_point in _entry_points() if entry_point.name == name]
    if not matches:
        available = ", ".join(available_specimen_kernels()) or "none"
        raise ValueError(f"Unknown specimen kernel {name!r}; available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Multiple specimen plugins are registered as {name!r}")
    return _instantiate_kernel(matches[0].load(), name)


def run_specimen_kernel(
    kernel: SpecimenKernel,
    incident_electrons: Mapping[str, np.ndarray],
    reference_energy_eV: float,
    parameters: Mapping[str, object],
    random_seed: int,
) -> SpecimenResult:
    validate_phase_space(incident_electrons)
    context = SpecimenContext(
        reference_energy_eV=reference_energy_eV,
        random_seed=random_seed,
        parameters=parameters,
        rng=np.random.default_rng(random_seed),
    )
    result = kernel.simulate(incident_electrons, context)
    if not isinstance(result, SpecimenResult):
        raise TypeError("Specimen kernels must return SpecimenResult")
    validate_phase_space(result.electrons)
    if result.weight_semantics not in WEIGHT_SEMANTICS:
        raise ValueError(f"weight_semantics must be one of {sorted(WEIGHT_SEMANTICS)}")
    if (
        result.weight_semantics == "branch_probability"
        and "parent_electron_id" not in result.electrons
    ):
        raise ValueError("branch_probability output requires parent_electron_id lineage")
    if result.weight_semantics == "individual_electrons" and not np.allclose(
        result.electrons["weight"], 1.0
    ):
        raise ValueError("individual_electrons output requires unit weights")
    return result
