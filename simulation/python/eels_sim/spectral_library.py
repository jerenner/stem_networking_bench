from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

SPECTRAL_LIBRARY_SCHEMA = "eels-sim-spectral-library-v1"


def _validated_distribution(
    energy_loss_eV: np.ndarray,
    probability_density_per_eV: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.asarray(energy_loss_eV, dtype=np.float64)
    density = np.asarray(probability_density_per_eV, dtype=np.float64)
    if energy.ndim != 1 or len(energy) < 2 or density.shape != energy.shape:
        raise ValueError(f"{name} energy and density must be equal 1D arrays")
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(density)):
        raise ValueError(f"{name} energy and density must be finite")
    if np.any(energy < 0.0) or np.any(np.diff(energy) <= 0.0):
        raise ValueError(f"{name} energy must be nonnegative and increasing")
    if np.any(density < 0.0):
        raise ValueError(f"{name} density must be nonnegative")
    area = float(np.trapezoid(density, energy))
    if area <= 0.0:
        raise ValueError(f"{name} density must have positive area")
    return energy.copy(), density / area


@dataclass(frozen=True)
class SpectralDistribution:
    energy_loss_eV: np.ndarray
    probability_density_per_eV: np.ndarray

    def __post_init__(self) -> None:
        energy, density = _validated_distribution(
            self.energy_loss_eV,
            self.probability_density_per_eV,
            "spectral distribution",
        )
        object.__setattr__(self, "energy_loss_eV", energy)
        object.__setattr__(self, "probability_density_per_eV", density)

    @property
    def bin_edges_eV(self) -> np.ndarray:
        energy = self.energy_loss_eV
        midpoints = 0.5 * (energy[:-1] + energy[1:])
        first = max(0.0, energy[0] - 0.5 * (energy[1] - energy[0]))
        last = energy[-1] + 0.5 * (energy[-1] - energy[-2])
        return np.concatenate(([first], midpoints, [last]))

    @property
    def discrete_probability(self) -> np.ndarray:
        probability = self.probability_density_per_eV * np.diff(self.bin_edges_eV)
        return probability / probability.sum()

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0:
            raise ValueError("sample count must be nonnegative")
        if count == 0:
            return np.empty(0, dtype=np.float64)
        interval_probability = (
            0.5
            * (self.probability_density_per_eV[:-1] + self.probability_density_per_eV[1:])
            * np.diff(self.energy_loss_eV)
        )
        interval_probability /= interval_probability.sum()
        indices = rng.choice(len(interval_probability), size=count, p=interval_probability)
        return rng.uniform(self.energy_loss_eV[indices], self.energy_loss_eV[indices + 1])


@dataclass(frozen=True)
class LowLossComponent:
    distribution: SpectralDistribution
    mean_events: float
    angular_sigma_mrad: float
    source: str = "unspecified"

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_events) or self.mean_events < 0.0:
            raise ValueError("low-loss mean_events must be finite and nonnegative")
        if not np.isfinite(self.angular_sigma_mrad) or self.angular_sigma_mrad < 0.0:
            raise ValueError("low-loss angular_sigma_mrad must be finite and nonnegative")


@dataclass(frozen=True)
class CoreLossComponent:
    name: str
    element: str
    edge: str
    onset_energy_eV: float
    integrated_probability: float
    angular_sigma_mrad: float
    distribution: SpectralDistribution
    source: str = "unspecified"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.element or not self.edge:
            raise ValueError("core-loss name, element, and edge must be nonempty")
        if not np.isfinite(self.onset_energy_eV) or self.onset_energy_eV <= 0.0:
            raise ValueError("core-loss onset_energy_eV must be positive")
        if (
            not np.isfinite(self.integrated_probability)
            or not 0.0 <= self.integrated_probability <= 1.0
        ):
            raise ValueError("core-loss integrated_probability must be in [0, 1]")
        if not np.isfinite(self.angular_sigma_mrad) or self.angular_sigma_mrad < 0.0:
            raise ValueError("core-loss angular_sigma_mrad must be finite and nonnegative")
        if self.distribution.energy_loss_eV[0] < self.onset_energy_eV - 1.0e-9:
            raise ValueError(f"core-loss component {self.name} has density below its onset")


@dataclass(frozen=True)
class SpectralLibrary:
    material: str
    beam_energy_eV: float
    specimen_thickness_A: float
    elastic_angular_sigma_mrad: float
    low_loss: LowLossComponent | None
    core_loss: tuple[CoreLossComponent, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.material:
            raise ValueError("spectral-library material must be nonempty")
        if not np.isfinite(self.beam_energy_eV) or self.beam_energy_eV <= 0.0:
            raise ValueError("spectral-library beam_energy_eV must be positive")
        if not np.isfinite(self.specimen_thickness_A) or self.specimen_thickness_A < 0.0:
            raise ValueError("spectral-library specimen_thickness_A must be nonnegative")
        if (
            not np.isfinite(self.elastic_angular_sigma_mrad)
            or self.elastic_angular_sigma_mrad < 0.0
        ):
            raise ValueError("spectral-library elastic_angular_sigma_mrad must be nonnegative")
        names = [component.name for component in self.core_loss]
        if len(names) != len(set(names)):
            raise ValueError("core-loss component names must be unique")
        if sum(component.integrated_probability for component in self.core_loss) > 1.0:
            raise ValueError("core-loss integrated probabilities sum above one")


def write_spectral_library(path: str | Path, library: SpectralLibrary) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = SPECTRAL_LIBRARY_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["material"] = library.material
        h5.attrs["beam_energy_eV"] = library.beam_energy_eV
        h5.attrs["specimen_thickness_A"] = library.specimen_thickness_A
        h5.attrs["elastic_angular_sigma_mrad"] = library.elastic_angular_sigma_mrad
        h5.attrs["metadata_json"] = json.dumps(library.metadata, sort_keys=True)
        h5.attrs["core_loss_order_json"] = json.dumps(
            [component.name for component in library.core_loss]
        )
        if library.low_loss is not None:
            low_group = h5.create_group("low_loss")
            low_group.attrs["mean_events"] = library.low_loss.mean_events
            low_group.attrs["angular_sigma_mrad"] = library.low_loss.angular_sigma_mrad
            low_group.attrs["source"] = library.low_loss.source
            low_group.create_dataset(
                "energy_loss_eV", data=library.low_loss.distribution.energy_loss_eV
            )
            low_group.create_dataset(
                "probability_density_per_eV",
                data=library.low_loss.distribution.probability_density_per_eV,
            )
        core_group = h5.create_group("core_loss")
        for component in library.core_loss:
            group = core_group.create_group(component.name)
            group.attrs["element"] = component.element
            group.attrs["edge"] = component.edge
            group.attrs["onset_energy_eV"] = component.onset_energy_eV
            group.attrs["integrated_probability"] = component.integrated_probability
            group.attrs["angular_sigma_mrad"] = component.angular_sigma_mrad
            group.attrs["source"] = component.source
            group.attrs["metadata_json"] = json.dumps(component.metadata, sort_keys=True)
            group.create_dataset("energy_loss_eV", data=component.distribution.energy_loss_eV)
            group.create_dataset(
                "probability_density_per_eV",
                data=component.distribution.probability_density_per_eV,
            )


def _text_attribute(group: h5py.Group, name: str) -> str:
    value = group.attrs[name]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_spectral_library(path: str | Path) -> SpectralLibrary:
    with h5py.File(path) as h5:
        if h5.attrs.get("schema") != SPECTRAL_LIBRARY_SCHEMA:
            raise ValueError(f"Unsupported spectral-library schema in {path}")
        low_loss = None
        if "low_loss" in h5:
            group = h5["low_loss"]
            low_loss = LowLossComponent(
                distribution=SpectralDistribution(
                    group["energy_loss_eV"][:],
                    group["probability_density_per_eV"][:],
                ),
                mean_events=float(group.attrs["mean_events"]),
                angular_sigma_mrad=float(group.attrs["angular_sigma_mrad"]),
                source=_text_attribute(group, "source"),
            )
        core_loss = []
        order_value = h5.attrs.get("core_loss_order_json")
        if isinstance(order_value, bytes):
            order_value = order_value.decode("utf-8")
        core_order = (
            sorted(h5["core_loss"]) if order_value is None else json.loads(str(order_value))
        )
        for name in core_order:
            group = h5["core_loss"][name]
            metadata_json = _text_attribute(group, "metadata_json")
            core_loss.append(
                CoreLossComponent(
                    name=name,
                    element=_text_attribute(group, "element"),
                    edge=_text_attribute(group, "edge"),
                    onset_energy_eV=float(group.attrs["onset_energy_eV"]),
                    integrated_probability=float(group.attrs["integrated_probability"]),
                    angular_sigma_mrad=float(group.attrs["angular_sigma_mrad"]),
                    distribution=SpectralDistribution(
                        group["energy_loss_eV"][:],
                        group["probability_density_per_eV"][:],
                    ),
                    source=_text_attribute(group, "source"),
                    metadata=json.loads(metadata_json),
                )
            )
        metadata_value = h5.attrs.get("metadata_json", "{}")
        if isinstance(metadata_value, bytes):
            metadata_value = metadata_value.decode("utf-8")
        return SpectralLibrary(
            material=_text_attribute(h5, "material"),
            beam_energy_eV=float(h5.attrs["beam_energy_eV"]),
            specimen_thickness_A=float(h5.attrs["specimen_thickness_A"]),
            elastic_angular_sigma_mrad=float(h5.attrs["elastic_angular_sigma_mrad"]),
            low_loss=low_loss,
            core_loss=tuple(core_loss),
            metadata=json.loads(str(metadata_value)),
        )
