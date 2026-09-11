from __future__ import annotations

import warnings
from dataclasses import dataclass
from hashlib import md5
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .spectral_library import (
    CoreLossComponent,
    LowLossComponent,
    SpectralDistribution,
    SpectralLibrary,
)
from .spectral_models import gaussian_mixture_distribution

DFT_GOSH_DOI = "10.5281/zenodo.7645765"
DFT_GOSH_FILENAME = "Segger_Guzzinati_Kohl_1.5.0.gosh"
DFT_GOSH_MD5 = "7fee8891c147a4f769668403b54c529b"
DFT_GOSH_URL = f"doi:{DFT_GOSH_DOI}/{DFT_GOSH_FILENAME}"
AVOGADRO_PER_MOL = 6.02214076e23
BARN_TO_CM2 = 1.0e-24


@dataclass(frozen=True)
class GOSComponentDefinition:
    name: str
    element: str
    edge: str
    subshells: tuple[str, ...]
    stoichiometric_coefficient: float
    angular_sigma_mrad: float


LMTO_GOS_COMPONENTS = (
    GOSComponentDefinition("Ti_M23", "Ti", "M2,3", ("Ti_M3", "Ti_M2"), 0.4, 0.9),
    GOSComponentDefinition("Mn_M23", "Mn", "M2,3", ("Mn_M3", "Mn_M2"), 0.4, 0.9),
    GOSComponentDefinition("Li_K", "Li", "K", ("Li_K",), 1.2, 0.8),
    GOSComponentDefinition("Ti_L23", "Ti", "L2,3", ("Ti_L3", "Ti_L2"), 0.4, 1.5),
    GOSComponentDefinition("O_K", "O", "K", ("O_K",), 2.0, 1.4),
    GOSComponentDefinition("Mn_L23", "Mn", "L2,3", ("Mn_L3", "Mn_L2"), 0.4, 1.7),
)

LMTO_MOLAR_MASS_G_MOL = 1.2 * 6.94 + 0.4 * 54.938044 + 0.4 * 47.867 + 2.0 * 15.999


@dataclass(frozen=True)
class EvaluatedGOSComponent:
    definition: GOSComponentDefinition
    distribution: SpectralDistribution
    onset_energy_eV: float
    integrated_cross_section_barn_per_atom: float
    subshell_metadata: tuple[dict[str, object], ...]


def validate_dft_gosh_file(path: str | Path, verify_hash: bool = True) -> dict[str, str]:
    database_path = Path(path)
    if not database_path.is_file():
        raise FileNotFoundError(
            f"DFT-GOSH database not found at {database_path}; run " "scripts/download_gosh.py"
        )
    if verify_hash:
        digest = md5(database_path.read_bytes()).hexdigest()
        if digest != DFT_GOSH_MD5:
            raise ValueError(f"DFT-GOSH MD5 mismatch: expected {DFT_GOSH_MD5}, got {digest}")
    with h5py.File(database_path) as h5:
        value = h5["metadata/data_ref"].attrs["data_doi"]
        doi = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if doi != DFT_GOSH_DOI:
        raise ValueError(f"Unexpected DFT-GOSH DOI {doi!r}")
    return {
        "path": str(database_path.resolve()),
        "doi": doi,
        "md5": DFT_GOSH_MD5,
        "filename": database_path.name,
    }


def _load_exspy_edge_class() -> Callable[..., Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Importing .* is deprecated.*")
            from exspy.components import EELSCLEdge
    except ImportError as error:
        raise RuntimeError(
            "The eXSpy/GOSH provider requires the 'gos' optional dependencies"
        ) from error
    return EELSCLEdge


def _installed_exspy_version() -> str:
    try:
        return version("exspy")
    except PackageNotFoundError:
        return "unavailable"


def evaluate_gosh_component(
    definition: GOSComponentDefinition,
    gosh_file: str | Path,
    beam_energy_keV: float,
    convergence_semiangle_mrad: float,
    collection_semiangle_mrad: float,
    maximum_energy_loss_eV: float,
    energy_step_eV: float,
    edge_class: Callable[..., Any] | None = None,
) -> EvaluatedGOSComponent:
    if beam_energy_keV <= 0.0:
        raise ValueError("beam_energy_keV must be positive")
    if convergence_semiangle_mrad < 0.0 or collection_semiangle_mrad <= 0.0:
        raise ValueError("GOS convergence/collection angles are invalid")
    if energy_step_eV <= 0.0:
        raise ValueError("energy_step_eV must be positive")
    if edge_class is None:
        edge_class = _load_exspy_edge_class()

    edges = [
        edge_class(
            subshell,
            GOS="dft",
            gos_file_path=str(Path(gosh_file).resolve()),
        )
        for subshell in definition.subshells
    ]
    for edge in edges:
        edge.set_microscope_parameters(
            E0=beam_energy_keV,
            alpha=convergence_semiangle_mrad,
            beta=collection_semiangle_mrad,
            energy_scale=energy_step_eV,
        )
    onset_energy_eV = min(float(edge.onset_energy.value) for edge in edges)
    if maximum_energy_loss_eV <= onset_energy_eV:
        raise ValueError(
            f"maximum energy loss does not include {definition.name} at " f"{onset_energy_eV:g} eV"
        )
    energy_eV = np.arange(
        onset_energy_eV,
        maximum_energy_loss_eV + 0.5 * energy_step_eV,
        energy_step_eV,
    )
    total_cross_section = np.zeros_like(energy_eV)
    subshell_metadata = []
    for subshell, edge in zip(definition.subshells, edges):
        cross_section = np.clip(np.asarray(edge.function(energy_eV), dtype=np.float64), 0.0, None)
        total_cross_section += cross_section
        subshell_metadata.append(
            {
                "subshell": subshell,
                "onset_energy_eV": float(edge.onset_energy.value),
                "effective_collection_angle_mrad": float(edge.effective_angle.value),
                "integrated_cross_section_barn_per_atom": float(
                    np.trapezoid(cross_section, energy_eV)
                ),
            }
        )
    integrated_cross_section = float(np.trapezoid(total_cross_section, energy_eV))
    if integrated_cross_section <= 0.0:
        raise ValueError(f"GOSH returned zero cross section for {definition.name}")
    return EvaluatedGOSComponent(
        definition=definition,
        distribution=SpectralDistribution(energy_eV, total_cross_section),
        onset_energy_eV=onset_energy_eV,
        integrated_cross_section_barn_per_atom=integrated_cross_section,
        subshell_metadata=tuple(subshell_metadata),
    )


def formula_unit_areal_density_cm2(
    density_g_cm3: float,
    molar_mass_g_mol: float,
    thickness_A: float,
) -> float:
    if density_g_cm3 <= 0.0 or molar_mass_g_mol <= 0.0 or thickness_A < 0.0:
        raise ValueError("density, molar mass, and thickness must be physical")
    thickness_cm = thickness_A * 1.0e-8
    return density_g_cm3 * thickness_cm * AVOGADRO_PER_MOL / molar_mass_g_mol


def exclusive_core_probabilities(optical_depths: np.ndarray) -> np.ndarray:
    """Convert thin-target optical depths into at-most-one core probabilities."""
    optical_depth = np.asarray(optical_depths, dtype=np.float64)
    if optical_depth.ndim != 1 or np.any(~np.isfinite(optical_depth)):
        raise ValueError("optical depths must be a finite 1D array")
    if np.any(optical_depth < 0.0):
        raise ValueError("optical depths must be nonnegative")
    total = float(optical_depth.sum())
    if total == 0.0:
        return np.zeros_like(optical_depth)
    probability_of_any_core_event = -np.expm1(-total)
    return optical_depth * (probability_of_any_core_event / total)


def lmto_gosh_library(
    gosh_file: str | Path,
    beam_energy_eV: float = 300_000.0,
    specimen_thickness_A: float = 500.0,
    density_g_cm3: float = 4.0,
    convergence_semiangle_mrad: float = 20.0,
    collection_semiangle_mrad: float = 50.0,
    maximum_energy_loss_eV: float = 790.0,
    energy_step_eV: float = 0.25,
    verify_gosh_hash: bool = True,
    edge_class: Callable[..., Any] | None = None,
) -> SpectralLibrary:
    database_metadata = validate_dft_gosh_file(gosh_file, verify_hash=verify_gosh_hash)
    exspy_version = _installed_exspy_version()
    evaluated = tuple(
        evaluate_gosh_component(
            definition,
            gosh_file,
            beam_energy_keV=beam_energy_eV * 1.0e-3,
            convergence_semiangle_mrad=convergence_semiangle_mrad,
            collection_semiangle_mrad=collection_semiangle_mrad,
            maximum_energy_loss_eV=maximum_energy_loss_eV,
            energy_step_eV=energy_step_eV,
            edge_class=edge_class,
        )
        for definition in LMTO_GOS_COMPONENTS
    )
    formula_areal_density = formula_unit_areal_density_cm2(
        density_g_cm3, LMTO_MOLAR_MASS_G_MOL, specimen_thickness_A
    )
    optical_depths = np.asarray(
        [
            component.integrated_cross_section_barn_per_atom
            * component.definition.stoichiometric_coefficient
            * formula_areal_density
            * BARN_TO_CM2
            for component in evaluated
        ]
    )
    probabilities = exclusive_core_probabilities(optical_depths)
    core_components = []
    for component, optical_depth, probability in zip(evaluated, optical_depths, probabilities):
        definition = component.definition
        core_components.append(
            CoreLossComponent(
                name=definition.name,
                element=definition.element,
                edge=definition.edge,
                onset_energy_eV=component.onset_energy_eV,
                integrated_probability=float(probability),
                angular_sigma_mrad=definition.angular_sigma_mrad,
                distribution=component.distribution,
                source=(
                    f"eXSpy {exspy_version} DFT-GOSH 1.5 atomic GOS integrated over the "
                    "configured microscope aperture"
                ),
                metadata={
                    "provider": "exspy-gosh",
                    "exspy_version": exspy_version,
                    "quantitative_atomic_cross_section": True,
                    "material_specific_fine_structure": False,
                    "subshells": list(component.subshell_metadata),
                    "integrated_cross_section_barn_per_atom": (
                        component.integrated_cross_section_barn_per_atom
                    ),
                    "stoichiometric_coefficient": (definition.stoichiometric_coefficient),
                    "formula_unit_areal_density_cm2": formula_areal_density,
                    "optical_depth": float(optical_depth),
                    "probability_model": (
                        "relative optical depth times 1-exp(-total core optical depth)"
                    ),
                    "beam_energy_keV": beam_energy_eV * 1.0e-3,
                    "convergence_semiangle_mrad": convergence_semiangle_mrad,
                    "collection_semiangle_mrad": collection_semiangle_mrad,
                    "angular_distribution": (
                        "Gaussian proxy; GOSH supplies aperture-integrated energy "
                        "cross section, not sampled scattering angles"
                    ),
                    "gosh_doi": DFT_GOSH_DOI,
                },
            )
        )

    low_loss = LowLossComponent(
        distribution=gaussian_mixture_distribution(
            0.4,
            38.0,
            0.05,
            peaks=(
                (5.2, 1.1, 0.25),
                (9.5, 2.2, 0.45),
                (22.0, 4.0, 1.0),
            ),
        ),
        mean_events=0.30,
        angular_sigma_mrad=0.4,
        source="analytic LMTO low-loss mixture; unchanged in the GOSH upgrade",
    )
    return SpectralLibrary(
        material="Li1.2Mn0.4Ti0.4O2 (LMTO atomic-GOS baseline)",
        beam_energy_eV=beam_energy_eV,
        specimen_thickness_A=specimen_thickness_A,
        elastic_angular_sigma_mrad=0.22,
        low_loss=low_loss,
        core_loss=tuple(core_components),
        metadata={
            "purpose": "LMTO 1D atomic-GOS sample-data demonstrator",
            "formula": "Li1.2Mn0.4Ti0.4O2",
            "density_g_cm3": density_g_cm3,
            "density_provenance": "configurable provisional LMTO bulk density",
            "molar_mass_g_mol": LMTO_MOLAR_MASS_G_MOL,
            "formula_unit_areal_density_cm2": formula_areal_density,
            "total_core_optical_depth": float(optical_depths.sum()),
            "total_core_event_probability": float(probabilities.sum()),
            "core_probability_model": "at most one sampled core event",
            "gosh_database": database_metadata,
            "exspy_version": exspy_version,
            "microscope": {
                "beam_energy_keV": beam_energy_eV * 1.0e-3,
                "convergence_semiangle_mrad": convergence_semiangle_mrad,
                "collection_semiangle_mrad": collection_semiangle_mrad,
                "maximum_energy_loss_eV": maximum_energy_loss_eV,
                "energy_step_eV": energy_step_eV,
            },
            "core_loss_physics": ("absolute atomic DFT-GOS cross sections; no LMTO-specific ELNES"),
            "low_loss_physics": "analytic provisional mixture",
        },
    )
