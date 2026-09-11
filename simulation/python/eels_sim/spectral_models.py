from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .spectral_library import (
    CoreLossComponent,
    LowLossComponent,
    SpectralDistribution,
    SpectralLibrary,
)


def gaussian_mixture_distribution(
    energy_min_eV: float,
    energy_max_eV: float,
    energy_step_eV: float,
    peaks: Sequence[tuple[float, float, float]],
) -> SpectralDistribution:
    energy = np.arange(energy_min_eV, energy_max_eV + 0.5 * energy_step_eV, energy_step_eV)
    density = np.zeros_like(energy)
    for center_eV, sigma_eV, strength in peaks:
        density += strength * np.exp(-0.5 * ((energy - center_eV) / sigma_eV) ** 2)
    return SpectralDistribution(energy, density)


def edge_distribution(
    onset_eV: float,
    maximum_eV: float,
    step_eV: float,
    peaks: Sequence[tuple[float, float, float]] = (),
    decay_power: float = 1.7,
    rise_eV: float = 2.0,
) -> SpectralDistribution:
    """Construct a smooth continuum plus optional near-edge peaks.

    This is useful for demonstrators and interface validation. It is not a
    replacement for an ab-initio ELNES/GOS calculation.
    """
    energy = np.arange(onset_eV, maximum_eV + 0.5 * step_eV, step_eV)
    excess = energy - onset_eV
    continuum = (1.0 - np.exp(-excess / rise_eV)) * (
        onset_eV / np.maximum(energy, onset_eV)
    ) ** decay_power
    density = continuum
    for offset_eV, sigma_eV, strength in peaks:
        density = density + strength * np.exp(-0.5 * ((excess - offset_eV) / sigma_eV) ** 2)
    # Retain a finite first-bin probability while keeping support at the onset.
    density[0] = max(density[0], 1.0e-12)
    return SpectralDistribution(energy, density)


def si_bulk_library_from_gpaw(
    energy_eV: np.ndarray,
    loss_function: np.ndarray,
    beam_energy_eV: float = 300_000.0,
    specimen_thickness_A: float = 500.0,
) -> SpectralLibrary:
    energy = np.asarray(energy_eV, dtype=np.float64)
    density = np.clip(np.asarray(loss_function, dtype=np.float64), 0.0, None)
    selected = (energy >= 0.5) & (energy <= 60.0)
    if np.count_nonzero(selected) < 2 or not np.any(density[selected] > 0.0):
        raise ValueError("GPAW loss function has no positive support from 0.5 to 60 eV")
    low_loss = LowLossComponent(
        distribution=SpectralDistribution(energy[selected], density[selected]),
        mean_events=0.35,
        angular_sigma_mrad=0.35,
        source="GPAW RPA macroscopic EELS loss function with local-field effects",
    )
    si_l23 = CoreLossComponent(
        name="Si_L23",
        element="Si",
        edge="L2,3",
        onset_energy_eV=99.2,
        integrated_probability=0.025,
        angular_sigma_mrad=1.1,
        distribution=edge_distribution(
            99.2,
            240.0,
            0.2,
            peaks=((2.2, 0.8, 1.4), (7.5, 2.0, 0.7), (23.0, 5.0, 0.25)),
            decay_power=1.8,
            rise_eV=1.0,
        ),
        source="analytic continuum proxy; onset from standard EELS reference data",
        metadata={
            "quantitative": False,
            "warning": "Approximate Si L2,3 shape, not a GPAW/FEFF ELNES calculation",
        },
    )
    return SpectralLibrary(
        material="bulk diamond Si",
        beam_energy_eV=beam_energy_eV,
        specimen_thickness_A=specimen_thickness_A,
        elastic_angular_sigma_mrad=0.18,
        low_loss=low_loss,
        core_loss=(si_l23,),
        metadata={
            "purpose": "bulk-Si energy-kernel validation",
            "absolute_scaling": "provisional event rates for a 50 nm specimen",
            "low_loss_physics": "first-principles GPAW RPA dielectric response",
            "core_loss_physics": "analytic validation proxy",
        },
    )


def lmto_demo_library(
    beam_energy_eV: float = 300_000.0,
    specimen_thickness_A: float = 500.0,
) -> SpectralLibrary:
    """Return an explicitly non-quantitative Li1.2Mn0.4Ti0.4O2 demonstrator."""
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
        source="analytic LMTO demonstrator low-loss mixture",
    )
    definitions = (
        ("Ti_M23", "Ti", "M2,3", 35.0, 0.018, 0.9, ((2.0, 0.8, 1.0),)),
        ("Mn_M23", "Mn", "M2,3", 49.0, 0.016, 0.9, ((2.5, 1.0, 0.8),)),
        ("Li_K", "Li", "K", 54.7, 0.014, 0.8, ((5.0, 2.0, 0.5),)),
        (
            "Ti_L23",
            "Ti",
            "L2,3",
            456.0,
            0.009,
            1.5,
            ((2.0, 0.9, 1.8), (7.8, 1.1, 1.2)),
        ),
        ("O_K", "O", "K", 532.0, 0.020, 1.4, ((3.0, 1.4, 1.2), (12.0, 3.0, 0.5))),
        (
            "Mn_L23",
            "Mn",
            "L2,3",
            640.0,
            0.009,
            1.7,
            ((1.0, 1.1, 1.8), (11.0, 1.3, 1.0)),
        ),
    )
    components = []
    for name, element, edge, onset, probability, sigma, peaks in definitions:
        maximum = min(onset + 120.0, 790.0)
        components.append(
            CoreLossComponent(
                name=name,
                element=element,
                edge=edge,
                onset_energy_eV=onset,
                integrated_probability=probability,
                angular_sigma_mrad=sigma,
                distribution=edge_distribution(
                    onset,
                    maximum,
                    0.2,
                    peaks=peaks,
                    decay_power=1.75,
                    rise_eV=1.5,
                ),
                source="analytic edge continuum with approximate reference onset",
                metadata={"quantitative": False},
            )
        )
    return SpectralLibrary(
        material="Li1.2Mn0.4Ti0.4O2 (LMTO demonstrator)",
        beam_energy_eV=beam_energy_eV,
        specimen_thickness_A=specimen_thickness_A,
        elastic_angular_sigma_mrad=0.22,
        low_loss=low_loss,
        core_loss=tuple(components),
        metadata={
            "purpose": "LMTO 1D sample-data demonstrator",
            "quantitative": False,
            "warning": (
                "Edge onsets are representative; shapes and integrated "
                "probabilities are provisional, not a material prediction"
            ),
            "formula": "Li1.2Mn0.4Ti0.4O2",
        },
    )
