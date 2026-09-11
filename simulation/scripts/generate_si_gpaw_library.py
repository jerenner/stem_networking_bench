#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from eels_sim.spectral_library import write_spectral_library
from eels_sim.spectral_models import si_bulk_library_from_gpaw


def calculate_gpaw_loss(
    work_directory: Path,
    k_points: int,
    ground_state_cutoff_eV: float,
    response_cutoff_eV: float,
    bands: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    from ase.build import bulk
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.response.df import DielectricFunction

    work_directory.mkdir(parents=True, exist_ok=True)
    gpw_path = work_directory / "si_bulk.gpw"
    response_path = work_directory / "si_bulk_eels.csv"
    if not gpw_path.exists():
        atoms = bulk("Si", "diamond", a=5.431)
        atoms.calc = GPAW(
            mode=PW(ground_state_cutoff_eV),
            xc="LDA",
            kpts=(k_points, k_points, k_points),
            occupations=FermiDirac(0.001),
            convergence={"density": 1.0e-6},
            txt=str(work_directory / "si_ground_state.txt"),
        )
        atoms.get_potential_energy()
        atoms.calc.diagonalize_full_hamiltonian(nbands=bands)
        atoms.calc.write(gpw_path, mode="all")
    if not response_path.exists():
        dielectric = DielectricFunction(
            calc=str(gpw_path),
            frequencies={
                "type": "nonlinear",
                "domega0": 0.05,
                "omega2": 10.0,
                "omegamax": 60.0,
            },
            ecut=response_cutoff_eV,
            nbands=bands,
            eta=0.2,
            txt=str(work_directory / "si_response.txt"),
        )
        dielectric.get_eels_spectrum(q_c=[1.0 / k_points, 0.0, 0.0], filename=str(response_path))
    values = np.loadtxt(response_path, delimiter=",")
    return values[:, 0], values[:, 2], response_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the bulk-Si energy library using GPAW-RPA low loss"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spectral_libraries/si_bulk_300keV.h5"),
    )
    parser.add_argument("--work-directory", type=Path, default=Path("output/gpaw_si"))
    parser.add_argument("--loss-csv", type=Path)
    parser.add_argument("--beam-energy-eV", type=float, default=300_000.0)
    parser.add_argument("--thickness-A", type=float, default=500.0)
    parser.add_argument("--k-points", type=int, default=6)
    parser.add_argument("--ground-state-cutoff-eV", type=float, default=300.0)
    parser.add_argument("--response-cutoff-eV", type=float, default=50.0)
    parser.add_argument("--bands", type=int, default=24)
    args = parser.parse_args()

    if args.loss_csv is None:
        energy, loss, source_path = calculate_gpaw_loss(
            args.work_directory,
            args.k_points,
            args.ground_state_cutoff_eV,
            args.response_cutoff_eV,
            args.bands,
        )
    else:
        values = np.loadtxt(args.loss_csv, delimiter=",")
        energy, loss, source_path = values[:, 0], values[:, 2], args.loss_csv
    library = si_bulk_library_from_gpaw(
        energy,
        loss,
        beam_energy_eV=args.beam_energy_eV,
        specimen_thickness_A=args.thickness_A,
    )
    metadata = dict(library.metadata)
    metadata["gpaw_response_csv"] = str(source_path)
    library = type(library)(
        material=library.material,
        beam_energy_eV=library.beam_energy_eV,
        specimen_thickness_A=library.specimen_thickness_A,
        elastic_angular_sigma_mrad=library.elastic_angular_sigma_mrad,
        low_loss=library.low_loss,
        core_loss=library.core_loss,
        metadata=metadata,
    )
    write_spectral_library(args.output, library)
    print(f"wrote {library.material} spectral library to {args.output}")
    print(f"GPAW loss function: {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
