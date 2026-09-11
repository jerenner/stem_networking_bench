#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from eels_sim.exspy_gosh import DFT_GOSH_FILENAME, lmto_gosh_library
from eels_sim.spectral_library import write_spectral_library


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spectral_libraries/lmto_gosh_300keV.h5"),
    )
    parser.add_argument(
        "--gosh-file",
        type=Path,
        default=Path("spectral_libraries/gosh") / DFT_GOSH_FILENAME,
    )
    parser.add_argument("--beam-energy-eV", type=float, default=300_000.0)
    parser.add_argument("--thickness-A", type=float, default=500.0)
    parser.add_argument("--density-g-cm3", type=float, default=4.0)
    parser.add_argument("--convergence-semiangle-mrad", type=float, default=20.0)
    parser.add_argument("--collection-semiangle-mrad", type=float, default=50.0)
    parser.add_argument("--maximum-energy-loss-eV", type=float, default=790.0)
    parser.add_argument("--energy-step-eV", type=float, default=0.25)
    args = parser.parse_args()
    library = lmto_gosh_library(
        gosh_file=args.gosh_file,
        beam_energy_eV=args.beam_energy_eV,
        specimen_thickness_A=args.thickness_A,
        density_g_cm3=args.density_g_cm3,
        convergence_semiangle_mrad=args.convergence_semiangle_mrad,
        collection_semiangle_mrad=args.collection_semiangle_mrad,
        maximum_energy_loss_eV=args.maximum_energy_loss_eV,
        energy_step_eV=args.energy_step_eV,
    )
    write_spectral_library(args.output, library)
    print(f"wrote {library.material} spectral library to {args.output}")
    print(
        "total core-event probability: " f"{library.metadata['total_core_event_probability']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
