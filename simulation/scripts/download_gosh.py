#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pooch
from eels_sim.exspy_gosh import (
    DFT_GOSH_FILENAME,
    DFT_GOSH_MD5,
    DFT_GOSH_URL,
    validate_dft_gosh_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned public DFT-GOSH database")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spectral_libraries/gosh") / DFT_GOSH_FILENAME,
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = pooch.retrieve(
        url=DFT_GOSH_URL,
        known_hash=f"md5:{DFT_GOSH_MD5}",
        path=args.output.parent,
        fname=args.output.name,
        progressbar=True,
    )
    metadata = validate_dft_gosh_file(path)
    print(f"DFT-GOSH database: {metadata['path']}")
    print(f"DOI={metadata['doi']}, MD5={metadata['md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
