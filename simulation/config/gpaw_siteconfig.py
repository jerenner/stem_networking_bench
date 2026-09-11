"""Native serial GPAW build settings for the project Conda environment."""

# GPAW injects these build-configuration lists before executing this file.
# ruff: noqa: F821

import os
from pathlib import Path

prefix = Path(os.environ["CONDA_PREFIX"])
include_dirs += [str(prefix / "include")]
library_dirs += [str(prefix / "lib")]
runtime_library_dirs += [str(prefix / "lib")]

# The core-loss adapter is serial today, so avoid an unnecessary MPI stack.
mpi = False
scalapack = False

if "xc" not in libraries:
    libraries.append("xc")
if "blas" not in libraries:
    libraries.append("blas")

fftw = True
if "fftw3" not in libraries:
    libraries.append("fftw3")
