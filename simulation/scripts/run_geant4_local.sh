#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <macro.mac>" >&2
  exit 2
fi

EELS_SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${EELS_GEANT4_PREFIX:-}" ]]; then
  echo "Set EELS_GEANT4_PREFIX to the Geant4 installation prefix." >&2
  exit 2
fi

source "${EELS_GEANT4_PREFIX}/bin/geant4.sh"
if [[ -n "${EELS_RUNTIME_LIBRARY_PATH:-}" ]]; then
  export DYLD_LIBRARY_PATH="${EELS_RUNTIME_LIBRARY_PATH}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${EELS_RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${EELS_SIM_ROOT}"
exec ./build/cpp/eels_detector "$1"
