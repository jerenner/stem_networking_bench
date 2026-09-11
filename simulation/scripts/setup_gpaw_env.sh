#!/usr/bin/env bash
set -euo pipefail

simulation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_prefix="$simulation_dir/.conda-envs/eels-sim-gpaw"
package_cache="$simulation_dir/.conda-pkgs"

if ! command -v conda >/dev/null 2>&1; then
    for conda_setup in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_setup" ]]; then
            # shellcheck disable=SC1090
            source "$conda_setup"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Could not find Conda; initialize Conda before running this script." >&2
    exit 1
fi
export CONDA_PKGS_DIRS="$package_cache"

if [[ -x "$environment_prefix/bin/python" ]]; then
    conda env update --prefix "$environment_prefix" \
        --file "$simulation_dir/environment-gpaw.yml"
else
    conda env create --prefix "$environment_prefix" \
        --file "$simulation_dir/environment-gpaw.yml"
fi

conda activate "$environment_prefix"
export GPAW_CONFIG="$simulation_dir/config/gpaw_siteconfig.py"

# GPAW has no native osx-arm64 conda-forge binary. Building 25.7.0 here uses
# the activated Conda compiler and numerical libraries. Rebuild only if the
# pinned version with LibXC support is not already present.
if ! python -c "import gpaw; assert gpaw.__version__ == '25.7.0'" 2>/dev/null \
    || ! gpaw info | grep -Eq 'libxc-[^ ]+[[:space:]]+yes'; then
    python -m pip install --no-build-isolation --no-cache-dir \
        --force-reinstall --no-deps "gpaw==25.7.0"
fi
python -m pip install --no-build-isolation --no-deps -e "$simulation_dir"

gpaw info
python -c "import abtem, exspy, gpaw, eels_sim; print('abTEM', abtem.__version__, 'eXSpy', exspy.__version__, 'GPAW', gpaw.__version__, 'eels-sim', eels_sim.__version__)"
