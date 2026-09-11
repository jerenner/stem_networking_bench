#!/usr/bin/env bash
set -euo pipefail

simulation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir="$simulation_dir/.venv-demo"
bootstrap_python="${PYTHON:-python3}"

if [[ ! -x "$environment_dir/bin/python" ]]; then
    "$bootstrap_python" -m venv "$environment_dir"
fi

"$environment_dir/bin/python" -m pip install --upgrade pip
"$environment_dir/bin/python" -m pip install \
    --requirement "$simulation_dir/demo/requirements.txt"

"$environment_dir/bin/python" -c \
    "import manim; print('Manim', manim.__version__)"
