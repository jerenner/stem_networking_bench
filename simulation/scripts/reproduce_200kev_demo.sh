#!/usr/bin/env bash
set -euo pipefail

simulation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$simulation_dir"

dark_hdf5="${EELS_DARK_HDF5:-}"
spectrum_hdf5="${EELS_SPECTRUM_HDF5:-}"
physics_python="${EELS_SIM_PYTHON:-$simulation_dir/.conda-envs/eels-sim-gpaw/bin/python}"
manim_python="${LMTO_MANIM_PYTHON:-$simulation_dir/.venv-demo/bin/python}"

if [[ -z "$dark_hdf5" || -z "$spectrum_hdf5" ]]; then
    echo "Set EELS_DARK_HDF5 and EELS_SPECTRUM_HDF5 to the measured NiO stacks." >&2
    exit 2
fi
if [[ ! -f "$dark_hdf5" || ! -f "$spectrum_hdf5" ]]; then
    echo "One or both measured NiO HDF5 inputs do not exist." >&2
    exit 2
fi
if [[ ! -x "$physics_python" ]]; then
    echo "Physics environment missing; run scripts/setup_gpaw_env.sh first." >&2
    exit 2
fi
if [[ ! -x "$manim_python" ]]; then
    echo "Manim environment missing; run scripts/setup_demo_env.sh first." >&2
    exit 2
fi
if [[ -z "${EELS_GEANT4_PREFIX:-}" ]]; then
    echo "Set EELS_GEANT4_PREFIX to the Geant4 installation prefix." >&2
    exit 2
fi

mkdir -p output calibration spectral_libraries/gosh

PYTHONPATH=python "$physics_python" scripts/download_gosh.py
PYTHONPATH=python "$physics_python" scripts/generate_lmto_library.py \
    --beam-energy-eV 200000 \
    --output spectral_libraries/lmto_gosh_200keV.h5

PYTHONPATH=python "$physics_python" -m eels_sim generate-lmto-abtem-spatial \
    --config config/energy_lmto_abtem_200keV.toml \
    --output-hdf5 output/lmto_abtem_spatial_200keV.h5
PYTHONPATH=python "$physics_python" -m eels_sim refine-lmto-abtem-haadf \
    --config config/energy_lmto_abtem_200keV.toml \
    --input-hdf5 output/lmto_abtem_spatial_200keV.h5 \
    --output-hdf5 output/lmto_abtem_spatial_200keV_refined.h5 \
    --potential-sampling-A 0.09
PYTHONPATH=python "$physics_python" -m eels_sim simulate-lmto-scan \
    --config config/energy_lmto_abtem_200keV.toml \
    --output-hdf5 output/lmto_abtem_200keV_spectrum_image.h5

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$EELS_GEANT4_PREFIX"
cmake --build build --parallel "${EELS_BUILD_JOBS:-4}"
scripts/run_geant4_local.sh macros/detector_response_200keV.mac
PYTHONPATH=python "$physics_python" -m eels_sim build-detector-response-kernel \
    --config config/energy_lmto_abtem_200keV.toml \
    --geant4-base output/detector_response_200keV \
    --output-hdf5 calibration/monte_carlo_200keV_response_kernel.h5 \
    --radius-pixels 3

PYTHONPATH=python "$physics_python" -m eels_sim calibrate-dark \
    "$dark_hdf5" \
    --output calibration/eels_15pa_dark_calibration.h5 \
    --calibration-frames 256 \
    --validation-frames 256 \
    --known-defect 0:480,2272:2288
PYTHONPATH=python "$physics_python" -m eels_sim calibrate-nio-sparse-detector \
    --spectrum-hdf5 "$spectrum_hdf5" \
    --dark-hdf5 "$dark_hdf5" \
    --readout-calibration calibration/eels_15pa_dark_calibration.h5 \
    --response-kernel calibration/monte_carlo_200keV_response_kernel.h5 \
    --output-hdf5 calibration/nio_15pa_sparse_detector_calibration_200keV.h5 \
    --output-png calibration/nio_15pa_sparse_detector_calibration_200keV.png

PYTHONPATH=python "$physics_python" -m eels_sim simulate-raw-doeels \
    --config config/energy_lmto_abtem_200keV.toml \
    --spectrum-image output/lmto_abtem_200keV_spectrum_image.h5 \
    --output-hdf5 output/lmto_raw_doeels_200keV_87khz_nio_calibrated_stride4.h5 \
    --scan-stride 4
PYTHONPATH=python "$physics_python" -m eels_sim simulate-raw-doeels-exposure-sweep \
    --config config/energy_lmto_abtem_200keV.toml \
    --spectrum-image output/lmto_abtem_200keV_spectrum_image.h5 \
    --output-hdf5 output/lmto_raw_doeels_200keV_exposure_sweep_16x16.h5 \
    --exposure 1 --exposure 10 --exposure 30 --exposure 100 \
    --exposure 300 --exposure 1000 --exposure 3000 --exposure 10000 \
    --exposure 30000 --exposure 100000

LMTO_ASSET_PYTHON="$physics_python" \
LMTO_MANIM_PYTHON="$manim_python" \
    demo/render_demo.sh full

echo "Wrote demo/renders/lmto_doeels_workflow_200keV_FullDemo.mp4"
