#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Run the correctness, 800-Gbit/s, and maximum-capacity B200 profiles.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${STEM_DAQIRI_IMAGE:-stem_daqiri:synthetic-amd64}"
DARK="${REPO_ROOT}/walking_dot_dark_frame.h5"
RX_BINARY="${STEM_DAQIRI_RX_BIN:-/opt/stem_daqiri/bin/stem_daqiri_rx}"
GPU_DEVICE="${GPU_DEVICE:-0}"
OUTPUT_ROOT="${PWD}/synthetic_benchmark_runs"
PERFORMANCE_SECONDS=300
RUN_MAXIMUM=true
NATIVE=false

usage() {
    cat <<EOF
Usage:
  $0 [options]

Run the three-stage single-B200 qualification sequence. Every stage uses the
generic runner and retains its own YAML, logs, checksums, and GPU telemetry.

Options:
  --dark PATH          dark-frame HDF5 file (default: ${DARK})
  --image NAME         container image (default: ${IMAGE})
  --native             run directly in the current container (Runpod)
  --binary PATH        native RX executable (default: ${RX_BINARY})
  --gpu INDEX          physical B200 exposed as device 0 (default: ${GPU_DEVICE})
  --seconds N          duration of each performance stage (default: 300)
  --output-root PATH   parent directory for timestamped runs
                       (default: ${OUTPUT_ROOT})
  --skip-maximum       omit the unconstrained capacity-margin stage
  -h, --help           show this help
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dark)
            require_value "$@"
            DARK="$2"
            shift 2
            ;;
        --image)
            require_value "$@"
            IMAGE="$2"
            shift 2
            ;;
        --native)
            NATIVE=true
            shift
            ;;
        --binary)
            require_value "$@"
            RX_BINARY="$2"
            shift 2
            ;;
        --gpu)
            require_value "$@"
            GPU_DEVICE="$2"
            shift 2
            ;;
        --seconds)
            require_value "$@"
            PERFORMANCE_SECONDS="$2"
            shift 2
            ;;
        --output-root)
            require_value "$@"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --skip-maximum)
            RUN_MAXIMUM=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "${PERFORMANCE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--seconds must be a positive integer." >&2
    exit 2
fi

if [[ "${NATIVE}" == "true" ]]; then
    RUNNER="${SCRIPT_DIR}/run_synthetic_benchmark_native.sh"
else
    RUNNER="${SCRIPT_DIR}/run_synthetic_benchmark.sh"
fi
COMMON_ARGS=(
    --dark "${DARK}"
    --gpu "${GPU_DEVICE}"
    --output-root "${OUTPUT_ROOT}"
)
if [[ "${NATIVE}" == "true" ]]; then
    COMMON_ARGS+=(--binary "${RX_BINARY}")
    echo "B200 suite execution mode: native (current Runpod container)"
else
    COMMON_ARGS+=(--image "${IMAGE}")
    echo "B200 suite execution mode: Docker (${IMAGE})"
fi

echo "Stage 1/3: eight-receiver assembly and correction validation"
"${RUNNER}" \
    --config "${REPO_ROOT}/cpp_daqiri/configs/synthetic_benchmark/b200_validate.yaml" \
    "${COMMON_ARGS[@]}"

echo "Stage 2/3: sustained 8 x 100 Gbit/s capacity gate"
"${RUNNER}" \
    --config "${REPO_ROOT}/cpp_daqiri/configs/synthetic_benchmark/b200_800gbps.yaml" \
    --seconds "${PERFORMANCE_SECONDS}" \
    "${COMMON_ARGS[@]}"

if [[ "${RUN_MAXIMUM}" == "true" ]]; then
    echo "Stage 3/3: unconstrained capacity-margin measurement"
    "${RUNNER}" \
        --config "${REPO_ROOT}/cpp_daqiri/configs/synthetic_benchmark/b200_maximum.yaml" \
        --seconds "${PERFORMANCE_SECONDS}" \
        "${COMMON_ARGS[@]}"
else
    echo "Stage 3/3: skipped by request"
fi

echo "B200 benchmark suite complete. Results: ${OUTPUT_ROOT}"
