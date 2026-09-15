#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Run one synthetic benchmark directly inside an already-running GPU container.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RX_BINARY="${STEM_DAQIRI_RX_BIN:-/opt/stem_daqiri/bin/stem_daqiri_rx}"
CONFIG="${REPO_ROOT}/cpp_daqiri/configs/synthetic_benchmark/b200_validate.yaml"
DARK="${REPO_ROOT}/walking_dot_dark_frame.h5"
GPU_DEVICE="${GPU_DEVICE:-0}"
OUTPUT_ROOT="${PWD}/synthetic_benchmark_runs"
SECONDS_OVERRIDE=""

usage() {
    cat <<EOF
Usage:
  $0 [options]

Run a synthetic receiver benchmark directly in the current container. This is
the Runpod/native counterpart of run_synthetic_benchmark.sh: it does not invoke
Docker, require a Docker daemon, or initialize NIC/DPDK resources.

Options:
  --config PATH       benchmark YAML (default: ${CONFIG})
  --dark PATH         dark-frame HDF5 file (default: ${DARK})
  --binary PATH       installed RX executable (default: ${RX_BINARY})
  --gpu INDEX         physical GPU made visible as CUDA device 0
                      (default: ${GPU_DEVICE})
  --seconds N         override synthetic.duration_seconds
  --output-root PATH  parent directory for timestamped runs
                      (default: ${OUTPUT_ROOT})
  -h, --help          show this help
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
        --config)
            require_value "$@"
            CONFIG="$2"
            shift 2
            ;;
        --dark)
            require_value "$@"
            DARK="$2"
            shift 2
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
            SECONDS_OVERRIDE="$2"
            shift 2
            ;;
        --output-root)
            require_value "$@"
            OUTPUT_ROOT="$2"
            shift 2
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

for path in "${CONFIG}" "${DARK}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file not found: ${path}" >&2
        exit 1
    fi
done
if [[ ! -x "${RX_BINARY}" ]]; then
    echo "RX executable not found or not executable: ${RX_BINARY}" >&2
    exit 1
fi
if ! [[ "${GPU_DEVICE}" =~ ^[0-9]+$ ]]; then
    echo "--gpu must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "${SECONDS_OVERRIDE}" ]] && ! [[ "${SECONDS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--seconds must be a positive integer." >&2
    exit 2
fi
for command in awk nvidia-smi sha256sum tee; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Required command not found: ${command}" >&2
        exit 1
    fi
done

CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd -P)/$(basename "${CONFIG}")"
DARK="$(cd "$(dirname "${DARK}")" && pwd -P)/$(basename "${DARK}")"
RX_BINARY="$(cd "$(dirname "${RX_BINARY}")" && pwd -P)/$(basename "${RX_BINARY}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd -P)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="$(basename "${CONFIG}" .yaml)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}_${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
    RUN_DIR="${RUN_DIR}_$$"
fi
mkdir -p "${RUN_DIR}"

cp "${CONFIG}" "${RUN_DIR}/config.source.yaml"
RUNTIME_CONFIG="${RUN_DIR}/config.yaml"

# Preserve the source YAML while making its container bind-mount calibration
# path point to the actual file uploaded to the Pod.
DARK_FRAME_PATH="${DARK}" awk '
    BEGIN { replacements = 0 }
    /^[[:space:]]*dark_frame_path:[[:space:]]*/ {
        prefix = $0
        sub(/dark_frame_path:.*/, "", prefix)
        print prefix "dark_frame_path: \"" ENVIRON["DARK_FRAME_PATH"] "\""
        replacements++
        next
    }
    { print }
    END {
        if (replacements != 1) {
            print "Expected exactly one processor.dark_frame_path entry; found " \
                  replacements > "/dev/stderr"
            exit 42
        }
    }
' "${CONFIG}" >"${RUNTIME_CONFIG}"

TELEMETRY_PID=""
cleanup() {
    if [[ -n "${TELEMETRY_PID}" ]]; then
        kill "${TELEMETRY_PID}" >/dev/null 2>&1 || true
        wait "${TELEMETRY_PID}" 2>/dev/null || true
        TELEMETRY_PID=""
    fi
}
trap cleanup EXIT INT TERM

{
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "execution_mode=native"
    echo "hostname=$(hostname)"
    echo "architecture=$(uname -m)"
    echo "kernel=$(uname -r)"
    echo "runpod_pod_id=${RUNPOD_POD_ID:-unavailable}"
    echo "gpu_device=${GPU_DEVICE}"
    echo "config_source=${CONFIG}"
    echo "config_source_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
    echo "config_runtime_sha256=$(sha256sum "${RUNTIME_CONFIG}" | awk '{print $1}')"
    echo "dark_source=${DARK}"
    echo "dark_sha256=$(sha256sum "${DARK}" | awk '{print $1}')"
    echo "rx_binary=${RX_BINARY}"
    echo "rx_binary_sha256=$(sha256sum "${RX_BINARY}" | awk '{print $1}')"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "seconds_override=${SECONDS_OVERRIDE:-none}"
} >"${RUN_DIR}/metadata.txt"

nvidia-smi -i "${GPU_DEVICE}" -q >"${RUN_DIR}/nvidia_smi_before.txt"
nvidia-smi \
    -i "${GPU_DEVICE}" \
    --query-gpu=timestamp,index,name,uuid,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem \
    --format=csv,nounits \
    --loop=1 >"${RUN_DIR}/nvidia_smi.csv" 2>"${RUN_DIR}/nvidia_smi.stderr.log" &
TELEMETRY_PID=$!

RUN_COMMAND=(
    "${RX_BINARY}"
    "${RUNTIME_CONFIG}"
)
if [[ -n "${SECONDS_OVERRIDE}" ]]; then
    RUN_COMMAND+=(--seconds "${SECONDS_OVERRIDE}")
fi

echo "Run directory: ${RUN_DIR}"
echo "Execution: native (no nested Docker)"
echo "Binary: ${RX_BINARY}"
echo "Config: ${RUNTIME_CONFIG}"
echo "Dark frame: ${DARK}"
echo "GPU: physical ${GPU_DEVICE}, exposed to CUDA as device 0"

set +e
CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    "${RUN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/daqiri.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

cleanup
nvidia-smi -i "${GPU_DEVICE}" -q >"${RUN_DIR}/nvidia_smi_after.txt"
{
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=${RUN_STATUS}"
} >>"${RUN_DIR}/metadata.txt"

if [[ ${RUN_STATUS} -ne 0 ]]; then
    echo "Benchmark failed with status ${RUN_STATUS}; logs retained in ${RUN_DIR}." >&2
    exit "${RUN_STATUS}"
fi
echo "Benchmark complete; results retained in ${RUN_DIR}."
