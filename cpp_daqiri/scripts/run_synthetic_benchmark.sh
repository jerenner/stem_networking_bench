#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Run one synthetic benchmark configuration and capture reproducibility data.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${STEM_DAQIRI_IMAGE:-stem_daqiri:synthetic-amd64}"
CONFIG="${REPO_ROOT}/cpp_daqiri/configs/synthetic_benchmark/cisneros_smoke.yaml"
DARK="${REPO_ROOT}/walking_dot_dark_frame.h5"
GPU_DEVICE="${GPU_DEVICE:-0}"
OUTPUT_ROOT="${PWD}/synthetic_benchmark_runs"
SECONDS_OVERRIDE=""

usage() {
    cat <<EOF
Usage:
  $0 [options]

Run a direct (no supervisor) synthetic receiver benchmark and record the exact
configuration, image/Git identity, application output, and GPU telemetry.

Options:
  --config PATH       benchmark YAML (default: ${CONFIG})
  --dark PATH         dark-frame HDF5 file (default: ${DARK})
  --image NAME        container image (default: ${IMAGE})
  --gpu INDEX         physical GPU exposed as device 0 (default: ${GPU_DEVICE})
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
        --image)
            require_value "$@"
            IMAGE="$2"
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
if ! [[ "${GPU_DEVICE}" =~ ^[0-9]+$ ]]; then
    echo "--gpu must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "${SECONDS_OVERRIDE}" ]] && ! [[ "${SECONDS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--seconds must be a positive integer." >&2
    exit 2
fi
for command in docker nvidia-smi sha256sum; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Required command not found: ${command}" >&2
        exit 1
    fi
done
if ! docker info >/dev/null 2>&1; then
    echo "Docker is not available to the current user." >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Container image not found: ${IMAGE}" >&2
    exit 1
fi

CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd -P)/$(basename "${CONFIG}")"
DARK="$(cd "$(dirname "${DARK}")" && pwd -P)/$(basename "${DARK}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd -P)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="$(basename "${CONFIG}" .yaml)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}_${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
    RUN_DIR="${RUN_DIR}_$$"
fi
mkdir -p "${RUN_DIR}"

cp "${CONFIG}" "${RUN_DIR}/config.yaml"
CONTAINER_NAME="stem-synthetic-${RUN_ID,,}-$$"
TELEMETRY_PID=""

cleanup() {
    if [[ -n "${TELEMETRY_PID}" ]]; then
        kill "${TELEMETRY_PID}" >/dev/null 2>&1 || true
        wait "${TELEMETRY_PID}" 2>/dev/null || true
        TELEMETRY_PID=""
    fi
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

{
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "architecture=$(uname -m)"
    echo "kernel=$(uname -r)"
    echo "gpu_device=${GPU_DEVICE}"
    echo "config_source=${CONFIG}"
    echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
    echo "dark_source=${DARK}"
    echo "dark_sha256=$(sha256sum "${DARK}" | awk '{print $1}')"
    echo "image=${IMAGE}"
    echo "image_id=$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
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
    /opt/stem_daqiri/bin/stem_daqiri_rx
    /run/stem_rx.yaml
)
if [[ -n "${SECONDS_OVERRIDE}" ]]; then
    RUN_COMMAND+=(--seconds "${SECONDS_OVERRIDE}")
fi

echo "Run directory: ${RUN_DIR}"
echo "Image: ${IMAGE}"
echo "Config: ${CONFIG}"
echo "GPU: ${GPU_DEVICE}"

set +e
docker run --rm \
    --name "${CONTAINER_NAME}" \
    --network none \
    --gpus "device=${GPU_DEVICE}" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${CONFIG}:/run/stem_rx.yaml:ro" \
    -v "${DARK}:/calibration/dark.h5:ro" \
    "${IMAGE}" \
    "${RUN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/daqiri.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

kill "${TELEMETRY_PID}" >/dev/null 2>&1 || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
TELEMETRY_PID=""
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
