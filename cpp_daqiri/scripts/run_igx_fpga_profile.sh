#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Run the live dual-FPGA DAQIRI receiver while recording IGX tegrastats.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${DAQIRI_IMAGE:-stem_daqiri:dual-fpga}"
CONFIG="${REPO_ROOT}/cpp_daqiri/configs/stem_rx_igx_fpga_dual.yaml"
DARK=""
RUN_DURATION_SECONDS=300
TEGRASTATS_INTERVAL_MS=1000
OUTPUT_ROOT="${PWD}/daqiri_profiles"

usage() {
    cat <<EOF
Usage:
  $0 [options]

Run the live DAQIRI FPGA receiver and tegrastats together. The default run is
300 seconds with one tegrastats sample per second.

Options:
  --config PATH               DAQIRI RX YAML (default: ${CONFIG})
  --dark PATH                 dark-frame HDF5 mounted at /calibration/dark.h5
  --seconds N                 receiver duration (default: ${RUN_DURATION_SECONDS})
  --tegrastats-interval-ms N  platform sample interval (default: ${TEGRASTATS_INTERVAL_MS})
  --output-root PATH          parent directory for timestamped runs
                              (default: ${OUTPUT_ROOT})
  --image NAME                container image (default: ${IMAGE})
  -h, --help                  show this help

Each run creates a unique directory containing:
  config.yaml                 exact receiver configuration used
  run_metadata.txt            host, hashes, and synchronized start/end times
  daqiri.log                  timestamped DAQIRI stdout/stderr
  tegrastats.log              raw tegrastats samples
  tegrastats_timeline.tsv     samples with a nominal elapsed-time column
  nvidia_smi.csv              discrete-GPU metrics when nvidia-smi is available
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
        --seconds)
            require_value "$@"
            RUN_DURATION_SECONDS="$2"
            shift 2
            ;;
        --tegrastats-interval-ms)
            require_value "$@"
            TEGRASTATS_INTERVAL_MS="$2"
            shift 2
            ;;
        --output-root)
            require_value "$@"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --image)
            require_value "$@"
            IMAGE="$2"
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

if [[ ! -f "${CONFIG}" ]]; then
    echo "DAQIRI config not found: ${CONFIG}" >&2
    exit 1
fi
if [[ -n "${DARK}" && ! -f "${DARK}" ]]; then
    echo "Dark-frame file not found: ${DARK}" >&2
    exit 1
fi
for value in "${RUN_DURATION_SECONDS}" "${TEGRASTATS_INTERVAL_MS}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Durations and intervals must be positive integers: ${value}" >&2
        exit 2
    fi
done
if ! command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats was not found; this script must run on the IGX host." >&2
    exit 1
fi
if pgrep -x tegrastats >/dev/null 2>&1; then
    echo "tegrastats is already running. Stop it with 'sudo tegrastats --stop' before profiling." >&2
    exit 1
fi

CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd -P)/$(basename "${CONFIG}")"
if [[ -n "${DARK}" ]]; then
    DARK="$(cd "$(dirname "${DARK}")" && pwd -P)/$(basename "${DARK}")"
fi
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd -P)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${RUN_DIR}" ]]; then
    RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}_$$"
fi
mkdir -p "${RUN_DIR}"

CONFIG_COPY="${RUN_DIR}/config.yaml"
METADATA_LOG="${RUN_DIR}/run_metadata.txt"
DAQIRI_LOG="${RUN_DIR}/daqiri.log"
TEGRASTATS_LOG="${RUN_DIR}/tegrastats.log"
TEGRASTATS_TIMELINE="${RUN_DIR}/tegrastats_timeline.tsv"
NVIDIA_SMI_LOG="${RUN_DIR}/nvidia_smi.csv"
NVIDIA_SMI_ERROR_LOG="${RUN_DIR}/nvidia_smi.stderr.log"
CONTAINER_NAME="stem-daqiri-profile-${RUN_ID,,}-$$"

cp "${CONFIG}" "${CONFIG_COPY}"
: >"${DAQIRI_LOG}"
: >"${TEGRASTATS_LOG}"
: >"${NVIDIA_SMI_ERROR_LOG}"

utc_now() {
    date -u +"%Y-%m-%dT%H:%M:%S.%NZ"
}

epoch_ns_now() {
    date +%s%N
}

timestamp_stream() {
    local line
    while IFS= read -r line; do
        printf '%s\t%s\n' "$(utc_now)" "${line}"
    done
}

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
fi

TEGRASTATS_PID=""
NVIDIA_SMI_PID=""
DAQIRI_STARTED=false

stop_tegrastats() {
    if [[ -n "${TEGRASTATS_PID}" ]]; then
        # tegrastats may outlive the sudo wrapper, so use its supported stop command.
        sudo tegrastats --stop >/dev/null 2>&1 || true
        wait "${TEGRASTATS_PID}" 2>/dev/null || true
        TEGRASTATS_PID=""
    fi
}

stop_nvidia_smi() {
    if [[ -n "${NVIDIA_SMI_PID}" ]]; then
        kill -INT "${NVIDIA_SMI_PID}" >/dev/null 2>&1 || true
        wait "${NVIDIA_SMI_PID}" 2>/dev/null || true
        NVIDIA_SMI_PID=""
    fi
}

cleanup() {
    local status=$?
    if [[ "${DAQIRI_STARTED}" == true ]] &&
       "${DOCKER[@]}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        "${DOCKER[@]}" stop --time 2 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
    stop_nvidia_smi
    stop_tegrastats
    return "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sudo -v

CONFIG_SHA256="$(sha256sum "${CONFIG_COPY}" | awk '{print $1}')"
DARK_SHA256=""
if [[ -n "${DARK}" ]]; then
    DARK_SHA256="$(sha256sum "${DARK}" | awk '{print $1}')"
fi
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"

{
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "image=${IMAGE}"
    echo "config_source=${CONFIG}"
    echo "config_sha256=${CONFIG_SHA256}"
    echo "dark_source=${DARK}"
    echo "dark_sha256=${DARK_SHA256}"
    echo "git_commit=${GIT_COMMIT}"
    echo "requested_duration_seconds=${RUN_DURATION_SECONDS}"
    echo "tegrastats_interval_ms=${TEGRASTATS_INTERVAL_MS}"
} >"${METADATA_LOG}"

NVIDIA_SMI_ENABLED=false
if command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi -L >/dev/null 2>&1; then
    NVIDIA_SMI_ENABLED=true
    printf '%s\n' \
        "timestamp,index,name,pstate,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c,clocks_sm_mhz,clocks_memory_mhz" \
        >"$NVIDIA_SMI_LOG"
else
    echo "nvidia_smi_enabled=false" >>"${METADATA_LOG}"
    echo "WARNING: nvidia-smi is unavailable; discrete-GPU metrics will not be recorded." >&2
fi

TEGRASTATS_START_UTC="$(utc_now)"
TEGRASTATS_START_EPOCH_NS="$(epoch_ns_now)"
{
    echo "tegrastats_start_utc=${TEGRASTATS_START_UTC}"
    echo "tegrastats_start_epoch_ns=${TEGRASTATS_START_EPOCH_NS}"
} >>"${METADATA_LOG}"

sudo tegrastats \
    --interval "${TEGRASTATS_INTERVAL_MS}" \
    --logfile "${TEGRASTATS_LOG}" &
TEGRASTATS_PID=$!

if [[ "${NVIDIA_SMI_ENABLED}" == true ]]; then
    NVIDIA_SMI_START_UTC="$(utc_now)"
    NVIDIA_SMI_START_EPOCH_NS="$(epoch_ns_now)"
    nvidia-smi \
        --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits \
        --loop-ms="${TEGRASTATS_INTERVAL_MS}" \
        >>"$NVIDIA_SMI_LOG" 2>"$NVIDIA_SMI_ERROR_LOG" &
    NVIDIA_SMI_PID=$!
    {
        echo "nvidia_smi_enabled=true"
        echo "nvidia_smi_start_utc=${NVIDIA_SMI_START_UTC}"
        echo "nvidia_smi_start_epoch_ns=${NVIDIA_SMI_START_EPOCH_NS}"
    } >>"${METADATA_LOG}"
fi

# Give tegrastats time to initialize and capture a pre-run baseline sample.
sleep 1
if ! sudo kill -0 "${TEGRASTATS_PID}" >/dev/null 2>&1; then
    echo "tegrastats exited before DAQIRI started." >&2
    exit 1
fi
if [[ "${NVIDIA_SMI_ENABLED}" == true ]] &&
   ! kill -0 "${NVIDIA_SMI_PID}" >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi sampling exited before DAQIRI started." >&2
    NVIDIA_SMI_ENABLED=false
    NVIDIA_SMI_PID=""
fi
echo "nvidia_smi_active_at_daqiri_start=${NVIDIA_SMI_ENABLED}" >>"${METADATA_LOG}"

MOUNTS=(
    -v /dev/hugepages:/dev/hugepages
    -v /tmp:/tmp
    -v "${CONFIG_COPY}:/run/stem_rx.yaml:ro"
)
if [[ -n "${DARK}" ]]; then
    MOUNTS+=(-v "${DARK}:/calibration/dark.h5:ro")
fi

DAQIRI_START_UTC="$(utc_now)"
DAQIRI_START_EPOCH_NS="$(epoch_ns_now)"
DAQIRI_START_OFFSET_NS=$((DAQIRI_START_EPOCH_NS - TEGRASTATS_START_EPOCH_NS))
{
    echo "daqiri_start_utc=${DAQIRI_START_UTC}"
    echo "daqiri_start_epoch_ns=${DAQIRI_START_EPOCH_NS}"
    echo "daqiri_start_after_tegrastats_ns=${DAQIRI_START_OFFSET_NS}"
} >>"${METADATA_LOG}"

echo "Starting synchronized DAQIRI platform profile:"
echo "  run directory : ${RUN_DIR}"
echo "  duration      : ${RUN_DURATION_SECONDS} s"
echo "  config        : ${CONFIG}"
if [[ -n "${DARK}" ]]; then
    echo "  dark frame    : ${DARK}"
fi
echo "  tegrastats    : every ${TEGRASTATS_INTERVAL_MS} ms"

DAQIRI_STARTED=true
set +e
"${DOCKER[@]}" run --rm --name "${CONTAINER_NAME}" \
    --privileged --network host --ipc=host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${MOUNTS[@]}" \
    "${IMAGE}" \
    bash -lc \
    'if command -v stdbuf >/dev/null 2>&1; then exec stdbuf -oL -eL "$@"; else exec "$@"; fi' \
    bash \
    /opt/stem_daqiri/bin/stem_daqiri_rx \
    /run/stem_rx.yaml \
    --seconds "${RUN_DURATION_SECONDS}" \
    2>&1 | timestamp_stream | tee "${DAQIRI_LOG}"
DAQIRI_EXIT_CODE=${PIPESTATUS[0]}
set -e
DAQIRI_STARTED=false

DAQIRI_END_UTC="$(utc_now)"
DAQIRI_END_EPOCH_NS="$(epoch_ns_now)"
stop_nvidia_smi
NVIDIA_SMI_END_UTC="$(utc_now)"
NVIDIA_SMI_END_EPOCH_NS="$(epoch_ns_now)"
stop_tegrastats
TEGRASTATS_END_UTC="$(utc_now)"
TEGRASTATS_END_EPOCH_NS="$(epoch_ns_now)"

{
    echo "daqiri_end_utc=${DAQIRI_END_UTC}"
    echo "daqiri_end_epoch_ns=${DAQIRI_END_EPOCH_NS}"
    echo "daqiri_exit_code=${DAQIRI_EXIT_CODE}"
    echo "nvidia_smi_end_utc=${NVIDIA_SMI_END_UTC}"
    echo "nvidia_smi_end_epoch_ns=${NVIDIA_SMI_END_EPOCH_NS}"
    echo "tegrastats_end_utc=${TEGRASTATS_END_UTC}"
    echo "tegrastats_end_epoch_ns=${TEGRASTATS_END_EPOCH_NS}"
} >>"${METADATA_LOG}"

awk -v interval_ms="${TEGRASTATS_INTERVAL_MS}" '
    BEGIN {
        print "sample_index\tnominal_elapsed_seconds\ttegrastats"
    }
    {
        printf "%d\t%.3f\t%s\n", NR, NR * interval_ms / 1000.0, $0
    }
' "${TEGRASTATS_LOG}" >"${TEGRASTATS_TIMELINE}"

echo "DAQIRI exit code: ${DAQIRI_EXIT_CODE}"
echo "Profile artifacts written to ${RUN_DIR}"
exit "${DAQIRI_EXIT_CODE}"
