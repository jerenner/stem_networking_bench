#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Spark parity/throughput sweep -- RX side. Run on spark-stacked-02 (spark-201a).
# Loops over target rates (must match the TX sweep) and starts a new RX
# instance for each iteration. Each RX run lasts a couple seconds longer
# than the matched TX run so it can drain after the TX stops sending.
#
# Args (same shape as run_spark_parity_sweep_tx.sh):
#   --rates "50 80 95"
#   --runs N
#   --seconds N       TX seconds per run; RX runs for SECS + 4
#   --image T
#   --outdir D
#   --label L
#   --binary daqiri|holoscan
#     daqiri   -> uses stem_daqiri:tx-rx + stem_daqiri_rx
#     holoscan -> uses stem_holoscan:local + cpp/stem_networking_bench
#                  with cpp/run_with_network_fpga_1rcv.yaml

set -euo pipefail

RATES="50 80 95"
RUNS=3
SECS=10
IMAGE_DAQIRI="stem_daqiri:tx-rx"
IMAGE_HOLO="stem_holoscan:local"
LABEL="daqiri_rx"
BINARY="daqiri"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rates)    RATES="$2";   shift 2 ;;
        --runs)     RUNS="$2";    shift 2 ;;
        --seconds)  SECS="$2";    shift 2 ;;
        --image)    IMAGE_DAQIRI="$2"; shift 2 ;;
        --outdir)   OUTDIR="$2";  shift 2 ;;
        --label)    LABEL="$2";   shift 2 ;;
        --binary)   BINARY="$2";  shift 2 ;;
        -h|--help)  sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OUTDIR:=${REPO_ROOT}/cpp_daqiri/benchmarks/logs_rx_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUTDIR}"
echo "writing RX sweep logs to ${OUTDIR} (binary=${BINARY})"

RX_SECS=$(( SECS + 4 ))

for rate in ${RATES}; do
    for run in $(seq 1 "${RUNS}"); do
        LOG="${OUTDIR}/${LABEL}_${rate}gbps_run${run}.log"
        echo "===== ${LABEL}: rate=${rate} Gbps run=${run}/${RUNS} (logging to ${LOG}) ====="
        if [[ "${BINARY}" == "holoscan" ]]; then
            docker run --rm \
                --privileged --network host \
                --gpus all \
                --ulimit memlock=-1 --ulimit stack=67108864 \
                -v /dev/hugepages:/dev/hugepages \
                -v "${REPO_ROOT}:/workspace/stem_holoscan" \
                -w /workspace/stem_holoscan \
                "${IMAGE_HOLO}" \
                bash -c 'cd cpp && [ -d build ] || cmake -S . -B build && cmake --build build -j && timeout '"${RX_SECS}"' ./build/stem_networking_bench run_with_network_fpga_1rcv.yaml' \
                > "${LOG}" 2>&1 || true
        else
            # YAML is mounted from the shared NFS workspace so tweaks (NIC
            # PCI, mempool size, frames_per_tensor) don't require rebuilds.
            docker run --rm \
                --privileged --network host \
                --gpus all \
                --ulimit memlock=-1 --ulimit stack=67108864 \
                -v /dev/hugepages:/dev/hugepages \
                -v "${REPO_ROOT}/cpp_daqiri/configs:/cfgs" \
                "${IMAGE_DAQIRI}" \
                /opt/stem_daqiri/bin/stem_daqiri_rx \
                    /cfgs/stem_rx_spark.yaml \
                    --seconds "${RX_SECS}" \
                > "${LOG}" 2>&1 || true
        fi
        sleep 2
    done
done

echo "===== ${LABEL} sweep done. Logs: ${OUTDIR} ====="
