#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Spark parity/throughput sweep orchestrator. Drives BOTH the RX (over SSH to
# spark-stacked-02) and the TX (locally on spark-960b) from a single
# process so they stay synchronized per rate. Run from spark-960b.
#
# Each rate iteration:
#   1. Launch RX detached on 201a with a generous --seconds budget.
#   2. Poll the RX log until DPDK has finished init and the RX worker
#      thread is polling the queue.
#   3. Run TX in the foreground (blocks until done).
#   4. Wait for the RX container to exit (it will hit its --seconds
#      budget shortly after TX stops sending).
#   5. Copy/keep RX log under the same OUTDIR as the TX logs.
#
# This avoids the prior shape where the two side scripts ran on uncoupled wall
# clocks and drifted apart after the first slow docker invocation.
#
# Args:
#   --rates "10 25 50 80"   space-separated target rates in Gbps
#   --runs N                iterations per rate (default 1)
#   --seconds N             TX seconds per rate (default 8). RX uses SECS+15.
#   --outdir D              output directory (default sweep_<utc>)
#   --image-daqiri T        container image for daqiri TX/RX (default stem_daqiri:tx-rx)
#   --rx-host H             SSH host for the RX side (default spark-stacked-02)

set -euo pipefail

RATES="10 25 50 80"
RUNS=1
SECS=8
RX_HOST="spark-stacked-02"
IMAGE_DAQIRI="stem_daqiri:tx-rx"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rates)         RATES="$2";         shift 2 ;;
        --runs)          RUNS="$2";          shift 2 ;;
        --seconds)       SECS="$2";          shift 2 ;;
        --outdir)        OUTDIR="$2";        shift 2 ;;
        --image-daqiri)  IMAGE_DAQIRI="$2";  shift 2 ;;
        --rx-host)       RX_HOST="$2";       shift 2 ;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OUTDIR:=${REPO_ROOT}/cpp_daqiri/benchmarks/sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUTDIR}"
echo "[orch] outdir=${OUTDIR}"
echo "[orch] rates='${RATES}' runs=${RUNS} secs=${SECS} rx_host=${RX_HOST}"

RX_SECS=$(( SECS + 15 ))

run_one_rate() {
    local rate="$1"
    local run="$2"
    local tag="${rate}gbps_run${run}"
    local rx_log="${OUTDIR}/daqiri_rx_${tag}.log"
    local tx_log="${OUTDIR}/daqiri_tx_${tag}.log"

    echo "[orch] ===== rate=${rate} run=${run}/${RUNS} ====="

    # Truncate the RX log so the readiness-poll only sees this iteration.
    : > "${rx_log}"

    # 1. Launch the RX container on the peer Spark, log to the shared NFS path.
    #    Note we deliberately do NOT pass --rm: we want the container to stick
    #    around after exit so we can pull its full stdout before cleanup.
    ssh -o BatchMode=yes "${RX_HOST}" "
      docker ps -aq --filter name=stem_sweep_rx | xargs -r docker rm -f >/dev/null 2>&1
      docker run -d --name stem_sweep_rx \
          --privileged --network host \
          --gpus all \
          --ulimit memlock=-1 --ulimit stack=67108864 \
          -v /dev/hugepages:/dev/hugepages \
          -v ${REPO_ROOT}/cpp_daqiri/configs:/cfgs \
          ${IMAGE_DAQIRI} \
          /opt/stem_daqiri/bin/stem_daqiri_rx /cfgs/stem_rx_spark.yaml \
              --seconds ${RX_SECS} \
          >/dev/null 2>&1
    "

    # 2. Poll the rx container's stdout until DPDK has started the RX worker.
    #    The line we wait for is 'Starting RX Core' which is emitted right
    #    before the poll loop starts.
    local waited=0
    while (( waited < 60 )); do
        ssh -o BatchMode=yes "${RX_HOST}" 'docker logs stem_sweep_rx 2>&1' > "${rx_log}" 2>/dev/null || true
        if grep -q 'Starting RX Core' "${rx_log}"; then
            break
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done
    if (( waited >= 60 )); then
        echo "[orch] WARN: RX never reached polling state at rate ${rate}" >&2
    else
        echo "[orch] RX ready after ${waited}s init; firing TX"
    fi
    # small buffer so the worker thread has at least one full poll period in hand
    sleep 1

    # 3. Run TX locally, blocking.
    docker run --rm \
        --privileged --network host \
        --gpus all \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v /dev/hugepages:/dev/hugepages \
        -v "${REPO_ROOT}/cpp_daqiri/configs:/cfgs" \
        "${IMAGE_DAQIRI}" \
        /opt/stem_daqiri/bin/stem_daqiri_tx \
            /cfgs/stem_tx_spark.yaml \
            --seconds "${SECS}" --rate "${rate}" \
        > "${tx_log}" 2>&1 || true
    echo "[orch] TX done (rate=${rate})"

    # 4. Let the RX drain, then wait for the container to exit and dump its log.
    sleep 3
    local waited2=0
    while (( waited2 < 30 )); do
        if ! ssh -o BatchMode=yes "${RX_HOST}" 'docker ps -q --filter name=stem_sweep_rx' 2>/dev/null | grep -q .; then
            break
        fi
        sleep 1
        waited2=$(( waited2 + 1 ))
    done
    ssh -o BatchMode=yes "${RX_HOST}" 'docker logs stem_sweep_rx 2>&1 || true' > "${rx_log}" 2>/dev/null
    # Now safe to remove the container; the log has been captured.
    ssh -o BatchMode=yes "${RX_HOST}" 'docker rm -f stem_sweep_rx >/dev/null 2>&1 || true'
}

for rate in ${RATES}; do
    for run in $(seq 1 "${RUNS}"); do
        run_one_rate "${rate}" "${run}"
    done
done

echo "[orch] ===== sweep done. logs in ${OUTDIR} ====="
