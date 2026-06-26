#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Spark parity/throughput sweep -- TX side. Run on spark-stacked-01 (spark-960b).
# Loops over target rates and runs N iterations of stem_daqiri_tx at each
# rate. Captures stdout into per-rate log files under cpp_daqiri/benchmarks/.
#
# Coordinated with run_spark_parity_sweep_rx.sh: start the RX sweep on
# spark-stacked-02 FIRST so it is ready to capture each TX run.
#
# Args:
#   --rates "50 80 95"   space-separated target rates in Gbps
#   --runs N             iterations per rate (default 3)
#   --seconds N          seconds per run (default 10)
#   --image T            container image (default stem_daqiri:tx-rx)
#   --outdir D           output directory (default cpp_daqiri/benchmarks/logs_tx_<utc>)
#   --label L            extra label baked into the log filenames

set -euo pipefail

RATES_DEFAULT="50 80 95"
RUNS=3
SECS=10
IMAGE="stem_daqiri:tx-rx"
LABEL="daqiri_tx"
RATES="${RATES_DEFAULT}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rates)   RATES="$2";   shift 2 ;;
        --runs)    RUNS="$2";    shift 2 ;;
        --seconds) SECS="$2";    shift 2 ;;
        --image)   IMAGE="$2";   shift 2 ;;
        --outdir)  OUTDIR="$2";  shift 2 ;;
        --label)   LABEL="$2";   shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OUTDIR:=${REPO_ROOT}/cpp_daqiri/benchmarks/logs_tx_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUTDIR}"
echo "writing TX sweep logs to ${OUTDIR}"

for rate in ${RATES}; do
    for run in $(seq 1 "${RUNS}"); do
        LOG="${OUTDIR}/${LABEL}_${rate}gbps_run${run}.log"
        echo "===== ${LABEL}: rate=${rate} Gbps run=${run}/${RUNS} (logging to ${LOG}) ====="
        # Wait a few seconds so the RX side has time to come up (especially
        # for the very first run).
        sleep 2
        # The YAML lives on the shared NFS workspace -- mount it in so YAML
        # tweaks (NIC PCI address, MACs, mempool size) don't require an
        # image rebuild between sweeps.
        docker run --rm \
            --privileged --network host \
            --gpus all \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v /dev/hugepages:/dev/hugepages \
            -v "${REPO_ROOT}/cpp_daqiri/configs:/cfgs" \
            "${IMAGE}" \
            /opt/stem_daqiri/bin/stem_daqiri_tx \
                /cfgs/stem_tx_spark.yaml \
                --seconds "${SECS}" --rate "${rate}" \
            > "${LOG}" 2>&1 || true
        # Cool-down between runs so the next RX iteration sees an idle line.
        sleep 3
    done
done

echo "===== ${LABEL} sweep done. Logs: ${OUTDIR} ====="
