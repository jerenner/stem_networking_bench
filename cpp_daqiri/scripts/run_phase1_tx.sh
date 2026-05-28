#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Phase 1 runtime gate -- TX side. Run on spark-stacked-01 (spark-960b,
# 169.254.100.253). Starts stem_daqiri_tx in the stem_daqiri:phase1
# container against cpp_daqiri/configs/stem_tx_spark.yaml.
#
# Args (all optional):
#   --seconds N   override total_time_to_send_s (default from YAML)
#   --rate    G   override target_rate_gbps   (default from YAML)
#   --image   T   override container image    (default stem_daqiri:phase1)
#   --config  P   override config path inside the container
#
# Coordinated with cpp_daqiri/scripts/run_phase1_rx.sh which must already
# be running on spark-stacked-02.

set -euo pipefail

SECONDS_ARG=""
RATE_ARG=""
IMAGE="stem_daqiri:phase1"
CONFIG="/opt/stem_daqiri/bin/configs/stem_tx_spark.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seconds) SECONDS_ARG="--seconds $2"; shift 2 ;;
        --rate)    RATE_ARG="--rate $2";       shift 2 ;;
        --image)   IMAGE="$2";                  shift 2 ;;
        --config)  CONFIG="$2";                  shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

# Pre-flight: hugepages must be allocated (DPDK refuses to start otherwise).
HP_TOTAL=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)
HP_SIZE_KB=$(awk '/Hugepagesize/ {print $2}' /proc/meminfo)
HP_TOTAL_MB=$(( HP_TOTAL * HP_SIZE_KB / 1024 ))
if [[ "${HP_TOTAL_MB}" -lt 512 ]]; then
    echo "WARNING: only ${HP_TOTAL_MB} MiB hugepages allocated; DPDK typically needs >= 1 GiB."
    echo "         allocate more with: sudo sysctl -w vm.nr_hugepages=512"
fi

set -x
docker run --rm -it \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    "${IMAGE}" \
    /opt/stem_daqiri/bin/stem_daqiri_tx "${CONFIG}" ${SECONDS_ARG} ${RATE_ARG}
