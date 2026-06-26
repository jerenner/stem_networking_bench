#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# RX side for the TX+RX Spark gate. Run on spark-stacked-02 (spark-201a,
# 169.254.95.47). Starts stem_daqiri_rx in the stem_daqiri:tx-rx container.
#
# Args:
#   --seconds N  override total_time_to_recv
#   --image T    container image (default stem_daqiri:tx-rx)
#   --config P   YAML path inside the container

set -euo pipefail

SECONDS_ARG=""
IMAGE="stem_daqiri:tx-rx"
CONFIG="/opt/stem_daqiri/bin/configs/stem_rx_spark.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seconds) SECONDS_ARG="--seconds $2"; shift 2 ;;
        --image)   IMAGE="$2";                  shift 2 ;;
        --config)  CONFIG="$2";                  shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

HP_TOTAL=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)
HP_SIZE_KB=$(awk '/Hugepagesize/ {print $2}' /proc/meminfo)
HP_TOTAL_MB=$(( HP_TOTAL * HP_SIZE_KB / 1024 ))
if [[ "${HP_TOTAL_MB}" -lt 512 ]]; then
    echo "WARNING: only ${HP_TOTAL_MB} MiB hugepages allocated; DPDK needs >= 1 GiB."
    echo "         sudo sysctl -w vm.nr_hugepages=512"
fi

set -x
docker run --rm -it \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    "${IMAGE}" \
    /opt/stem_daqiri/bin/stem_daqiri_rx "${CONFIG}" ${SECONDS_ARG}
