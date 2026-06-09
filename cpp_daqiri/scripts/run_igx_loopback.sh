#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# IGX Orin + RTX 6000 Ada hardware-loopback gate. Start RX first on
# 0005:03:00.1, then TX on 0005:03:00.0 over the direct cable.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_RX_CONFIG="${SCRIPT_DIR}/../configs/stem_rx_igx_loopback.yaml"
SOURCE_TX_CONFIG="${SCRIPT_DIR}/../configs/stem_tx_igx_loopback.yaml"

IMAGE="stem_daqiri:phase2"
RX_SECONDS="60"
TX_SECONDS="10"
RATE="20"
RX_CONFIG="/opt/stem_daqiri/bin/configs/stem_rx_igx_loopback.yaml"
TX_CONFIG="/opt/stem_daqiri/bin/configs/stem_tx_igx_loopback.yaml"
VALIDATE_RAMP=""
RX_CONTAINER="stem_igx_rx"

if [[ -f "${SOURCE_RX_CONFIG}" ]]; then
    RX_CONFIG="${SOURCE_RX_CONFIG}"
fi
if [[ -f "${SOURCE_TX_CONFIG}" ]]; then
    TX_CONFIG="${SOURCE_TX_CONFIG}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --rx-seconds) RX_SECONDS="$2"; shift 2 ;;
        --tx-seconds) TX_SECONDS="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        --rx-config) RX_CONFIG="$2"; shift 2 ;;
        --tx-config) TX_CONFIG="$2"; shift 2 ;;
        --validate-ramp) VALIDATE_RAMP="--validate-ramp"; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

RX_DOCKER_CONFIG="${RX_CONFIG}"
TX_DOCKER_CONFIG="${TX_CONFIG}"
RX_CONFIG_MOUNT=()
TX_CONFIG_MOUNT=()

if [[ -f "${RX_CONFIG}" ]]; then
    RX_CONFIG_HOST="$(realpath "${RX_CONFIG}")"
    RX_DOCKER_CONFIG="/tmp/stem_rx_config.yaml"
    RX_CONFIG_MOUNT=(-v "${RX_CONFIG_HOST}:${RX_DOCKER_CONFIG}:ro")
fi

if [[ -f "${TX_CONFIG}" ]]; then
    TX_CONFIG_HOST="$(realpath "${TX_CONFIG}")"
    TX_DOCKER_CONFIG="/tmp/stem_tx_config.yaml"
    TX_CONFIG_MOUNT=(-v "${TX_CONFIG_HOST}:${TX_DOCKER_CONFIG}:ro")
fi

HP_TOTAL=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)
HP_SIZE_KB=$(awk '/Hugepagesize/ {print $2}' /proc/meminfo)
HP_TOTAL_MB=$(( HP_TOTAL * HP_SIZE_KB / 1024 ))
if [[ "${HP_TOTAL_MB}" -lt 3072 ]]; then
    echo "WARNING: only ${HP_TOTAL_MB} MiB hugepages allocated; IGX loopback expects 3 GiB."
fi

docker rm -f "${RX_CONTAINER}" >/dev/null 2>&1 || true

docker run -d --name "${RX_CONTAINER}" \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    "${RX_CONFIG_MOUNT[@]}" \
    "${IMAGE}" \
    /opt/stem_daqiri/bin/stem_daqiri_rx \
    "${RX_DOCKER_CONFIG}" \
    --seconds "${RX_SECONDS}" \
    ${VALIDATE_RAMP}

for _ in $(seq 1 60); do
    if docker logs "${RX_CONTAINER}" 2>&1 | grep -q "Starting RX Core"; then
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${RX_CONTAINER}"; then
        docker logs "${RX_CONTAINER}" >&2 || true
        exit 1
    fi
    sleep 1
done

if ! docker logs "${RX_CONTAINER}" 2>&1 | grep -q "Starting RX Core"; then
    docker logs "${RX_CONTAINER}" >&2 || true
    echo "RX did not report readiness within 60 seconds" >&2
    exit 1
fi

docker run --rm \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    "${TX_CONFIG_MOUNT[@]}" \
    "${IMAGE}" \
    /opt/stem_daqiri/bin/stem_daqiri_tx \
    "${TX_DOCKER_CONFIG}" \
    --seconds "${TX_SECONDS}" \
    --rate "${RATE}"

docker logs -f "${RX_CONTAINER}"
docker rm "${RX_CONTAINER}" >/dev/null
