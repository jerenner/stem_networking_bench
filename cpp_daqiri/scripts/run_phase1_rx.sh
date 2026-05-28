#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Phase 1 runtime gate -- RX side. Run on spark-stacked-02 (spark-201a,
# 169.254.95.47). Starts the existing Holoscan stem_networking_bench
# binary against cpp/run_with_network_fpga_1rcv.yaml so it acts as the
# STEM frame assembler that validates the daqiri TX.
#
# This script assumes the Holoscan container image has already been built
# once via the repo's primary Dockerfile (the one that builds Holoscan SDK
# + libtorch + DPDK + HDF5). Build it with:
#
#     cd /srv/nfs/share/users/ccrozier/stem_networking_bench
#     docker build -t stem_holoscan:local .
#
# That image takes ~1-2 hours on aarch64 (PyTorch from source). Once built
# it is cached and you only rebuild when cpp/* changes.
#
# Args:
#   --image T  container image tag (default stem_holoscan:local)
#   --config P config path inside the container

set -euo pipefail

IMAGE="stem_holoscan:local"
CONFIG="cpp/run_with_network_fpga_1rcv.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)  IMAGE="$2";  shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Pre-flight: hugepages
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
    -v "${REPO_ROOT}:/workspace/stem_holoscan" \
    -w /workspace/stem_holoscan \
    "${IMAGE}" \
    bash -c 'cd cpp && [ -d build ] || cmake -S . -B build && cmake --build build -j && ./build/stem_networking_bench '"${CONFIG##cpp/}"
