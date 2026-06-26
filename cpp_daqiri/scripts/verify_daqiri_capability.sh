#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Build/runtime gate verifier. Runs smoke tests for the hello/link-check,
# TX-only, TX+RX, and parity-HDF5 images. Intended to be run on each DGX Spark
# independently.
#
# Usage:
#   cpp_daqiri/scripts/verify_daqiri_capability.sh hello
#   cpp_daqiri/scripts/verify_daqiri_capability.sh tx-only
#   cpp_daqiri/scripts/verify_daqiri_capability.sh tx-rx
#   cpp_daqiri/scripts/verify_daqiri_capability.sh parity-hdf5
#
# Exits 0 if all checks for the selected capability pass on this Spark.
# Exits non-zero on any failure.

set -euo pipefail

CAPABILITY="${1:-}"
if [[ -z "${CAPABILITY}" ]]; then
    echo "Usage: $0 {hello|tx-only|tx-rx|parity-hdf5}" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

case "${CAPABILITY}" in
    hello)
        echo "[hello] building stem_daqiri:hello"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=OFF \
            --build-arg STEM_DAQIRI_BUILD_RX=OFF \
            -t stem_daqiri:hello .
        echo "[hello] running stem_daqiri_hello --self-test"
        docker run --rm stem_daqiri:hello ./stem_daqiri_hello --self-test
        echo "[hello] OK"
        ;;
    tx-only)
        echo "[tx-only] building stem_daqiri:tx-only"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=ON \
            --build-arg STEM_DAQIRI_BUILD_RX=OFF \
            -t stem_daqiri:tx-only .
        echo "[tx-only] confirming stem_daqiri_tx binary is installed"
        docker run --rm --entrypoint bash stem_daqiri:tx-only -c \
            'ls -la /opt/stem_daqiri/bin/stem_daqiri_tx && /opt/stem_daqiri/bin/stem_daqiri_tx 2>&1 | tail -1'
        # NOTE: the Holoscan RX validator (cpp/stem_networking_bench) needs a
        # SEPARATE container build using cpp/Dockerfile -- that container
        # builds libtorch from source on aarch64 and is intentionally
        # excluded from this script. Build it once with:
        #   docker build -t stem_holoscan:local .
        # then verify on spark-stacked-02 that
        #   cpp/run_with_network_fpga_1rcv.yaml
        # configures Advanced Network without errors.
        echo "[tx-only] OK (daqiri side; Holoscan RX build is a separate one-time step)"
        ;;
    tx-rx)
        echo "[tx-rx] building stem_daqiri:tx-rx"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=ON \
            --build-arg STEM_DAQIRI_BUILD_RX=ON \
            -t stem_daqiri:tx-rx .
        echo "[tx-rx] confirming both binaries installed"
        docker run --rm --entrypoint bash stem_daqiri:tx-rx -c \
            'ls -la /opt/stem_daqiri/bin/stem_daqiri_tx /opt/stem_daqiri/bin/stem_daqiri_rx'
        echo "[tx-rx] OK (daqiri side)"
        ;;
    parity-hdf5)
        echo "[parity-hdf5] building stem_daqiri:parity-hdf5"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=ON \
            --build-arg STEM_DAQIRI_BUILD_RX=ON \
            --build-arg STEM_DAQIRI_REQUIRE_HDF5=ON \
            -t stem_daqiri:parity-hdf5 .
        echo "[parity-hdf5] confirming binaries and HDF5-capable runtime are installed"
        docker run --rm --entrypoint bash stem_daqiri:parity-hdf5 -c \
            'ls -la /opt/stem_daqiri/bin/stem_daqiri_tx /opt/stem_daqiri/bin/stem_daqiri_rx'
        echo "[parity-hdf5] image OK. Next steps:"
        echo "    cpp_daqiri/scripts/run_spark_parity_sweep_rx.sh   # on spark-stacked-02"
        echo "    cpp_daqiri/scripts/run_spark_parity_sweep_tx.sh   # on spark-stacked-01"
        echo "    cpp_daqiri/scripts/parse_spark_parity_results.py  # after both sweeps complete"
        ;;
    *)
        echo "Unknown capability: ${CAPABILITY}" >&2
        echo "Usage: $0 {hello|tx-only|tx-rx|parity-hdf5}" >&2
        exit 1
        ;;
esac
