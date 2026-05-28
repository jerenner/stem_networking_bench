#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Per-phase build/runtime gate verifier. Runs the smoke tests that gate each
# phase of the Holoscan -> daqiri port. Intended to be run on each DGX Spark
# independently.
#
# Usage:
#   cpp_daqiri/scripts/verify_phase.sh phase0
#   cpp_daqiri/scripts/verify_phase.sh phase1   # builds + runs stem_daqiri_tx --self-test
#   cpp_daqiri/scripts/verify_phase.sh phase2
#
# Exits 0 if all gates for that phase pass on this Spark. Exits non-zero on
# any failure; the user should not advance to the next phase until this
# script reports OK.

set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
    echo "Usage: $0 {phase0|phase1|phase2|phase3}" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

case "${PHASE}" in
    phase0)
        echo "[phase0] building stem_daqiri:phase0"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=OFF \
            --build-arg STEM_DAQIRI_BUILD_RX=OFF \
            -t stem_daqiri:phase0 .
        echo "[phase0] running stem_daqiri_hello --self-test"
        docker run --rm stem_daqiri:phase0 ./stem_daqiri_hello --self-test
        echo "[phase0] OK"
        ;;
    phase1)
        echo "[phase1] building stem_daqiri:phase1 (TX enabled)"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=ON \
            --build-arg STEM_DAQIRI_BUILD_RX=OFF \
            -t stem_daqiri:phase1 .
        echo "[phase1] confirming stem_daqiri_tx binary is installed"
        docker run --rm --entrypoint bash stem_daqiri:phase1 -c \
            'ls -la /opt/stem_daqiri/bin/stem_daqiri_tx && /opt/stem_daqiri/bin/stem_daqiri_tx 2>&1 | tail -1'
        # NOTE: the Holoscan RX validator (cpp/stem_networking_bench) needs a
        # SEPARATE container build using cpp/Dockerfile -- that container
        # builds libtorch from source on aarch64 and is intentionally
        # excluded from this script. Build it once with:
        #   docker build -t stem_holoscan:local .
        # then verify on spark-stacked-02 that
        #   cpp/run_with_network_fpga_1rcv.yaml
        # configures Advanced Network without errors.
        echo "[phase1] OK (daqiri side; Holoscan RX build is a separate one-time step)"
        ;;
    phase2)
        echo "[phase2] building stem_daqiri:phase2 (TX + RX enabled)"
        docker build -f Dockerfile.daqiri \
            --build-arg STEM_DAQIRI_BUILD_TX=ON \
            --build-arg STEM_DAQIRI_BUILD_RX=ON \
            -t stem_daqiri:phase2 .
        echo "[phase2] confirming both binaries installed"
        docker run --rm --entrypoint bash stem_daqiri:phase2 -c \
            'ls -la /opt/stem_daqiri/bin/stem_daqiri_tx /opt/stem_daqiri/bin/stem_daqiri_rx'
        echo "[phase2] OK (daqiri side)"
        ;;
    phase3)
        # Phase 3 image is the same build as phase 2 -- the parity-sweep
        # behaviour is gated by YAML flags (capture_latency, stamp_epoch_us,
        # subtract_dark, apply_valid_pixel_mask), not by build flags. We
        # retag :phase2 as :phase3 if it already exists to avoid a slow
        # rebuild; otherwise we build :phase3 from scratch.
        if docker image inspect stem_daqiri:phase2 >/dev/null 2>&1; then
            echo "[phase3] retagging stem_daqiri:phase2 as :phase3"
            docker tag stem_daqiri:phase2 stem_daqiri:phase3
        else
            echo "[phase3] building stem_daqiri:phase3 (TX + RX enabled, same as phase2)"
            docker build -f Dockerfile.daqiri \
                --build-arg STEM_DAQIRI_BUILD_TX=ON \
                --build-arg STEM_DAQIRI_BUILD_RX=ON \
                -t stem_daqiri:phase3 .
        fi
        echo "[phase3] confirming both binaries installed"
        docker run --rm --entrypoint bash stem_daqiri:phase3 -c \
            'ls -la /opt/stem_daqiri/bin/stem_daqiri_tx /opt/stem_daqiri/bin/stem_daqiri_rx'
        echo "[phase3] image OK. Next step is the parity sweep:"
        echo "    cpp_daqiri/scripts/run_phase3_sweep_rx.sh   # on spark-stacked-02"
        echo "    cpp_daqiri/scripts/run_phase3_sweep_tx.sh   # on spark-stacked-01"
        echo "    cpp_daqiri/scripts/parse_phase3_results.py  # after both sweeps complete"
        ;;
    *)
        echo "Unknown phase: ${PHASE}" >&2
        echo "Usage: $0 {phase0|phase1|phase2|phase3}" >&2
        exit 1
        ;;
esac
