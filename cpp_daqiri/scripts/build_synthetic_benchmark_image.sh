#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Build the portable x86_64 DAQIRI image used by the synthetic GPU benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DAQIRI_BASE_IMAGE="${DAQIRI_BASE_IMAGE:-daqiri-torch:cuda13.0}"
STEM_IMAGE="${STEM_DAQIRI_IMAGE:-stem_daqiri:synthetic-amd64}"
DAQIRI_TORCH_IMAGE="${DAQIRI_TORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
CUDA_ARCHITECTURES="${STEM_CUDA_ARCHITECTURES:-100;120}"
REBUILD_DAQIRI_BASE="${REBUILD_DAQIRI_BASE:-false}"

usage() {
    cat <<EOF
Usage:
  $0 [--image NAME] [--base-image NAME] [--torch-image NAME]
     [--cuda-architectures LIST] [--rebuild-base]

Build an x86_64 DAQIRI + STEM RX image suitable for both the local RTX smoke
test and a one-GPU B200 cloud benchmark.

Options:
  --image NAME       final image tag (default: ${STEM_IMAGE})
  --base-image NAME  DAQIRI base image tag (default: ${DAQIRI_BASE_IMAGE})
  --torch-image NAME NGC PyTorch base (default: ${DAQIRI_TORCH_IMAGE})
  --cuda-architectures LIST
                     CMake CUDA targets (default: ${CUDA_ARCHITECTURES})
  --rebuild-base     rebuild the DAQIRI base even if its tag already exists
  -h, --help         show this help

Environment equivalents:
  STEM_DAQIRI_IMAGE, DAQIRI_BASE_IMAGE, DAQIRI_TORCH_IMAGE,
  STEM_CUDA_ARCHITECTURES, REBUILD_DAQIRI_BASE=true
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
        --image)
            require_value "$@"
            STEM_IMAGE="$2"
            shift 2
            ;;
        --base-image)
            require_value "$@"
            DAQIRI_BASE_IMAGE="$2"
            shift 2
            ;;
        --torch-image)
            require_value "$@"
            DAQIRI_TORCH_IMAGE="$2"
            shift 2
            ;;
        --cuda-architectures)
            require_value "$@"
            CUDA_ARCHITECTURES="$2"
            shift 2
            ;;
        --rebuild-base)
            REBUILD_DAQIRI_BASE=true
            shift
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

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This benchmark image must be built natively on x86_64, not $(uname -m)." >&2
    exit 1
fi
for command in docker git patch; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Required command not found: ${command}" >&2
        exit 1
    fi
done
if ! docker info >/dev/null 2>&1; then
    echo "Docker is not available to the current user." >&2
    exit 1
fi

cd "${REPO_ROOT}"
git submodule update --init --recursive third_party/daqiri

if [[ "${REBUILD_DAQIRI_BASE}" == "true" ]] || \
        ! docker image inspect "${DAQIRI_BASE_IMAGE}" >/dev/null 2>&1; then
    echo "Building DAQIRI base image ${DAQIRI_BASE_IMAGE}..."
    DAQIRI_BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/stem-daqiri-base.XXXXXX")"
    cleanup_build_context() {
        rm -rf "${DAQIRI_BUILD_CONTEXT}"
    }
    trap cleanup_build_context EXIT
    cp -a third_party/daqiri/. "${DAQIRI_BUILD_CONTEXT}/"
    patch --batch --forward -d "${DAQIRI_BUILD_CONTEXT}" -p1 \
        < cpp_daqiri/patches/daqiri-container-disable-examples.patch
    (
        cd "${DAQIRI_BUILD_CONTEXT}"
        IMAGE_TAG="${DAQIRI_BASE_IMAGE}" \
        BASE_IMAGE=torch \
        BASE_TARGET=dpdk \
        DAQIRI_ENGINE=dpdk \
        DAQIRI_BUILD_EXAMPLES=OFF \
        DAQIRI_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        DAQIRI_OS_BASE_IMAGE="${DAQIRI_TORCH_IMAGE}" \
        scripts/build-container.sh
    )
    cleanup_build_context
    trap - EXIT
else
    echo "Reusing existing DAQIRI base image ${DAQIRI_BASE_IMAGE}."
fi

echo "Building STEM synthetic benchmark image ${STEM_IMAGE}..."
docker build \
    -f Dockerfile.daqiri \
    --build-arg "DAQIRI_BASE=${DAQIRI_BASE_IMAGE}" \
    --build-arg "STEM_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    --build-arg STEM_DAQIRI_BUILD_TX=OFF \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    --build-arg STEM_DAQIRI_REQUIRE_HDF5=ON \
    --build-arg STEM_DAQIRI_REQUIRE_ZMQ=ON \
    -t "${STEM_IMAGE}" \
    .

echo "Build complete:"
docker image inspect \
    --format '  tag={{index .RepoTags 0}} id={{.Id}} size={{.Size}} architecture={{.Architecture}}' \
    "${STEM_IMAGE}"
