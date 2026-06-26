#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# docker exec helper for long-running DAQIRI containers.
#
# NVIDIA's container entrypoint prepares CUDA forward-compat libraries at
# container startup, but `docker exec` sessions do not inherit the adjusted
# LD_LIBRARY_PATH. This wrapper injects the compat lib directory before
# running the requested command.

set -euo pipefail

CONTAINER="${DAQIRI_CONTAINER:-stem_daqiri_live}"
CUDA_COMPAT_LIB="${DAQIRI_CUDA_COMPAT_LIB:-/usr/local/cuda/compat/lib}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        --cuda-compat-lib)
            CUDA_COMPAT_LIB="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage:
  $0 [--container NAME] [--cuda-compat-lib PATH] [COMMAND...]

Defaults:
  container       ${CONTAINER}
  cuda compat lib ${CUDA_COMPAT_LIB}

If COMMAND is omitted, opens an interactive shell with the compat path set.
EOF
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    exec docker exec -it "${CONTAINER}" bash -lc \
        "export LD_LIBRARY_PATH=${CUDA_COMPAT_LIB}:\${LD_LIBRARY_PATH:-}; exec bash"
fi

exec docker exec "${CONTAINER}" bash -lc \
    'export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}"; shift; exec "$@"' \
    bash "${CUDA_COMPAT_LIB}" "$@"
