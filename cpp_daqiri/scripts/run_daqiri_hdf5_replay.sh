#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

set -euo pipefail

IMAGE="${DAQIRI_IMAGE:-stem_daqiri:dual-fpga}"
INPUT=""
DARK=""
OUTPUT="${PWD}/stem_replay_out.h5"
INPUT_DATASET="/frames"
DARK_DATASET="/processed"
MASK_DATASET="/valid_pixel_mask"
START_FRAME=0
COUNT=0
FRAMES_PER_TENSOR=128
USE_MASK=true
USE_BLR=true
USE_DYNAMIC_MASK=true
REDUCE=false

usage() {
    cat <<EOF
Usage:
  $0 --input INPUT.h5 [options]

Required:
  --input PATH                 uint16/float32 HDF5 input with [frames,H,W]

Processing and output:
  --dark PATH                  matching dark-frame HDF5; enables full processing
  --output PATH                output HDF5 (default: ${OUTPUT})
  --input-dataset PATH         input dataset (default: ${INPUT_DATASET})
  --dark-dataset PATH          dark dataset (default: ${DARK_DATASET})
  --mask-dataset PATH          valid-pixel mask dataset (default: ${MASK_DATASET})
  --no-valid-mask              do not apply the static valid-pixel mask
  --no-blr                     do not apply grouped BLR correction
  --no-dynamic-mask            do not apply dynamic two-sided masking
  --reduce                     sum each input tensor to one output frame

Replay selection:
  --start-frame N              first input frame (default: ${START_FRAME})
  --count N                    frames to replay; 0 means all remaining (default: ${COUNT})
  --frames-per-tensor N        replay batch size (default: ${FRAMES_PER_TENSOR})
  --image NAME                 container image (default: ${IMAGE})

With --dark, the default path is the same full processing chain used for live
DAQIRI input: dark subtraction, valid-pixel masking, grouped BLR correction,
and two-sided dynamic half-column masking. Without --dark, processing defaults
to a raw pass-through unless the generated config is edited.
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
        --input)
            require_value "$@"
            INPUT="$2"
            shift 2
            ;;
        --dark)
            require_value "$@"
            DARK="$2"
            shift 2
            ;;
        --output)
            require_value "$@"
            OUTPUT="$2"
            shift 2
            ;;
        --input-dataset)
            require_value "$@"
            INPUT_DATASET="$2"
            shift 2
            ;;
        --dark-dataset)
            require_value "$@"
            DARK_DATASET="$2"
            shift 2
            ;;
        --mask-dataset)
            require_value "$@"
            MASK_DATASET="$2"
            shift 2
            ;;
        --start-frame)
            require_value "$@"
            START_FRAME="$2"
            shift 2
            ;;
        --count)
            require_value "$@"
            COUNT="$2"
            shift 2
            ;;
        --frames-per-tensor)
            require_value "$@"
            FRAMES_PER_TENSOR="$2"
            shift 2
            ;;
        --image)
            require_value "$@"
            IMAGE="$2"
            shift 2
            ;;
        --no-valid-mask)
            USE_MASK=false
            shift
            ;;
        --no-blr)
            USE_BLR=false
            shift
            ;;
        --no-dynamic-mask)
            USE_DYNAMIC_MASK=false
            shift
            ;;
        --reduce)
            REDUCE=true
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

if [[ -z "${INPUT}" ]]; then
    echo "--input is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "${INPUT}" ]]; then
    echo "Input file not found: ${INPUT}" >&2
    exit 1
fi
if [[ -n "${DARK}" && ! -f "${DARK}" ]]; then
    echo "Dark-frame file not found: ${DARK}" >&2
    exit 1
fi
for value in "${START_FRAME}" "${COUNT}" "${FRAMES_PER_TENSOR}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "Replay frame values must be non-negative integers: ${value}" >&2
        exit 2
    fi
done
if [[ "${FRAMES_PER_TENSOR}" -eq 0 ]]; then
    echo "--frames-per-tensor must be greater than zero" >&2
    exit 2
fi

INPUT="$(cd "$(dirname "${INPUT}")" && pwd -P)/$(basename "${INPUT}")"
if [[ -n "${DARK}" ]]; then
    DARK="$(cd "$(dirname "${DARK}")" && pwd -P)/$(basename "${DARK}")"
fi
mkdir -p "$(dirname "${OUTPUT}")"
OUTPUT_DIR="$(cd "$(dirname "${OUTPUT}")" && pwd -P)"
OUTPUT_NAME="$(basename "${OUTPUT}")"

SUBTRACT_DARK=false
APPLY_MASK=false
APPLY_BLR=false
APPLY_DYNAMIC=false
DARK_PATH=""
if [[ -n "${DARK}" ]]; then
    SUBTRACT_DARK=true
    APPLY_MASK="${USE_MASK}"
    APPLY_BLR="${USE_BLR}"
    APPLY_DYNAMIC="${USE_DYNAMIC_MASK}"
    DARK_PATH="/calibration/dark.h5"
fi

PROCESSOR_NOOP=true
if [[ "${REDUCE}" == true ]]; then
    PROCESSOR_NOOP=false
fi

CONFIG="$(mktemp "${TMPDIR:-/tmp}/stem_daqiri_replay.XXXXXX.yaml")"
trap 'rm -f "${CONFIG}"' EXIT

cat >"${CONFIG}" <<EOF
%YAML 1.2
---
source: "hdf5"

replayer:
  filepath: "/data/input.h5"
  dataset_name: "${INPUT_DATASET}"
  repeat: false
  start_frame: ${START_FRAME}
  frames_per_tensor: ${FRAMES_PER_TENSOR}
  count: ${COUNT}

processor:
  # noop:true preserves every processed frame; it does not disable corrections.
  noop: ${PROCESSOR_NOOP}
  subtract_dark_frame: ${SUBTRACT_DARK}
  dark_frame_path: "${DARK_PATH}"
  dark_frame_dataset: "${DARK_DATASET}"
  apply_valid_pixel_mask: ${APPLY_MASK}
  valid_pixel_mask_dataset: "${MASK_DATASET}"
  apply_blr_correction: ${APPLY_BLR}
  blr_rows: 30
  blr_zlp_width: 768
  blr_zlp_group_columns: 4
  blr_core_group_columns: 16
  apply_dynamic_half_column_mask: ${APPLY_DYNAMIC}
  dynamic_mask_median_window_pixels: 31
  dynamic_mask_threshold_ratio: 1.0
  dynamic_mask_threshold_offset: 500.0
  dynamic_mask_excluded_edge_rows: 32
  dynamic_mask_two_sided: true

writer:
  filepath: "/output/${OUTPUT_NAME}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 3
EOF

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
fi
DOCKER_RUN=("${DOCKER[@]}" run --rm)
if [[ -t 0 && -t 1 ]]; then
    DOCKER_RUN+=(-it)
fi

MOUNTS=(
    -v "${CONFIG}:/run/stem_replay.yaml:ro"
    -v "${INPUT}:/data/input.h5:ro"
    -v "${OUTPUT_DIR}:/output"
)
if [[ -n "${DARK}" ]]; then
    MOUNTS+=(-v "${DARK}:/calibration/dark.h5:ro")
fi

echo "Running DAQIRI HDF5 replay:"
echo "  input  : ${INPUT}:${INPUT_DATASET}"
if [[ -n "${DARK}" ]]; then
    echo "  dark   : ${DARK}:${DARK_DATASET}"
fi
echo "  output : ${OUTPUT_DIR}/${OUTPUT_NAME}:/processed"
echo "  frames : start=${START_FRAME}, count=${COUNT}, tensor=${FRAMES_PER_TENSOR}"

"${DOCKER_RUN[@]}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${MOUNTS[@]}" \
    "${IMAGE}" \
    /opt/stem_daqiri/bin/stem_daqiri_rx /run/stem_replay.yaml

if [[ ! -f "${OUTPUT_DIR}/${OUTPUT_NAME}" ]]; then
    echo "DAQIRI completed without creating ${OUTPUT_DIR}/${OUTPUT_NAME}" >&2
    exit 1
fi
echo "Wrote ${OUTPUT_DIR}/${OUTPUT_NAME}"
