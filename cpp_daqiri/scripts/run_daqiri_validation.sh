#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Repeatable DAQIRI validation gates for the parity work:
#   - deterministic HDF5 replay/processor/writer suite
#   - IGX live non-HDS loopback smoke
#   - IGX live non-HDS wire-rate loopback
#   - IGX live HDS loopback smoke

set -euo pipefail

IMAGE="${DAQIRI_IMAGE:-stem_daqiri:phase3-hdf5}"
CONTAINER="${DAQIRI_CONTAINER:-stem_daqiri_live}"
CUDA_COMPAT_LIB="${DAQIRI_CUDA_COMPAT_LIB:-/usr/local/cuda/compat/lib}"
KEEP_TMP=0

usage() {
    cat <<EOF
Usage:
  $0 [command] [options]

Commands:
  hdf5        Run deterministic HDF5 replay/processor/writer suite. Default.
  live-smoke  Run non-HDS IGX loopback at 1 Gbps for 5 seconds.
  live-wire   Run non-HDS IGX loopback unbounded for 10 seconds.
  hds-smoke   Run HDS IGX loopback at 1 Gbps for 5 seconds.
  live-all    Run live-smoke, live-wire, then hds-smoke.
  all         Run hdf5 then live-all.

Options:
  --image IMAGE              Docker image for hdf5 one-shot tests.
                             Default: ${IMAGE}
  --container NAME           Running privileged live-test container.
                             Default: ${CONTAINER}
  --cuda-compat-lib PATH     CUDA forward-compat lib path for docker exec.
                             Default: ${CUDA_COMPAT_LIB}
  --keep-tmp                 Keep generated HDF5 suite files under /tmp.

Live tests require a running container launched with privileged host network,
GPU access, /dev/hugepages, and /tmp mounted. Example:

  docker run -d --name stem_daqiri_live \\
    --privileged --network host --ipc=host --gpus all \\
    --ulimit memlock=-1 --ulimit stack=67108864 \\
    -v /dev/hugepages:/dev/hugepages -v /tmp:/tmp \\
    -v "\$PWD":/workspace/stem -w /workspace/stem \\
    ${IMAGE} sleep infinity
EOF
}

COMMAND="${1:-hdf5}"
if [[ $# -gt 0 ]]; then
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        --cuda-compat-lib)
            CUDA_COMPAT_LIB="$2"
            shift 2
            ;;
        --keep-tmp)
            KEEP_TMP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

docker_exec_cuda() {
    local cmd="$1"
    docker exec "${CONTAINER}" bash -lc \
        "export LD_LIBRARY_PATH=${CUDA_COMPAT_LIB}:\${LD_LIBRARY_PATH:-}; ${cmd}"
}

docker_exec_cuda_detached() {
    local cmd="$1"
    docker exec -d "${CONTAINER}" bash -lc \
        "export LD_LIBRARY_PATH=${CUDA_COMPAT_LIB}:\${LD_LIBRARY_PATH:-}; ${cmd}"
}

require_live_container() {
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
        echo "live container '${CONTAINER}' is not running" >&2
        echo "Launch it first, or pass --container NAME." >&2
        exit 1
    fi
}

require_host_h5py() {
    python3 - <<'PY'
import h5py  # noqa: F401
import numpy  # noqa: F401
PY
}

run_hdf5_suite() {
    require_host_h5py

    local tmpdir
    tmpdir="$(mktemp -d /tmp/stem_daqiri_hdf5_validation.XXXXXX)"
    echo "[hdf5] workdir: ${tmpdir}"

    STEM_DAQIRI_TMPDIR="${tmpdir}" python3 - <<'PY'
import os
from pathlib import Path

import h5py
import numpy as np

tmp = Path(os.environ["STEM_DAQIRI_TMPDIR"])

with h5py.File(tmp / "stem_pipeline_input.h5", "w") as f:
    frames = np.arange(5 * 4 * 4, dtype=np.uint16).reshape(5, 4, 4)
    f.create_dataset("/frames", data=frames)
    dyn = np.full((2, 6, 3), 10, dtype=np.uint16)
    dyn[:, 1, 1] = 100
    dyn[:, 4, 2] = 80
    f.create_dataset("/dyn_frames", data=dyn)

with h5py.File(tmp / "stem_pipeline_corr.h5", "w") as f:
    dark = np.arange(16, dtype=np.float32).reshape(4, 4) / 2.0
    mask = np.ones((4, 4), dtype=np.float32)
    mask[1, 2] = 0.0
    mask[2, 1] = 0.5
    mask[3, 0] = 0.0
    f.create_dataset("/dark", data=dark)
    f.create_dataset("/valid_pixel_mask", data=mask)
    f.create_dataset("/bad_dark", data=np.zeros((3, 4), dtype=np.float32))

def write(name: str, body: str) -> None:
    (tmp / name).write_text(body, encoding="utf-8")

common_input = tmp / "stem_pipeline_input.h5"
common_corr = tmp / "stem_pipeline_corr.h5"

write("stem_case_raw.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 2
  count: 5
processor:
  noop: true
  subtract_dark_frame: false
  apply_valid_pixel_mask: false
  apply_dynamic_half_column_mask: false
writer:
  filepath: "{tmp / 'stem_case_raw_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 2
""")

write("stem_case_window.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 1
  frames_per_tensor: 2
  count: 3
processor:
  noop: true
  subtract_dark_frame: false
  apply_valid_pixel_mask: false
  apply_dynamic_half_column_mask: false
writer:
  filepath: "{tmp / 'stem_case_window_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 2
""")

write("stem_case_correction.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 2
  count: 2
processor:
  noop: true
  subtract_dark_frame: true
  dark_frame_path: "{common_corr}"
  dark_frame_dataset: "/dark"
  apply_valid_pixel_mask: true
  valid_pixel_mask_dataset: "/valid_pixel_mask"
  apply_dynamic_half_column_mask: false
writer:
  filepath: "{tmp / 'stem_case_correction_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 2
""")

write("stem_case_reduce.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 3
  count: 3
processor:
  noop: false
  subtract_dark_frame: false
  apply_valid_pixel_mask: false
  apply_dynamic_half_column_mask: false
writer:
  filepath: "{tmp / 'stem_case_reduce_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 1
""")

write("stem_case_reduce_correction.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 3
  count: 3
processor:
  noop: false
  subtract_dark_frame: true
  dark_frame_path: "{common_corr}"
  dark_frame_dataset: "/dark"
  apply_valid_pixel_mask: true
  valid_pixel_mask_dataset: "/valid_pixel_mask"
  apply_dynamic_half_column_mask: false
writer:
  filepath: "{tmp / 'stem_case_reduce_correction_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 1
""")

write("stem_case_dynamic.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/dyn_frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 2
  count: 2
processor:
  noop: true
  subtract_dark_frame: false
  apply_valid_pixel_mask: false
  apply_dynamic_half_column_mask: true
  dynamic_mask_median_window_pixels: 3
  dynamic_mask_threshold_ratio: 1.0
  dynamic_mask_threshold_offset: 5.0
writer:
  filepath: "{tmp / 'stem_case_dynamic_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 1
""")

write("stem_case_bad_repeat.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: true
  start_frame: 0
  frames_per_tensor: 1
  count: 1
processor:
  noop: true
writer:
  noop: true
""")

write("stem_case_bad_writer.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 1
  count: 1
processor:
  noop: true
writer:
  filepath: "{tmp / 'missing_dir' / 'stem_bad_writer_out.h5'}"
  dataset_name: "/processed"
  noop: false
  num_concurrent: 1
""")

write("stem_case_bad_correction_shape.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/frames"
  repeat: false
  start_frame: 0
  frames_per_tensor: 1
  count: 1
processor:
  noop: true
  subtract_dark_frame: true
  dark_frame_path: "{common_corr}"
  dark_frame_dataset: "/bad_dark"
  apply_valid_pixel_mask: false
writer:
  noop: true
""")
PY

    cat > "${tmpdir}/run_hdf5_container.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}

run_ok() {
    local cfg="$1"
    echo "RUN ok ${cfg}"
    stem_daqiri_rx "${ROOT}/${cfg}"
}

run_fail() {
    local cfg="$1"
    echo "RUN expected-fail ${cfg}"
    if stem_daqiri_rx "${ROOT}/${cfg}"; then
        echo "expected ${cfg} to fail" >&2
        exit 1
    fi
}

run_ok stem_case_raw.yaml
run_ok stem_case_window.yaml
run_ok stem_case_correction.yaml
run_ok stem_case_reduce.yaml
run_ok stem_case_reduce_correction.yaml
run_ok stem_case_dynamic.yaml

run_fail stem_case_bad_repeat.yaml
run_fail stem_case_bad_writer.yaml
run_fail stem_case_bad_correction_shape.yaml
EOS
    chmod +x "${tmpdir}/run_hdf5_container.sh"

    local status=0
    docker run --rm --gpus all \
        -v "${tmpdir}:${tmpdir}" \
        "${IMAGE}" \
        bash "${tmpdir}/run_hdf5_container.sh" "${tmpdir}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
        echo "[hdf5] FAILED; keeping ${tmpdir}" >&2
        return "${status}"
    fi

    STEM_DAQIRI_TMPDIR="${tmpdir}" python3 - <<'PY'
import os
from pathlib import Path

import h5py
import numpy as np

tmp = Path(os.environ["STEM_DAQIRI_TMPDIR"])
with h5py.File(tmp / "stem_pipeline_input.h5", "r") as f_in, \
        h5py.File(tmp / "stem_pipeline_corr.h5", "r") as f_corr:
    frames = f_in["/frames"][...]
    dyn = f_in["/dyn_frames"][...]
    dark = f_corr["/dark"][...]
    mask = f_corr["/valid_pixel_mask"][...]

expected = {
    "raw": (frames, np.uint16, tmp / "stem_case_raw_out.h5"),
    "window": (frames[1:4], np.uint16, tmp / "stem_case_window_out.h5"),
    "correction": (
        ((frames[:2].astype(np.float32) - dark) * mask).astype(np.float32),
        np.float32,
        tmp / "stem_case_correction_out.h5",
    ),
    "reduce": (
        np.sum(frames[:3].astype(np.float32), axis=0, keepdims=True),
        np.float32,
        tmp / "stem_case_reduce_out.h5",
    ),
    "reduce_correction": (
        np.sum(
            ((frames[:3].astype(np.float32) - dark) * mask).astype(np.float32),
            axis=0,
            keepdims=True,
        ),
        np.float32,
        tmp / "stem_case_reduce_correction_out.h5",
    ),
}

dyn_expected = dyn.astype(np.float32)
dyn_expected[:, 1, 1] = 0.0
dyn_expected[:, 4, 2] = 0.0
expected["dynamic"] = (dyn_expected, np.float32, tmp / "stem_case_dynamic_out.h5")

for name, (want, dtype, path) in expected.items():
    with h5py.File(path, "r") as f:
        got = f["/processed"][...]
    if got.shape != want.shape:
        raise AssertionError(f"{name}: shape {got.shape} != {want.shape}")
    if got.dtype != np.dtype(dtype):
        raise AssertionError(f"{name}: dtype {got.dtype} != {np.dtype(dtype)}")
    if np.issubdtype(got.dtype, np.integer):
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)
    print(f"PASS {name}: shape={got.shape} dtype={got.dtype} sum={float(np.sum(got))}")
PY

    if [[ "${KEEP_TMP}" -eq 1 ]]; then
        echo "[hdf5] kept ${tmpdir}"
    else
        rm -rf "${tmpdir}"
    fi
    echo "[hdf5] OK"
}

wait_for_rx_ready() {
    local log="$1"
    local rc="$2"
    for _ in $(seq 1 40); do
        if docker_exec_cuda "grep -q 'Starting RX Core' '${log}' 2>/dev/null"; then
            echo "[live] RX ready"
            return 0
        fi
        if docker_exec_cuda "test -f '${rc}'"; then
            echo "[live] RX exited before readiness" >&2
            docker_exec_cuda "cat '${log}'" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "[live] RX readiness timeout" >&2
    docker_exec_cuda "tail -180 '${log}'" >&2 || true
    return 1
}

wait_for_rx_done() {
    local rc="$1"
    for _ in $(seq 1 60); do
        if docker_exec_cuda "test -f '${rc}'"; then
            local value
            value="$(docker_exec_cuda "cat '${rc}'")"
            [[ "${value}" == "0" ]] || {
                echo "[live] RX failed with rc=${value}" >&2
                return 1
            }
            return 0
        fi
        sleep 1
    done
    echo "[live] RX completion timeout" >&2
    return 1
}

require_log_regex() {
    local log="$1"
    local regex="$2"
    if ! docker_exec_cuda "grep -Eq '${regex}' '${log}'"; then
        echo "[live] missing expected log regex: ${regex}" >&2
        docker_exec_cuda "tail -160 '${log}'" >&2 || true
        return 1
    fi
}

assert_live_log_clean() {
    local log="$1"
    require_log_regex "${log}" 'packets received[[:space:]]*:[[:space:]]*[1-9][0-9]*'
    require_log_regex "${log}" 'Missed packets:[[:space:]]+0'
    require_log_regex "${log}" 'Errored packets:[[:space:]]+0'
    require_log_regex "${log}" 'RX out of buffers:[[:space:]]+0'
    require_log_regex "${log}" 'unexpected source:[[:space:]]+0'
    require_log_regex "${log}" 'sink errors[[:space:]]+:[[:space:]]+0'
}

run_live_case() {
    local label="$1"
    local rx_config="$2"
    local rate="$3"
    local tx_seconds="$4"
    local rx_seconds="$5"
    local require_hds="$6"

    require_live_container

    local log="/tmp/stem_daqiri_${label}.log"
    local rc="/tmp/stem_daqiri_${label}.rc"

    echo "[live:${label}] starting RX (${rx_config})"
    docker_exec_cuda_detached \
        "rm -f '${log}' '${rc}'; /opt/stem_daqiri/bin/stem_daqiri_rx '${rx_config}' --seconds '${rx_seconds}' > '${log}' 2>&1; echo \$? > '${rc}'"

    wait_for_rx_ready "${log}" "${rc}"

    echo "[live:${label}] starting TX rate=${rate}Gbps seconds=${tx_seconds}"
    docker_exec_cuda \
        "/opt/stem_daqiri/bin/stem_daqiri_tx /opt/stem_daqiri/bin/configs/stem_tx_igx_loopback.yaml --seconds '${tx_seconds}' --rate '${rate}'"

    wait_for_rx_done "${rc}"
    assert_live_log_clean "${log}"
    if [[ "${require_hds}" == "true" ]]; then
        require_log_regex "${log}" 'HDS layout verified'
    fi

    echo "[live:${label}] final RX summary"
    docker_exec_cuda "tail -35 '${log}'"
    echo "[live:${label}] OK"
}

run_live_smoke() {
    run_live_case \
        "live_smoke_nonhds" \
        "/opt/stem_daqiri/bin/configs/stem_rx_igx_loopback.yaml" \
        "1" \
        "5" \
        "30" \
        "false"
}

run_live_wire() {
    run_live_case \
        "live_wire_nonhds" \
        "/opt/stem_daqiri/bin/configs/stem_rx_igx_loopback.yaml" \
        "0" \
        "10" \
        "30" \
        "false"
}

run_hds_smoke() {
    run_live_case \
        "live_smoke_hds" \
        "/opt/stem_daqiri/bin/configs/stem_rx_igx_loopback_hds.yaml" \
        "1" \
        "5" \
        "30" \
        "true"
}

case "${COMMAND}" in
    hdf5)
        run_hdf5_suite
        ;;
    live-smoke)
        run_live_smoke
        ;;
    live-wire)
        run_live_wire
        ;;
    hds-smoke)
        run_hds_smoke
        ;;
    live-all)
        run_live_smoke
        run_live_wire
        run_hds_smoke
        ;;
    all)
        run_hdf5_suite
        run_live_smoke
        run_live_wire
        run_hds_smoke
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "unknown command: ${COMMAND}" >&2
        usage >&2
        exit 1
        ;;
esac
