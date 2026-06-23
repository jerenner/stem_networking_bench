#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Repeatable DAQIRI validation gates for the parity work:
#   - deterministic HDF5 replay/processor/writer suite
#   - IGX live non-HDS loopback smoke
#   - IGX live non-HDS wire-rate loopback
#   - IGX live HDS loopback smoke
#   - config-shape and failure-mode checks

set -euo pipefail

IMAGE="${DAQIRI_IMAGE:-stem_daqiri:phase3-hdf5}"
CONTAINER="${DAQIRI_CONTAINER:-stem_daqiri_live}"
CUDA_COMPAT_LIB="${DAQIRI_CUDA_COMPAT_LIB:-/usr/local/cuda/compat/lib}"
KEEP_TMP=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<EOF
Usage:
  $0 [command] [options]

Commands:
  hdf5        Run deterministic HDF5 replay/processor/writer suite. Default.
  config      Run config parser/failure-mode checks.
  live-smoke  Run non-HDS IGX loopback at 1 Gbps for 5 seconds.
  live-wire   Run non-HDS IGX loopback unbounded for 10 seconds.
  hds-smoke   Run HDS IGX loopback at 1 Gbps for 5 seconds.
  hds-wire    Run HDS IGX loopback unbounded for 10 seconds as a stress check.
  live-writer Run non-HDS loopback with writer.noop=false and verify HDF5 output.
  live-pixel  Run deterministic live non-HDS/HDS HDF5 writer comparison.
  live-all    Run all live checks.
  all         Run hdf5, config, then live-all.

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
    f.create_dataset("/float_frames", data=frames.astype(np.float32) / 10.0)
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

write("stem_case_float_raw.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/float_frames"
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
  filepath: "{tmp / 'stem_case_float_raw_out.h5'}"
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

write("stem_case_float_correction.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/float_frames"
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
  filepath: "{tmp / 'stem_case_float_correction_out.h5'}"
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

write("stem_case_float_reduce.yaml", f"""%YAML 1.2
---
source: "hdf5"
replayer:
  filepath: "{common_input}"
  dataset_name: "/float_frames"
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
  filepath: "{tmp / 'stem_case_float_reduce_out.h5'}"
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
run_ok stem_case_float_raw.yaml
run_ok stem_case_window.yaml
run_ok stem_case_correction.yaml
run_ok stem_case_float_correction.yaml
run_ok stem_case_reduce.yaml
run_ok stem_case_float_reduce.yaml
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
    float_frames = f_in["/float_frames"][...]
    dyn = f_in["/dyn_frames"][...]
    dark = f_corr["/dark"][...]
    mask = f_corr["/valid_pixel_mask"][...]

expected = {
    "raw": (frames, np.uint16, tmp / "stem_case_raw_out.h5"),
    "float_raw": (float_frames, np.float32, tmp / "stem_case_float_raw_out.h5"),
    "window": (frames[1:4], np.uint16, tmp / "stem_case_window_out.h5"),
    "correction": (
        ((frames[:2].astype(np.float32) - dark) * mask).astype(np.float32),
        np.float32,
        tmp / "stem_case_correction_out.h5",
    ),
    "float_correction": (
        ((float_frames[:2] - dark) * mask).astype(np.float32),
        np.float32,
        tmp / "stem_case_float_correction_out.h5",
    ),
    "reduce": (
        np.sum(frames[:3].astype(np.float32), axis=0, keepdims=True),
        np.float32,
        tmp / "stem_case_reduce_out.h5",
    ),
    "float_reduce": (
        np.sum(float_frames[:3], axis=0, keepdims=True).astype(np.float32),
        np.float32,
        tmp / "stem_case_float_reduce_out.h5",
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

run_config_fail() {
    local cfg="$1"
    local expected_regex="$2"
    local log="${cfg}.log"

    if docker run --rm --gpus all \
        -v "$(dirname "${cfg}"):$(dirname "${cfg}")" \
        "${IMAGE}" \
        bash -lc "export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:\${LD_LIBRARY_PATH:-}; stem_daqiri_rx '${cfg}'" \
        > "${log}" 2>&1; then
        echo "[config] expected ${cfg} to fail" >&2
        cat "${log}" >&2
        return 1
    fi
    if ! grep -Eq "${expected_regex}" "${log}"; then
        echo "[config] ${cfg} failed without expected regex: ${expected_regex}" >&2
        cat "${log}" >&2
        return 1
    fi
}

run_config_suite() {
    local tmpdir
    tmpdir="$(mktemp -d /tmp/stem_daqiri_config_validation.XXXXXX)"
    echo "[config] workdir: ${tmpdir}"

    cat > "${tmpdir}/bad_frames.yaml" <<'EOF'
%YAML 1.2
---
source: "network"
stem_rx:
  interface_name: "rx_port"
  frames_per_tensor: 0
  expected_source_mask: 255
writer:
  noop: true
EOF

    cat > "${tmpdir}/bad_source_mask.yaml" <<'EOF'
%YAML 1.2
---
source: "network"
stem_rx:
  interface_name: "rx_port"
  frames_per_tensor: 16
  expected_source_mask: 0
writer:
  noop: true
EOF

    cat > "${tmpdir}/top_level_receiver_zero_frames.yaml" <<'EOF'
%YAML 1.2
---
source: "network"
stem_rx:
  interface_name: "rx_port"
  frames_per_tensor: 16
  expected_source_mask: 255
receiver0:
  frames_per_tensor: 0
writer:
  noop: true
EOF

    cat > "${tmpdir}/top_level_multi_writer.yaml" <<'EOF'
%YAML 1.2
---
source: "network"
num_receivers: 2
stem_rx:
  interface_name: "rx_port"
  frames_per_tensor: 16
  expected_source_mask: 255
receiver0:
  interface_name: "rx_port0"
receiver1:
  interface_name: "rx_port1"
writer:
  filepath: "/tmp/stem_multi_writer.h5"
  dataset_name: "/processed"
  noop: false
EOF

    cat > "${tmpdir}/ambiguous_receiver.yaml" <<'EOF'
%YAML 1.2
---
source: "network"
stem_rx:
  interface_name: "rx_port"
  frames_per_tensor: 16
  expected_source_mask: 255
  receiver0:
    interface_name: "nested_rx_port"
receiver0:
  interface_name: "top_rx_port"
writer:
  noop: true
EOF

    run_config_fail "${tmpdir}/bad_frames.yaml" \
        'frames_per_tensor must be > 0'
    run_config_fail "${tmpdir}/bad_source_mask.yaml" \
        'expected_source_mask must enable at least one source'
    run_config_fail "${tmpdir}/top_level_receiver_zero_frames.yaml" \
        'frames_per_tensor must be > 0'
    run_config_fail "${tmpdir}/top_level_multi_writer.yaml" \
        'num_receivers > 1 with writer\.noop=false'
    run_config_fail "${tmpdir}/ambiguous_receiver.yaml" \
        'receiver override.*ambiguous'

    python3 - "${REPO_ROOT}" <<'PY'
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
cfg_dir = root / "cpp_daqiri" / "configs"
rx_configs = sorted(cfg_dir.glob("stem_rx_*.yaml"))
if not rx_configs:
    raise AssertionError("no checked-in stem_rx configs found")

for path in rx_configs:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rx = data.get("stem_rx", {})
    frames_per_tensor = int(rx.get("frames_per_tensor", 0))
    source_mask = int(rx.get("expected_source_mask", 0))
    if frames_per_tensor <= 0:
        raise AssertionError(f"{path.name}: frames_per_tensor must be > 0")
    if source_mask & 0xff == 0:
        raise AssertionError(f"{path.name}: expected_source_mask enables no sources")
    if rx.get("hds", False) and rx.get("gpu_header_extract", False):
        raise AssertionError(f"{path.name}: hds and gpu_header_extract are mutually exclusive")
    if path.name.startswith("stem_rx_spark"):
        if frames_per_tensor != 128:
            raise AssertionError(f"{path.name}: Spark parity configs should use frames_per_tensor=128")
    if path.name == "stem_rx_igx_production.yaml":
        processor = data.get("processor", {})
        writer = data.get("writer", {})
        if frames_per_tensor != 16:
            raise AssertionError(f"{path.name}: IGX production is expected to keep frames_per_tensor=16")
        if not processor.get("noop", True) or not writer.get("noop", True):
            raise AssertionError(f"{path.name}: 16-frame IGX production config must remain noop/RX-only")
    print(f"PASS {path.name}: frames_per_tensor={frames_per_tensor} source_mask=0x{source_mask:x}")
PY

    if [[ "${KEEP_TMP}" -eq 1 ]]; then
        echo "[config] kept ${tmpdir}"
    else
        rm -rf "${tmpdir}"
    fi
    echo "[config] OK"
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

assert_live_log_core_clean() {
    local log="$1"
    require_log_regex "${log}" 'packets received[[:space:]]*:[[:space:]]*[1-9][0-9]*'
    require_log_regex "${log}" 'frames assembled[[:space:]]*:[[:space:]]*[1-9][0-9]*'
    require_log_regex "${log}" 'unexpected source:[[:space:]]+0'
    require_log_regex "${log}" 'sink pool drops[[:space:]]+:[[:space:]]+0'
    require_log_regex "${log}" 'sink errors[[:space:]]+:[[:space:]]+0'
}

assert_live_log_dpdk_zero() {
    local log="$1"
    require_log_regex "${log}" 'Missed packets:[[:space:]]+0'
    require_log_regex "${log}" 'Errored packets:[[:space:]]+0'
    require_log_regex "${log}" 'RX out of buffers:[[:space:]]+0'
}

assert_live_log_clean() {
    local log="$1"
    assert_live_log_core_clean "${log}"
    assert_live_log_dpdk_zero "${log}"
}

make_live_writer_config() {
    local source_cfg="$1"
    local out_cfg="$2"
    local out_h5="$3"

    python3 - "${source_cfg}" "${out_cfg}" "${out_h5}" <<'PY'
from pathlib import Path
import sys
import yaml

source_cfg = Path(sys.argv[1])
out_cfg = Path(sys.argv[2])
out_h5 = Path(sys.argv[3])

data = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
writer = data.setdefault("writer", {})
writer["filepath"] = str(out_h5)
writer["dataset_name"] = "/processed"
writer["noop"] = False
writer["num_concurrent"] = max(int(writer.get("num_concurrent", 3)), 3)
out_cfg.write_text(
    "%YAML 1.2\n---\n" + yaml.safe_dump(data, sort_keys=False),
    encoding="utf-8",
)
PY
}

verify_hdf5_dataset() {
    local h5_path="$1"
    local min_frames="$2"
    local expected_dtype="$3"

    python3 - "${h5_path}" "${min_frames}" "${expected_dtype}" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

h5_path = Path(sys.argv[1])
min_frames = int(sys.argv[2])
expected_dtype = np.dtype(sys.argv[3])

with h5py.File(h5_path, "r") as f:
    if "/processed" not in f:
        raise AssertionError(f"{h5_path}: missing /processed")
    dset = f["/processed"]
    if dset.ndim != 3:
        raise AssertionError(f"{h5_path}: /processed rank {dset.ndim} != 3")
    if dset.shape[0] < min_frames:
        raise AssertionError(f"{h5_path}: only {dset.shape[0]} frames, expected >= {min_frames}")
    if dset.dtype != expected_dtype:
        raise AssertionError(f"{h5_path}: dtype {dset.dtype} != {expected_dtype}")
    prefix = dset[: min(dset.shape[0], min_frames)]
    print(f"PASS {h5_path}: shape={dset.shape} dtype={dset.dtype} prefix_sum={float(np.sum(prefix))}")
PY
}

run_live_case() {
    local label="$1"
    local rx_config="$2"
    local rate="$3"
    local tx_seconds="$4"
    local rx_seconds="$5"
    local require_hds="$6"
    local allow_dpdk_drops="${7:-false}"

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
    assert_live_log_core_clean "${log}"
    if [[ "${allow_dpdk_drops}" == "true" ]]; then
        echo "[live:${label}] DPDK drop counters are informational for this stress case"
        require_log_regex "${log}" 'Missed packets:[[:space:]]+[0-9]+'
        require_log_regex "${log}" 'Errored packets:[[:space:]]+[0-9]+'
        require_log_regex "${log}" 'RX out of buffers:[[:space:]]+[0-9]+'
    else
        assert_live_log_dpdk_zero "${log}"
    fi
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

run_hds_wire() {
    run_live_case \
        "live_wire_hds" \
        "/opt/stem_daqiri/bin/configs/stem_rx_igx_loopback_hds.yaml" \
        "0" \
        "10" \
        "30" \
        "true" \
        "true"
}

run_live_writer() {
    require_live_container
    require_host_h5py

    local tmpdir
    tmpdir="$(mktemp -d /tmp/stem_daqiri_live_writer.XXXXXX)"
    local cfg="${tmpdir}/stem_rx_igx_loopback_writer.yaml"
    local h5="${tmpdir}/stem_rx_igx_loopback_writer.h5"
    make_live_writer_config \
        "${REPO_ROOT}/cpp_daqiri/configs/stem_rx_igx_loopback.yaml" \
        "${cfg}" \
        "${h5}"

    run_live_case \
        "live_writer_nonhds" \
        "${cfg}" \
        "1" \
        "2" \
        "20" \
        "false"
    verify_hdf5_dataset "${h5}" 16 "uint16"

    if [[ "${KEEP_TMP}" -eq 1 ]]; then
        echo "[live-writer] kept ${tmpdir}"
    else
        rm -rf "${tmpdir}"
    fi
    echo "[live-writer] OK"
}

run_live_pixel() {
    require_live_container
    require_host_h5py

    local tmpdir
    tmpdir="$(mktemp -d /tmp/stem_daqiri_live_pixel.XXXXXX)"
    local nonhds_cfg="${tmpdir}/stem_rx_igx_loopback_writer.yaml"
    local hds_cfg="${tmpdir}/stem_rx_igx_loopback_hds_writer.yaml"
    local nonhds_h5="${tmpdir}/stem_rx_igx_loopback_writer.h5"
    local hds_h5="${tmpdir}/stem_rx_igx_loopback_hds_writer.h5"
    make_live_writer_config \
        "${REPO_ROOT}/cpp_daqiri/configs/stem_rx_igx_loopback.yaml" \
        "${nonhds_cfg}" \
        "${nonhds_h5}"
    make_live_writer_config \
        "${REPO_ROOT}/cpp_daqiri/configs/stem_rx_igx_loopback_hds.yaml" \
        "${hds_cfg}" \
        "${hds_h5}"

    run_live_case \
        "live_pixel_nonhds" \
        "${nonhds_cfg}" \
        "1" \
        "2" \
        "20" \
        "false"
    run_live_case \
        "live_pixel_hds" \
        "${hds_cfg}" \
        "1" \
        "2" \
        "20" \
        "true"

    verify_hdf5_dataset "${nonhds_h5}" 16 "uint16"
    verify_hdf5_dataset "${hds_h5}" 16 "uint16"
    python3 "${REPO_ROOT}/cpp_daqiri/scripts/compare_h5_outputs.py" \
        "${nonhds_h5}" \
        "${hds_h5}" \
        --dataset /processed \
        --max-frames 16

    if [[ "${KEEP_TMP}" -eq 1 ]]; then
        echo "[live-pixel] kept ${tmpdir}"
    else
        rm -rf "${tmpdir}"
    fi
    echo "[live-pixel] OK"
}

case "${COMMAND}" in
    hdf5)
        run_hdf5_suite
        ;;
    config)
        run_config_suite
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
    hds-wire)
        run_hds_wire
        ;;
    live-writer)
        run_live_writer
        ;;
    live-pixel)
        run_live_pixel
        ;;
    live-all)
        run_live_smoke
        run_live_wire
        run_hds_smoke
        run_hds_wire
        run_live_writer
        run_live_pixel
        ;;
    all)
        run_hdf5_suite
        run_config_suite
        run_live_smoke
        run_live_wire
        run_hds_smoke
        run_hds_wire
        run_live_writer
        run_live_pixel
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
