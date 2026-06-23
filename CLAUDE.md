# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A GPU-accelerated networking benchmark for EELS/STEM microscopy data acquisition:
receive high-speed UDP packets, assemble them into frames on the GPU, process, and
write to HDF5. It contains **two parallel implementations** of the same pipeline that
are meant to be compared head-to-head:

- **`cpp/`** — the original, "known-good" pipeline built on the **NVIDIA Holoscan SDK**
  (Advanced Network / DPDK). RX-only: `StemReceiverOp` → `PyTorchProcessorOp` →
  `HDF5WriterOp`, with `HDF5ReplayerOp` as a file-based source for offline runs. Packaged
  as a HoloHub-style example app (see `cpp/metadata.json`).
- **`cpp_daqiri/`** — a phased port of that pipeline onto **daqiri** (`third_party/daqiri`,
  a NVIDIA DPDK transport lib), adding a paced **TX** binary and a matching **RX** binary
  for **IGX Orin + RTX 6000 Ada** and **two DGX Spark (GB10) nodes** connected
  back-to-back by ConnectX-7 NICs.

The linchpin tying them together is the **STEM wire format** (`cpp_daqiri/common/stem_packet.h`,
mirrored from `cpp/stem_receiver_op.h`): the daqiri TX emits bytes the original Holoscan RX
can assemble, and the new daqiri RX assembles bytes the Holoscan TX path produces. The end
goal (`cpp_daqiri/` "Phase 3") is a **parity gate**: daqiri must match or beat Holoscan on
Gbps, drops, p50/p99 latency, and processor fps. Results live in
`cpp_daqiri/benchmarks/results.md`.

### STEM frame geometry (shared by both implementations)

- Frame = 1024 rows × 3840 `uint16` samples = 8 sources × 128 rows/source.
- **1024 packets per frame**, one row per packet. On-wire packet = 7786 B =
  42 B (Eth+IPv4+UDP) + 64 B STEM custom header + 7680 B row payload.
- `row_number` is a 16-bit header field that wraps every 16384 rows (= 128 frames);
  the gather kernel reconstructs absolute frame index from the wrapped sequence.
- Changing any of these constants means changing **both** `stem_packet.h` and
  `stem_receiver_op.h` or TX/RX parity silently breaks.

## Building

`third_party/daqiri` may be checked out empty. In this workspace `.git/config` can be a
read-only bind mount, so populate daqiri with a direct clone instead of `git submodule
update --init`:

```bash
rmdir third_party/daqiri 2>/dev/null || true
git clone --recursive https://github.com/NVIDIA/daqiri.git third_party/daqiri
git -C third_party/daqiri checkout 8c5d69fa3c9bf9e57f6625114b5f0828bb592729
git -C third_party/daqiri submodule update --init --recursive
```

### `cpp_daqiri/` (the active work) — two-stage Docker build

Build the daqiri base image and then the project image on the target machine.

```bash
# 1) One-time base image: daqiri + PyTorch + patched DPDK 25.11 + CUDA, at /opt/daqiri.
cd third_party/daqiri && \
    IMAGE_TAG=daqiri-torch:local BASE_IMAGE=torch BASE_TARGET=dpdk \
    DAQIRI_ENGINE="dpdk" scripts/build-container.sh
cd ../..

# 2) stem_daqiri image. Phase flags opt into the TX/RX binaries (both default OFF).
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    -t stem_daqiri:phase3 .
```

The phased CMake flags (`STEM_DAQIRI_BUILD_TX` / `_RX` in `cpp_daqiri/CMakeLists.txt`) gate
which binaries get built: Phase 0 builds only `stem_daqiri_hello` (link check), Phase 1 adds
`stem_daqiri_tx`, Phase 2 adds `stem_daqiri_rx`. Phase 3 is the **same binary as Phase 2** —
its behavior (latency stamping, dark correction, valid-pixel mask, dynamic mask, and
frame reduction) is toggled by YAML (`stamp_epoch_us`, `capture_latency`, and the
top-level `processor` block), not build flags. Legacy `stem_rx.subtract_dark` and
`stem_rx.apply_valid_pixel_mask` remain deprecated aliases when `processor` is absent.
Use `-DSTEM_DAQIRI_REQUIRE_HDF5=ON` for parity builds that must include HDF5
writer/replay/correction-file support.

`cpp_daqiri/scripts/verify_phase.sh {phase0|phase1|phase2|phase3}` runs the build + smoke-test
gate for a given phase on a single Spark.

### `cpp/` (Holoscan) — environment image + CMake

The top-level `Dockerfile` produces only the **environment** image (Holoscan SDK + DPDK +
LibTorch/PyTorch built from source on aarch64 + HDF5 + MatX); it does **not** build the app.
The `cpp/` app is a standalone CMake project (`find_package(holoscan / Torch / HDF5)`, fetches
MatX) built against that environment, or installed as a HoloHub example. Build the env image
once with `docker build -t stem_holoscan:local .`.

When building `cpp/` directly on the IGX host, set the LibTorch env first (see `README.md`):
```bash
export LIBTORCH="/home/daquser/jrenner/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-12.6/bin:$LIBTORCH/bin:$PATH"
```
**ABI gotcha:** `cpp/CMakeLists.txt` deliberately strips `-D_GLIBCXX_USE_CXX11_ABI=0` and forces
ABI=1, because PyTorch ships ABI=0 and it must match Holoscan's ABI=1. Don't "simplify" that block.

## Running

### Two-Spark topology (`cpp_daqiri/`)

| Logical | Hostname | NIC role | Use this PCIe addr |
| --- | --- | --- | --- |
| spark-stacked-01 | `spark-960b` | TX | `0002:01:00.0` |
| spark-stacked-02 | `spark-201a` | RX | `0002:01:00.0` |

Per-side smoke wrappers (start RX ~5 s before TX):
```bash
# RX, on spark-stacked-02
cpp_daqiri/scripts/run_phase2_rx.sh --seconds 14
# TX, on spark-stacked-01
cpp_daqiri/scripts/run_phase2_tx.sh --seconds 10 --rate 50
```

Phase 3 parity sweep — **preferred** path is the orchestrator, run from spark-960b, which
drives the RX over SSH so RX init and TX launch stay aligned every iteration:
```bash
cpp_daqiri/scripts/run_phase3_sweep_orchestrated.sh \
    --rates "10 25 50 80" --runs 1 --seconds 8 \
    --outdir cpp_daqiri/benchmarks/sweep_<utc>
cpp_daqiri/scripts/parse_phase3_results.py --duration 8 \
    --daqiri-tx-dir <sweep> --daqiri-rx-dir <sweep> \
    --holoscan-tx-dir <logs> --holoscan-rx-dir <logs>   # writes benchmarks/results.md
```
The orchestrator only knows the daqiri binary; for the Holoscan baseline use the older
per-side `run_phase3_sweep_{tx,rx}.sh --binary holoscan` scripts.

`stem_daqiri_tx <config.yaml> [--seconds N] [--rate GBPS]` and
`stem_daqiri_rx <config.yaml> [--seconds N]` take CLI overrides so one YAML serves the whole
sweep. Configs are baked into the image at `/opt/stem_daqiri/bin/configs/`.

### Holoscan (`cpp/`)

Pick the data source via the YAML's `source:` key (`hdf5` vs `network`); the network path
reads `num_receivers` (defaults to 2 = dual-NIC FPGA topology; single-NIC reads a bare
`receiver:` block). Configs: `run_with_hdf5.yaml`, `run_with_network*.yaml`,
`run_with_network_fpga_1rcv.yaml` (single-NIC, used to validate daqiri TX against Holoscan RX).

## Critical runtime requirements — `cpp_daqiri/`

`cpp_daqiri/` has three memory/header paths selected by YAML. Do not replace one with the
other:

- **Spark / GB10 unified memory**: daqiri memory region `kind: host_pinned`; RX keeps
  `stem_rx.gpu_header_extract: false` or omits the key. Headers are read on the CPU from
  host-readable packet buffers.
- **IGX / discrete RTX 6000 Ada dGPU**: daqiri memory region `kind: device`; RX sets
  `stem_rx.gpu_header_extract: true`. Packet payloads stay in VRAM, and only extracted
  header metadata is copied back to the CPU.
- **Header/data split (HDS)**: RX queue memory regions must be ordered header first,
  payload second, and `stem_rx.hds: true` makes the gather pointer point at segment 1
  payload byte 0. The runtime verifies the first observed split is exactly 106/7680
  before admitting packets. Use `configs/stem_rx_spark_hds.yaml` for Spark or
  `configs/stem_rx_igx_loopback_hds.yaml` for IGX loopback. Keep
  `tile_duplicate_prefix_to_simulate_payload: true` until the production FPGA wire
  payload is confirmed to be native 8192 B and the memory-region sizes/MTU docs are
  updated together. Treat HDS as a functional parity path, not a high-throughput
  parity gate: Holoscan's current configs leave `split_boundary: false`, and its
  HDS code consumes one RX queue with two segments, not a separate two-queue
  header/payload pairing.

IGX production RX is RX-only against the FPGA. Use `configs/stem_rx_igx_production.yaml`
as the starting point:

- RX `master_core: 8`, RX poll `cpu_core: 9`.
- Verified loopback NICs are TX `0005:03:00.0` and RX `0005:03:00.1`; production FPGA
  PF and flow match are per box.
- Boot-reserved hugepages are 3 x 1 GiB; keep `/dev/hugepages` mounted in the container.
- `writer.noop: true` is the production default. The HDF5 sink is smoke/debug only and
  must not be placed on the poll loop; RX uses a bounded leased output-buffer pool and
  counts `sink pool drops` instead of blocking when the pool is exhausted.
- Multi-receiver DAQIRI output is valid only with `writer.noop: true`; a shared HDF5
  writer would interleave receiver streams by arrival order and is refused.
- IGX production keeps `frames_per_tensor: 16` intentionally for RX-only/noop latency and
  buffering. Use a 128-frame parity config for reduced `/processed` comparisons.
- Latency percentiles are `system_clock(now) - epoch_us`; absolute values require
  synchronized TX/RX clocks, and negative-skew samples are dropped.

On the Spark bench the following cause silent total packet loss or refusal to start if
wrong, and are not optional:

- **Hugepages**: `sudo sysctl -w vm.nr_hugepages=2048` (4 GiB). The Phase 3 RX pool
  (262144 mbufs × 8064 B ≈ 2.1 GiB host-pinned) fails to allocate at the old 512-page setting.
- **NIC selection**: use `0002:01:00.0`, **not** `0000:01:00.0`. The primary pair is bound to
  the kernel's link-local management IP; putting DPDK there triggers `flow_isolation: true`
  silently dropping 100% of UDP-4096 ingress even though `rx_packets_phy` counts it at the NIC.
- **Memory regions** must be `kind: host_pinned` (GB10's GPUDirect path), not device memory.
- **MTU**: keep ≥ packet size. The daqiri stock 8064 B benchmark exceeds spark-201a's primary
  NIC default MTU (8046) and silently drops on RX; the STEM 7786 B frame is fine.
- Containers run `--privileged --network host --gpus all --ulimit memlock=-1` with
  `/dev/hugepages` mounted.

## Profiling

Set `HOLOSCAN_ENABLE_PROFILE=1` to enable Holoscan's `app->track()` and the NVTX ranges
(`cpp/nvtx_ranges.hpp`). `run_profile` is a ready-made `nsys profile` invocation for the
FPGA-network config.

## Python utilities (host-side, offline)

- `make_dark_frame.py` — build a blinker-aware averaged dark frame from an HDF5 frame stack
  (input to the processor's dark-subtraction).
- `cpp_daqiri/scripts/compare_h5_outputs.py` — exact/toleranced pixel-level HDF5 parity check.
  Exact output is expected for deterministic replay or `processor.noop:true` uint16 gather
  output. DAQIRI-vs-Holoscan reduced float output should use `--rtol` because DAQIRI's CUDA
  frame sum is not bit-identical to `torch::sum(0)`.
- `verify_output.py` / `plot_h5_frames.py` — compare/plot input vs processed HDF5 frames.

## Conventions

- `.gitignore` keeps the **parsed** sweep outputs (`benchmarks/parity_table_*.md`,
  `results.md`) but discards raw per-run logs (`sweep_*/`, `logs_*`); they regenerate on
  every sweep. Don't commit raw logs.
- The CUDA kernels in `cpp/kernels.cu` (`gather_packets`, `extract_packet_headers`, …) are
  ported 1:1 into `cpp_daqiri/common/stem_kernels.{cu,h}` with a `stem_` prefix. Keep them in
  sync — the parity test depends on identical frame-assembly math.
