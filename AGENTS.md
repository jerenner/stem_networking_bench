# AGENTS.md

Guidance for coding agents working in this repository.

## Repository Shape

This is a GPU-accelerated networking benchmark for EELS/STEM microscopy data
acquisition. It receives high-speed UDP packets, assembles STEM frames on the
GPU, optionally processes them, and writes to HDF5 for verification.

There are two implementations of the same wire-format pipeline:

- `cpp/`: the original NVIDIA Holoscan SDK implementation. It is the reference
  for operator semantics, Advanced Network/DPDK RX, PyTorch processing, HDF5
  writing, and Holoscan HDF5 replay.
- `cpp_daqiri/`: the active daqiri implementation using
  `third_party/daqiri`. It contains the paced TX binary, daqiri RX binary, IGX
  loopback/FPGA configs, two-Spark configs, HDF5 replay/writer support, live
  validation, and Spark sweep wrappers.

The shared contract is the STEM wire format in
`cpp_daqiri/common/stem_packet.h`, mirrored from `cpp/stem_receiver_op.h`.
The daqiri TX emits bytes that the Holoscan RX can assemble, and the daqiri RX
assembles the same wire format. Keep frame-assembly constants synchronized
between both implementations.

## STEM Frame Geometry

- Frame = 1024 rows x 3840 `uint16` samples = 8 sources x 128 rows/source.
- One frame has 1024 packets, one row per packet.
- On-wire packet = 7786 B:
  42 B Ethernet/IPv4/UDP + 64 B STEM custom header + 7680 B row payload.
- `row_number` is a 16-bit header field that wraps every 16384 rows
  (= 128 frames). Gather code reconstructs absolute frame index from the
  wrapped sequence.

## DAQIRI Submodule And Build

Current DAQIRI pin:

```bash
git -C third_party/daqiri checkout 3cce706f5caf1a97351aeaf459fffb4a39478922
```

If `third_party/daqiri` is empty, first try:

```bash
git submodule update --init --recursive third_party/daqiri
```

If `.git/config` is a read-only bind mount, populate it with a direct clone:

```bash
rmdir third_party/daqiri 2>/dev/null || true
git clone --recursive https://github.com/NVIDIA/daqiri.git third_party/daqiri
git -C third_party/daqiri checkout 3cce706f5caf1a97351aeaf459fffb4a39478922
git -C third_party/daqiri submodule update --init --recursive
```

Build the daqiri base image once on the target machine:

```bash
cd third_party/daqiri
IMAGE_TAG=daqiri-torch:local BASE_IMAGE=torch BASE_TARGET=dpdk \
    DAQIRI_ENGINE="dpdk" scripts/build-container.sh
cd ../..
```

Build the parity-capable STEM DAQIRI image. The `stem_daqiri:parity-hdf5`
tag enables TX, RX, and mandatory HDF5 replay/writer/correction-file support.

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    --build-arg STEM_DAQIRI_REQUIRE_HDF5=ON \
    -t stem_daqiri:parity-hdf5 .
```

`DAQIRI_ENGINE` is the current daqiri build knob for engine backends. Older
documentation used `DAQIRI_MGR`; the STEM CMake wrapper still accepts it as a
compatibility alias, but new builds should use `DAQIRI_ENGINE`.

`STEM_DAQIRI_BUILD_TX` and `STEM_DAQIRI_BUILD_RX` control which binaries are
built. With both OFF, only `stem_daqiri_hello` is built for link/self-test
validation. Enable TX for `stem_daqiri_tx`; enable both TX and RX for
`stem_daqiri_rx`. Use `STEM_DAQIRI_REQUIRE_HDF5=ON` when HDF5 replay, writer,
and correction-file support must be present.

## HDF5 Replay Caveat

DAQIRI HDF5 replay is finite-only and intentionally rejects
`replayer.repeat:true`. Holoscan can repeat/wrap HDF5 replay input, but parity
replay runs should use `repeat:false` on both sides.

The DAQIRI replay path accepts `uint16` and `float32` `[frames,H,W]` datasets.
Exact HDF5 comparisons are strongest for deterministic replay or
`processor.noop:true` uint16 gather output. Reduced float `/processed` output
from DAQIRI vs Holoscan should use a relative tolerance because DAQIRI's CUDA
sum order is not bit-identical to `torch::sum(0)`.

## Holoscan Build Notes

The top-level `Dockerfile` builds only the Holoscan environment image
(Holoscan SDK, DPDK, LibTorch/PyTorch on aarch64, HDF5, MatX). It does not
build the app. The `cpp/` app is a standalone CMake project built against that
environment or installed as a HoloHub-style example.

Build the environment image with:

```bash
docker build -t stem_holoscan:local .
```

When building `cpp/` directly on the IGX host, set:

```bash
export LIBTORCH="/home/daquser/jrenner/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-12.6/bin:$LIBTORCH/bin:$PATH"
```

`cpp/CMakeLists.txt` deliberately strips `-D_GLIBCXX_USE_CXX11_ABI=0` and
forces ABI=1 because PyTorch ships ABI=0 and Holoscan expects ABI=1. Do not
remove that compatibility block without validating both sides.

## Common DAQIRI Commands

Offline validation:

```bash
cpp_daqiri/scripts/run_daqiri_validation.sh hdf5
cpp_daqiri/scripts/run_daqiri_validation.sh config
```

IGX loopback:

```bash
cpp_daqiri/scripts/run_igx_loopback.sh --rate 20 --tx-seconds 10 --rx-seconds 60
```

Two-Spark wrappers:

```bash
# RX on spark-stacked-02 / spark-201a
cpp_daqiri/scripts/run_spark_daqiri_rx.sh --seconds 14

# TX on spark-stacked-01 / spark-960b
cpp_daqiri/scripts/run_spark_daqiri_tx_for_rx.sh --seconds 10 --rate 50

# Orchestrated Spark parity/throughput sweep from spark-960b
cpp_daqiri/scripts/run_spark_parity_sweep_orchestrated.sh \
    --rates "10 25 50 80" --runs 1 --seconds 8 \
    --outdir cpp_daqiri/benchmarks/sweep_<utc>
```

Parse sweep logs with the existing parser:

```bash
cpp_daqiri/scripts/parse_spark_parity_results.py \
    --daqiri-tx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --daqiri-rx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --holoscan-tx-dir cpp_daqiri/benchmarks/logs_tx_<utc> \
    --holoscan-rx-dir cpp_daqiri/benchmarks/logs_rx_<utc> \
    --duration 8
```

Do not edit `cpp_daqiri/scripts/parse_spark_parity_results.py` unless
specifically asked.

## Runtime Requirements

`cpp_daqiri/` has three memory/header paths selected by YAML. Do not collapse
them into one path:

- Spark / GB10 unified memory: daqiri memory region `kind: host_pinned`; RX
  keeps `stem_rx.gpu_header_extract: false` or omits the key.
- IGX / discrete RTX 6000 Ada dGPU: daqiri memory region `kind: device`; RX
  sets `stem_rx.gpu_header_extract: true`.
- Header/data split (HDS): RX queue memory regions are ordered header first,
  payload second, and `stem_rx.hds: true` makes gather use segment 1 payload
  byte 0. Use `configs/stem_rx_spark_hds.yaml` for Spark or
  `configs/stem_rx_igx_loopback_hds.yaml` for IGX loopback.

Keep `tile_duplicate_prefix_to_simulate_payload: true` until the production
FPGA wire payload is confirmed to be native 8192 B and the memory-region sizes
and MTU docs are updated together. Treat HDS as a functional parity path, not
the high-throughput gate: Holoscan's current configs leave
`split_boundary:false`, and its HDS code consumes one RX queue with two
segments rather than a separate two-queue header/payload pairing.

IGX production RX starts from `configs/stem_rx_igx_production.yaml`:

- RX `master_core: 8`, RX poll `cpu_core: 9`.
- Verified loopback NICs are TX `0005:03:00.0` and RX `0005:03:00.1`;
  production FPGA PF and flow match are per box.
- Boot-reserved hugepages are 3 x 1 GiB; mount `/dev/hugepages` in the
  container.
- `writer.noop:true` is the production default. The HDF5 sink is smoke/debug
  only and must not be placed on the poll loop.
- RX uses a bounded leased output-buffer pool and counts `sink pool drops`
  instead of blocking when the pool is exhausted.
- Multi-receiver DAQIRI output is valid only with `writer.noop:true`; shared
  HDF5 output is refused because streams would interleave by arrival order.
- IGX production keeps `frames_per_tensor:16` intentionally for RX-only/noop
  latency and bounded buffering. Use a 128-frame config for reduced
  `/processed` comparisons.
- Latency percentiles are `system_clock(now) - epoch_us`; absolute values
  require synchronized TX/RX clocks, and negative-skew samples are dropped.

On the Spark bench:

- Hugepages: `sudo sysctl -w vm.nr_hugepages=2048` for 4 GiB.
- Use PCIe NIC `0002:01:00.0`, not `0000:01:00.0`.
- Memory regions must be `kind: host_pinned`.
- Keep MTU greater than or equal to the 7786 B packet size.
- Containers need `--privileged --network host --gpus all --ulimit memlock=-1`
  with `/dev/hugepages` mounted.

## Conventions

- Do not revert unrelated user changes in this shared workspace.
- Prefer existing repo patterns and helper scripts over adding new wrappers.
- Keep `cpp/kernels.cu` and `cpp_daqiri/common/stem_kernels.{cu,h}` frame
  assembly math synchronized.
- `.gitignore` keeps parsed sweep outputs such as
  `cpp_daqiri/benchmarks/results.md` but discards raw per-run logs such as
  `sweep_*/` and `logs_*`; do not commit raw logs.
- Use capability wording in user-facing docs: hello/link check, TX-only,
  TX+RX, parity-capable HDF5 build, live validation, sweep, replay.
  Preserve existing script filenames and legacy image tags where they are
  actual interfaces, but explain what they do.
