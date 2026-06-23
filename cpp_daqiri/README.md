# stem_daqiri

daqiri-based STEM networking pipeline for `stem_networking_bench`.

The code supports three topologies from one TX/RX implementation:

1. RX-only production integration on IGX Orin + RTX 6000 Ada, fed by an FPGA.
2. Single-box IGX hardware loopback, TX port cabled to RX port.
3. Two DGX Spark nodes connected back-to-back.

The STEM wire format is shared with the Holoscan implementation in `../cpp/`:
one 7786 B UDP packet carries one 1024-frame row payload:
42 B Ethernet/IPv4/UDP + 64 B STEM header + 7680 B row data.

## Build

Populate daqiri before building. If `.git/config` is a read-only bind mount,
use a direct clone instead of `git submodule update --init`:

```bash
rmdir third_party/daqiri 2>/dev/null || true
git clone --recursive https://github.com/NVIDIA/daqiri.git third_party/daqiri
git -C third_party/daqiri checkout 8c5d69fa3c9bf9e57f6625114b5f0828bb592729
git -C third_party/daqiri submodule update --init --recursive
```

Build the daqiri base image once on the target machine:

```bash
cd third_party/daqiri
IMAGE_TAG=daqiri-torch:local BASE_IMAGE=torch BASE_TARGET=dpdk \
    DAQIRI_MGR="dpdk" scripts/build-container.sh
cd ../..
```

Build the STEM daqiri image:

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    -t stem_daqiri:phase2 .
```

The same Phase 2 binary carries the Phase 3 controls via YAML:
`stamp_epoch_us`, `capture_latency`, `gpu_header_extract`, `hds`, the
Holoscan-compatible top-level `processor` block, and the `writer` block.
Legacy `stem_rx.subtract_dark` and `stem_rx.apply_valid_pixel_mask` are still
accepted as deprecated aliases when `processor` is absent.
Use `-DSTEM_DAQIRI_REQUIRE_HDF5=ON` for parity-capable builds that must include
HDF5 writer/replay/correction-file support; configure fails if HDF5 is missing.

## RX-Only Production

Config: `configs/stem_rx_igx_production.yaml`

Target: IGX Orin with discrete RTX 6000 Ada dGPU, RX only, real FPGA source.
This is the primary deployment. RX packet buffers are device VRAM:

```yaml
memory_regions:
- kind: "device"
  num_bufs: 262144
  buf_size: 8064

stem_rx:
  gpu_header_extract: true
  total_time_to_recv: -1.0

writer:
  noop: true
```

`stem_rx.frames_per_tensor: 16` is intentional in the IGX production config:
this path is RX-only/noop and tuned for lower latency and bounded in-flight
buffering. It is not comparable to 128-frame reduced-output parity baselines.

Set the production PCIe PF and flow match per box:

- `interfaces[0].address`: FPGA-facing PF.
- `flows[0].match.udp_dst`: FPGA UDP destination port.
- `expected_source_mask`: active FPGA sources.

Core assignment on the verified IGX box:

- RX `master_core: 8`
- RX poll queue `cpu_core: 9`
- isolated cores are `9-11`; keep DPDK pollers there.

Run RX in the project container:

```bash
docker run --rm -it \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    stem_daqiri:phase2 \
    /opt/stem_daqiri/bin/stem_daqiri_rx \
    /opt/stem_daqiri/bin/configs/stem_rx_igx_production.yaml
```

The output sink is deliberately nonblocking. `FrameAssembler` gathers into a
pool of assembled GPU buffers; a writer thread receives a lease for a completed
slot. If no slot is free, RX advances the window and increments `sink pool drops`
instead of blocking the DPDK poll loop. `writer.noop: true` is the production
default; the HDF5 writer is only a smoke/debug sink.

## IGX HW Loopback

Configs:

- RX: `configs/stem_rx_igx_loopback.yaml`
- TX: `configs/stem_tx_igx_loopback.yaml`

Verified hardware values for the local IGX:

- TX PF: `0005:03:00.0`, MAC `48:b0:2d:f4:04:23`
- RX PF: `0005:03:00.1`, MAC `48:b0:2d:f4:04:24`
- TX poll core: `10`, TX master core: `7`
- RX poll core: `9`, RX master core: `8`
- Hugepages: 3 x 1 GiB boot-reserved
- `buf_size: 8064`
- `kind: "device"` with `stem_rx.gpu_header_extract: true`
- TX uses `update_headers_per_burst: true`, so device-memory header writes run
  through the CUDA kernel.

Connect `0005:03:00.0` directly to `0005:03:00.1`, then run:

```bash
cpp_daqiri/scripts/run_igx_loopback.sh --rate 20 --tx-seconds 10 --rx-seconds 60
```

The loopback RX config defaults to `writer.noop: true` and leaves the optional
dark/mask processor off so this command measures RX/TX throughput without disk
I/O or extra processing on the hot path.

For repeatable validation, use the DAQIRI validation script:

```bash
# Deterministic replay/processor/writer suite; does not need live NICs.
cpp_daqiri/scripts/run_daqiri_validation.sh hdf5

# Live IGX loopback tests against a running stem_daqiri_live container.
cpp_daqiri/scripts/run_daqiri_validation.sh live-smoke  # non-HDS, 1 Gbps
cpp_daqiri/scripts/run_daqiri_validation.sh live-wire   # non-HDS, unbounded TX
cpp_daqiri/scripts/run_daqiri_validation.sh hds-smoke   # HDS, 1 Gbps
```

If you prefer a long-running container for manual `docker exec` testing,
launch it from the repo root:

```bash
docker rm -f stem_daqiri_live >/dev/null 2>&1 || true

docker run -d --name stem_daqiri_live \
    --privileged --network host --ipc=host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    -v /tmp:/tmp \
    -v "$PWD":/workspace/stem \
    -w /workspace/stem \
    stem_daqiri:phase3-hdf5 \
    sleep infinity
```

`docker exec` does not rerun NVIDIA's entrypoint setup, so exec sessions must
prepend CUDA's forward-compat library directory or CUDA calls can fail with
`CUDA driver version is insufficient for CUDA runtime version`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}
```

The helper wrapper applies that automatically:

```bash
cpp_daqiri/scripts/daqiri_docker_exec.sh \
    stem_daqiri_hello --self-test
```

Manual RX-first launch:

```bash
docker rm -f stem_igx_rx 2>/dev/null || true

docker run -d --name stem_igx_rx \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    stem_daqiri:phase2 \
    /opt/stem_daqiri/bin/stem_daqiri_rx \
    /opt/stem_daqiri/bin/configs/stem_rx_igx_loopback.yaml \
    --seconds 60
```

Then TX:

```bash
docker run --rm \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    stem_daqiri:phase2 \
    /opt/stem_daqiri/bin/stem_daqiri_tx \
    /opt/stem_daqiri/bin/configs/stem_tx_igx_loopback.yaml \
    --seconds 10 \
    --rate 20
```

RX assembly is now tile-only (`gather_tile_packets_by_placement`); the legacy
row-based gather and its `--validate-ramp` correctness gate were removed
because LBNL's FPGA cannot emit row-shaped payloads. The test TX still emits
128 row packets/source; the RX drops `row_offset >= 120` as `tile dropped pkts`
and fills the remaining 256 tile samples by wrapping the payload prefix
(`tile_duplicate_prefix_to_simulate_payload: true`). Keep that knob true until
the real FPGA wire payload is confirmed. If the FPGA emits native 4096-sample
tile payloads, update the RX buffer sizes, MTU/packet-size notes, and
`ipv4_len` together before setting it false.

For an HDF5 smoke run, copy the RX config, set `writer.noop: false`, and run at
a low rate. The HDF5 writer is a debug sink, not the throughput default.
Use `scripts/compare_h5_outputs.py` to make the HDS/non-HDS parity check a
pixel-level gate. Exact comparisons are strongest for deterministic replay or
`processor.noop:true` uint16 gather output. Reduced float `/processed` output
from DAQIRI vs Holoscan should use relative tolerance because DAQIRI's CUDA
sum order is not bit-identical to `torch::sum(0)`:

```bash
python3 cpp_daqiri/scripts/compare_h5_outputs.py \
    /tmp/stem_rx_igx_loopback.h5 \
    /tmp/stem_rx_igx_loopback_hds.h5 \
    --dataset /processed

python3 cpp_daqiri/scripts/compare_h5_outputs.py \
    /tmp/holoscan_replay_out.h5 \
    /tmp/daqiri_replay_out.h5 \
    --dataset /processed --max-frames 128 --rtol 1e-5
```

`--max-frames` is a deterministic replay-prefix tool. Do not treat a prefix
PASS between two independent live network captures as meaningful unless frame
alignment is independently guaranteed.

The HDS configs are for functional parity, not for the 95 Gbps throughput gate.
Holoscan's checked-in configs leave `split_boundary: false`, and the Holoscan
HDS code consumes one RX queue with two segments rather than a separate
two-queue header/payload pairing. Do not add a DAQIRI-only two-queue HDS path
unless Holoscan grows an equivalent path to compare against.
Multi-receiver DAQIRI output is currently allowed only with `writer.noop:true`;
HDF5 output is refused because receiver streams would otherwise interleave by
arrival order in one dataset.

## Two-Spark Setup

Configs:

- RX: `configs/stem_rx_spark.yaml`
- RX HDS: `configs/stem_rx_spark_hds.yaml`
- TX: `configs/stem_tx_spark.yaml`

This path remains host-pinned and CPU-header-read by default:

- memory region `kind: "host_pinned"`
- `stem_rx.gpu_header_extract` omitted or false
- HDS variant uses separate host-pinned header/payload regions and verifies
  the observed 106/7680 split on the first non-empty burst

Topology:

| Logical name | Hostname | Role | PCIe NIC |
| --- | --- | --- | --- |
| spark-stacked-01 | `spark-960b` | TX | `0002:01:00.0` |
| spark-stacked-02 | `spark-201a` | RX | `0002:01:00.0` |

Use `0002:01:00.0`, not the kernel-managed `0000:01:00.0` link-local NIC.
The Spark RX pool uses 262144 buffers of 8064 B, so allocate 4 GiB of 2 MiB
hugepages before running:

```bash
sudo sysctl -w vm.nr_hugepages=2048
```

Start RX on `spark-201a`:

```bash
cpp_daqiri/scripts/run_phase2_rx.sh --seconds 14
```

Start TX on `spark-960b`:

```bash
cpp_daqiri/scripts/run_phase2_tx.sh --seconds 10 --rate 50
```

For the Phase 3 parity sweep:

```bash
cpp_daqiri/scripts/run_phase3_sweep_orchestrated.sh \
    --rates "10 25 50 80" --runs 1 --seconds 8 \
    --outdir cpp_daqiri/benchmarks/sweep_<utc>
```

Latency percentiles are computed as RX `system_clock(now) - epoch_us` from TX
headers. Absolute latency requires synchronized TX/RX clocks; otherwise values
include clock offset, and samples with negative skew are dropped.

Parse sweep logs into `cpp_daqiri/benchmarks/results.md`:

```bash
cpp_daqiri/scripts/parse_phase3_results.py \
    --daqiri-tx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --daqiri-rx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --holoscan-tx-dir cpp_daqiri/benchmarks/logs_tx_<utc> \
    --holoscan-rx-dir cpp_daqiri/benchmarks/logs_rx_<utc> \
    --duration 8
```

## Files

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Phased build and optional HDF5 sink linkage |
| `common/stem_packet.h` | STEM wire layout and frame geometry |
| `common/stem_kernels.{cu,h}` | TX header update, RX header extract, gather, processor kernels |
| `tx/stem_tx_main.cpp` | paced STEM TX |
| `rx/stem_rx_main.cpp` | daqiri RX, frame assembly, output sink |
| `scripts/compare_h5_outputs.py` | pixel-level HDF5 parity comparator |
| `configs/stem_rx_igx_production.yaml` | IGX RX-only production config |
| `configs/stem_rx_igx_loopback.yaml` | IGX hardware-loopback RX config |
| `configs/stem_rx_igx_loopback_hds.yaml` | IGX hardware-loopback RX config with HDS |
| `configs/stem_replay_hdf5.yaml` | finite uint16 HDF5 replay config for processor parity |
| `configs/stem_tx_igx_loopback.yaml` | IGX hardware-loopback TX config |
| `configs/stem_{tx,rx}_spark.yaml` | two-Spark configs |
| `configs/stem_rx_spark_hds.yaml` | two-Spark RX config with HDS |
| `scripts/run_igx_loopback.sh` | single-box IGX loopback wrapper |
| `scripts/run_phase*.sh` | Spark smoke and sweep wrappers |
