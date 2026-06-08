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
`stamp_epoch_us`, `capture_latency`, `subtract_dark`, `apply_valid_pixel_mask`,
`gpu_header_extract`, and the `writer` block.

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

Correctness gate:

```bash
/opt/stem_daqiri/bin/stem_daqiri_rx \
    /opt/stem_daqiri/bin/configs/stem_rx_igx_loopback.yaml \
    --seconds 20 --validate-ramp
```

The ramp validator copies complete assembled uint16 batches back to host and
checks the deterministic TX row ramp byte-for-byte in the assembled columns.
It exits nonzero if no complete batch is checked or if any mismatch is found.
For throughput, leave `--validate-ramp` off.

For an HDF5 smoke run, copy the RX config, set `writer.noop: false`, and run at
a low rate. The HDF5 writer is a debug sink, not the throughput default.

## Two-Spark Setup

Configs:

- RX: `configs/stem_rx_spark.yaml`
- TX: `configs/stem_tx_spark.yaml`

This path remains host-pinned and CPU-header-read by default:

- memory region `kind: "host_pinned"`
- `stem_rx.gpu_header_extract` omitted or false

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
| `configs/stem_rx_igx_production.yaml` | IGX RX-only production config |
| `configs/stem_rx_igx_loopback.yaml` | IGX hardware-loopback RX config |
| `configs/stem_tx_igx_loopback.yaml` | IGX hardware-loopback TX config |
| `configs/stem_{tx,rx}_spark.yaml` | two-Spark configs |
| `scripts/run_igx_loopback.sh` | single-box IGX loopback wrapper |
| `scripts/run_phase*.sh` | Spark smoke and sweep wrappers |
