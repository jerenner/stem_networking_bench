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

Populate the pinned daqiri submodule before building:

```bash
git submodule update --init --recursive third_party/daqiri
git -C third_party/daqiri rev-parse --short HEAD
```

The expected pin after the current update is `3cce706`. If `.git/config` is a
read-only bind mount, use a direct clone instead of `git submodule update`:

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

`DAQIRI_ENGINE` is the current daqiri build knob for engine backends. Older
documentation used `DAQIRI_MGR`; the STEM CMake wrapper still accepts it as a
compatibility alias, but new builds should use `DAQIRI_ENGINE`.

Build the parity-capable STEM daqiri image with TX, RX, and mandatory HDF5
replay/writer/correction-file support:

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    --build-arg STEM_DAQIRI_REQUIRE_HDF5=ON \
    -t stem_daqiri:parity-hdf5 .
```

`STEM_DAQIRI_REQUIRE_HDF5=ON` makes HDF5 replay, writer, and correction-file
support mandatory. For a throughput-only image, omit that build arg and use a
separate local tag such as `stem_daqiri:tx-rx`.

The same RX/TX binaries carry the live-validation and sweep controls via YAML:
`stamp_epoch_us`, `capture_latency`, `gpu_header_extract`, `hds`, the
Holoscan-compatible top-level `processor` block, and the `writer` block.
Legacy `stem_rx.subtract_dark` and `stem_rx.apply_valid_pixel_mask` are still
accepted as deprecated aliases when `processor` is absent.

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
    stem_daqiri:parity-hdf5 \
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

## DAQIRI PCAP Capture

DAQIRI also ships a raw packet capture utility, independent of
`stem_daqiri_rx`. The upstream DAQIRI docs describe
`daqiri_example_pcap_writer` as an RX-first pcap writer: it receives DAQIRI raw
bursts, writes a classic Ethernet `.pcap`, and closes the file so tools such as
`tcpdump -r` or Wireshark can read it. The command shape is:

```bash
daqiri_example_pcap_writer <pcap-yaml> <output.pcap> [--tx]
```

Use `--tx` only for DAQIRI's self-contained demo transmitter. For tcpdump-like
capture from the FPGA or an external sender, omit `--tx`; if the YAML still
contains `bench_tx`, the example prints that the transmitter is disabled.
Remove any unused TX interface placeholders from an RX-only YAML unless they
have been filled with valid values.

In the STEM DAQIRI container, the utility comes from the base DAQIRI install:

```bash
docker run --rm -it \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount type=bind,source=/dev/hugepages,target=/dev/hugepages \
    --mount type=bind,source=/tmp,target=/capture \
    --mount type=bind,source="$PWD/cpp_daqiri/configs/pcap_capture.yaml",target=/cfg/pcap_capture.yaml,readonly \
    stem_daqiri:tx-rx \
    timeout --signal=INT --preserve-status 10s \
    /opt/daqiri/bin/daqiri_example_pcap_writer \
        /cfg/pcap_capture.yaml \
        /capture/stem-fpga.pcap
```

The `timeout --signal=INT --preserve-status 10s` wrapper bounds the capture and
lets the example close the pcap cleanly; omit it for an interactive capture
that runs until Ctrl+C. The example mounts host `/tmp` for a short smoke
capture; substitute a fast NVMe-backed directory for longer captures.

Use a pcap-specific DAQIRI YAML, not `stem_rx_*.yaml`; the pcap writer expects
`bench_rx`. Start from DAQIRI's
`third_party/daqiri/examples/daqiri_example_pcap_writer_tx_rx.yaml` or the
installed `/opt/daqiri/bin/daqiri_example_pcap_writer_tx_rx.yaml`, then adapt it
for STEM traffic:

- Set `bench_rx.interface_name` to the RX interface.
- For RX-only capture, remove `bench_tx` and any unused TX-only interface or
  replace all TX placeholders with valid loopback values.
- Set the RX interface PCIe address, queue core, and flow match for the FPGA
  source and UDP port.
- Set the RX memory-region `buf_size` to at least the 7786 B STEM packet size;
  use 8064 to match current STEM configs, or 8192 only after native tile payloads
  are confirmed.
- Mount the output directory into the container. A `--rm` container loses an
  unmounted `/tmp` capture when it exits.

The pcap writer is a diagnostic capture path, not the production frame pipeline:
it records raw packets before STEM assembly, does not apply dark/mask processing,
and may be limited by device-to-host copy bandwidth or storage I/O.

To validate the capture path with this repository's STEM TX, run the pcap
writer detached, wait for the RX core, then send a short loopback burst:

```bash
PCAP=/tmp/stem-tx-loopback.pcap
docker rm -f stem_pcap_rx >/dev/null 2>&1 || true

docker run -d --name stem_pcap_rx \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount type=bind,source=/dev/hugepages,target=/dev/hugepages \
    --mount type=bind,source=/tmp,target=/capture \
    --mount type=bind,source="$PWD/cpp_daqiri/configs/pcap_capture.yaml",target=/cfg/pcap_capture.yaml,readonly \
    stem_daqiri:tx-rx \
    timeout --signal=INT --preserve-status 20s \
    /opt/daqiri/bin/daqiri_example_pcap_writer \
        /cfg/pcap_capture.yaml \
        "/capture/$(basename "$PCAP")"

for _ in $(seq 1 60); do
    docker logs stem_pcap_rx 2>&1 | grep -q "Starting RX Core" && break
    sleep 1
done

docker run --rm \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount type=bind,source=/dev/hugepages,target=/dev/hugepages \
    --mount type=bind,source="$PWD/cpp_daqiri/configs/stem_tx_igx_loopback.yaml",target=/cfg/stem_tx_igx_loopback.yaml,readonly \
    stem_daqiri:tx-rx \
    /opt/stem_daqiri/bin/stem_daqiri_tx \
        /cfg/stem_tx_igx_loopback.yaml \
        --seconds 0.1 \
        --rate 1

docker wait stem_pcap_rx
tcpdump -nn -r "$PCAP" -c 3 -e -vv
docker rm stem_pcap_rx
```

The captured packets should be Ethernet/IPv4/UDP frames from
`48:b0:2d:f4:04:23` to `48:b0:2d:f4:04:24`, UDP `4096 -> 4096`, with Ethernet
length 7786 and IPv4 length 7772.

For repeatable validation, use the DAQIRI validation script:

```bash
# Deterministic replay/processor/writer suite; does not need live NICs.
cpp_daqiri/scripts/run_daqiri_validation.sh hdf5

# Parser and failure-mode checks; does not need live NICs.
cpp_daqiri/scripts/run_daqiri_validation.sh config

# Live IGX loopback tests against a running stem_daqiri_live container.
cpp_daqiri/scripts/run_daqiri_validation.sh live-smoke  # non-HDS, 1 Gbps
cpp_daqiri/scripts/run_daqiri_validation.sh live-wire   # non-HDS, unbounded TX
cpp_daqiri/scripts/run_daqiri_validation.sh hds-smoke   # HDS, 1 Gbps
cpp_daqiri/scripts/run_daqiri_validation.sh hds-wire    # HDS, unbounded stress
cpp_daqiri/scripts/run_daqiri_validation.sh live-writer # writer.noop:false
cpp_daqiri/scripts/run_daqiri_validation.sh live-pixel  # non-HDS/HDS HDF5 compare
```

Latest local validation after the daqiri pin update to `3cce706`:

- `hdf5`, `config`, and `live-all` passed against `stem_daqiri:parity-hdf5`.
- Non-HDS unbounded 30-minute soak passed: 1800 s TX, 22.4 TB sent,
  99.632 Gbps TX, 2,811,664 frames assembled, zero DPDK missed, error, or
  out-of-buffer counters, and zero sink drops/errors.
- HDS has passed the scripted short stress gate. A long HDS soak and a fresh
  Holoscan end-to-end comparison are still separate follow-up tests.

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
    stem_daqiri:parity-hdf5 \
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

For a long non-HDS soak, keep RX alive longer than TX so it can drain and print
the final summary. This reproduces the current 30-minute coverage shape:

```bash
docker exec -d stem_daqiri_live bash -lc \
  'export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-};
   rm -f /tmp/stem_daqiri_soak_nonhds.log /tmp/stem_daqiri_soak_nonhds.rc;
   /opt/stem_daqiri/bin/stem_daqiri_rx \
     /opt/stem_daqiri/bin/configs/stem_rx_igx_loopback.yaml \
     --seconds 1860 > /tmp/stem_daqiri_soak_nonhds.log 2>&1;
   echo $? > /tmp/stem_daqiri_soak_nonhds.rc'

cpp_daqiri/scripts/daqiri_docker_exec.sh \
  /opt/stem_daqiri/bin/stem_daqiri_tx \
  /opt/stem_daqiri/bin/configs/stem_tx_igx_loopback.yaml \
  --seconds 1800 --rate 0

cpp_daqiri/scripts/daqiri_docker_exec.sh \
  bash -lc 'cat /tmp/stem_daqiri_soak_nonhds.rc;
            tail -80 /tmp/stem_daqiri_soak_nonhds.log'
```

Manual RX-first launch:

```bash
docker rm -f stem_igx_rx 2>/dev/null || true

docker run -d --name stem_igx_rx \
    --privileged --network host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    stem_daqiri:parity-hdf5 \
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
    stem_daqiri:parity-hdf5 \
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
pixel-level gate. The HDF5 replay path accepts `uint16` or `float32`
`[frames,H,W]` input datasets. Exact comparisons are strongest for
deterministic replay or `processor.noop:true` uint16 gather output. Reduced
float `/processed` output from DAQIRI vs Holoscan should use relative tolerance
because DAQIRI's CUDA sum order is not bit-identical to `torch::sum(0)`:

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
`run_daqiri_validation.sh hds-wire` is a stress coverage command: it still
requires HDS layout verification, frames assembled, and zero sink drops/errors,
but DPDK missed/out-of-buffer counters are reported rather than treated as a
zero-drop throughput gate.
Multi-receiver DAQIRI output is currently allowed only with `writer.noop:true`;
HDF5 output is refused because receiver streams would otherwise interleave by
arrival order in one dataset.

DAQIRI HDF5 replay is finite-only and intentionally rejects
`replayer.repeat:true`. Holoscan can repeat/wrap HDF5 replay input, but parity
replay runs should use `repeat:false` on both sides.

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

Start RX on `spark-201a` with the TX+RX Spark wrapper:

```bash
cpp_daqiri/scripts/run_spark_daqiri_rx.sh --seconds 14
```

Start TX on `spark-960b` with the matching wrapper:

```bash
cpp_daqiri/scripts/run_spark_daqiri_tx_for_rx.sh --seconds 10 --rate 50
```

For the Spark parity sweep:

```bash
cpp_daqiri/scripts/run_spark_parity_sweep_orchestrated.sh \
    --rates "10 25 50 80" --runs 1 --seconds 8 \
    --outdir cpp_daqiri/benchmarks/sweep_<utc>
```

Latency percentiles are computed as RX `system_clock(now) - epoch_us` from TX
headers. Absolute latency requires synchronized TX/RX clocks; otherwise values
include clock offset, and samples with negative skew are dropped.

Parse sweep logs into `cpp_daqiri/benchmarks/results.md`:

```bash
cpp_daqiri/scripts/parse_spark_parity_results.py \
    --daqiri-tx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --daqiri-rx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --holoscan-tx-dir cpp_daqiri/benchmarks/logs_tx_<utc> \
    --holoscan-rx-dir cpp_daqiri/benchmarks/logs_rx_<utc> \
    --duration 8
```

## Files

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Build options for hello/link-check, TX, RX, and optional HDF5 linkage |
| `common/stem_packet.h` | STEM wire layout and frame geometry |
| `common/stem_kernels.{cu,h}` | TX header update, RX header extract, gather, processor kernels |
| `tx/stem_tx_main.cpp` | paced STEM TX |
| `rx/stem_rx_main.cpp` | daqiri RX, frame assembly, output sink |
| `scripts/compare_h5_outputs.py` | pixel-level HDF5 parity comparator |
| `configs/stem_rx_igx_production.yaml` | IGX RX-only production config |
| `configs/stem_rx_igx_loopback.yaml` | IGX hardware-loopback RX config |
| `configs/stem_rx_igx_loopback_hds.yaml` | IGX hardware-loopback RX config with HDS |
| `configs/stem_replay_hdf5.yaml` | finite uint16/float32 HDF5 replay config for processor parity |
| `scripts/run_daqiri_validation.sh` | repeatable HDF5, config, live, writer, and HDS validation gates |
| `configs/stem_tx_igx_loopback.yaml` | IGX hardware-loopback TX config |
| `configs/stem_{tx,rx}_spark.yaml` | two-Spark configs |
| `configs/stem_rx_spark_hds.yaml` | two-Spark RX config with HDS |
| `scripts/run_igx_loopback.sh` | single-box IGX loopback wrapper |
| `scripts/run_spark_*.sh` | Spark TX-only, TX+RX, Holoscan RX validation, and parity-sweep wrappers |
