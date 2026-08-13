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

Build the parity-capable STEM daqiri image with TX, RX, mandatory HDF5
replay/writer/correction-file support, and ZeroMQ output/control support:

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    --build-arg STEM_DAQIRI_REQUIRE_HDF5=ON \
    --build-arg STEM_DAQIRI_REQUIRE_ZMQ=ON \
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

The processor uses the same fused correction order as the Holoscan `tiling`
path for both network `uint16` and HDF5 replay `float32` input: optional
dark-aware grouped BLR estimation, fused conversion/dark/BLR correction and
batch mean, combined valid-pixel and two-sided dynamic masking with excluded
edge rows, then optional frame reduction.

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

### Dual FPGA RX

Config: `configs/stem_rx_igx_fpga_dual.yaml`

This configuration receives independent frame streams on PCI functions
`0005:03:00.0` and `0005:03:00.1`. Each interface owns its own frame assembler;
completed batches join only at the shared GPU processor and output sink. The
four-source test mask (`0x0f`) fills 480 of 960 tiles per frame. Change it to
`0xff` when all eight source IDs are active.

The throughput default is `writer.noop:true`. With `writer.noop:false`, the
single sink thread serializes batches from both receivers into one HDF5 dataset
in arrival order. Receiver identity is not encoded in that dataset, matching
the previous Holoscan shared-writer behavior.

```bash
docker run --rm -it \
    --privileged --network host --ipc=host --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /dev/hugepages:/dev/hugepages \
    -v /tmp:/tmp \
    stem_daqiri:parity-hdf5 \
    /opt/stem_daqiri/bin/stem_daqiri_rx \
    /opt/stem_daqiri/bin/configs/stem_rx_igx_fpga_dual.yaml
```

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

### Live FPGA platform profile

Use `scripts/run_igx_fpga_profile.sh` on the IGX host to run the dual-FPGA
receiver and `tegrastats` over the same timed interval. The default duration is
five minutes and the default platform sampling interval is one second:

```bash
cpp_daqiri/scripts/run_igx_fpga_profile.sh \
  --config cpp_daqiri/configs/stem_rx_igx_fpga_dual.yaml \
  --dark /absolute/path/to/walking_dot_dark_frame.h5 \
  --seconds 300 \
  --output-root /absolute/path/to/daqiri_profiles
```

The script creates a unique timestamped directory containing the exact YAML,
its hash and Git commit, timestamped DAQIRI output, raw `tegrastats` output,
a tab-separated platform timeline, and synchronized `nvidia-smi` samples when
the discrete GPU is visible. `tegrastats` captures Orin CPU, memory-controller,
power, temperature, and integrated-GPU data; `nvidia-smi` captures utilization,
memory, clocks, temperature, and power for the RTX GPU used by CUDA. The
script starts both samplers one second before the container to capture a
baseline and records the start times in `run_metadata.txt`. It stops the
samplers and container on normal completion, Ctrl-C, or termination.

The script must run on the IGX host, where `tegrastats`, Docker, the NICs, and
the hugepage mount are available. The dark-frame argument can be omitted only
when the selected YAML does not refer to `/calibration/dark.h5`.

Analyze a completed timestamped profile on a system with Python, pandas, NumPy,
and Matplotlib:

```bash
python cpp_daqiri/scripts/analyze_igx_fpga_profile.py \
  /absolute/path/to/daqiri_profiles/20260726T170603Z
```

The analyzer aligns telemetry to the DAQIRI worker interval and writes an
`analysis` subdirectory containing a keep-up dashboard, synchronized platform
timeline, CPU-core heatmap, Markdown summary, and machine-readable JSON metrics.
The dashboard reports startup NIC-drop events separately from steady-state
pipeline backpressure and labels the compatibility tile discard as intentional.

### Controlled burst acquisition

`burst_writer` captures a configured number of contiguous frame buckets without
turning the continuous HDF5 writer into a source of uncontrolled backpressure:

```yaml
writer:
  noop: true

burst_writer:
  enabled: true
  start_armed: false
  processing_stage: "corrected"
  filepath_template: "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5"
  dataset_name: "/frames"
  buckets_per_capture: 1
  capture_count: 1          # 0 means unlimited
  rearm_after_write: true
  strict_complete: true
  threshold:                # used only for processing_stage: thresholded
    zlp: 0.0
    core_loss: 0.0
```

The `enabled` flag allocates sink-owned GPU and pinned-host buffers at process
startup. `start_armed` chooses the initial state; the control API can arm or
disarm later without allocating memory. With `strict_complete: true`, it
waits for the next complete bucket. It reserves all buffers needed by
`buckets_per_capture`, copies those consecutive buckets into sink-owned GPU
buffers, and writes one HDF5 file. NIC polling, assembly, and GPU processing
continue while the file drains; later buckets are deliberately omitted from
the burst sink and counted as `burst buckets ... skipped busy`. They are not
reported as NIC drops. After writing, the sink rearms unless the capture limit
was reached or `rearm_after_write` is false.

The two HDF5 modes are intentionally mutually exclusive: set `writer.noop` to
true whenever `burst_writer.enabled` is true. For multiple receivers, the
filepath template must include `{receiver}`. Supported substitutions are
`{receiver}`, `{capture}`, `{stage}`, and `{first_frame}`.

Supported stages are:

| Stage | Burst dtype | Definition |
|---|---|---|
| `raw` | input dtype | Assembled frames before detector corrections |
| `dark_subtracted` | `float32` | Raw minus the configured dark frame |
| `dark_blr` | `float32` | Dark subtraction plus grouped BLR, before masks |
| `corrected` | `float32` | Dark + BLR + valid-pixel and dynamic masks |
| `thresholded` | `float32` | Corrected analog values above the configured positive regional threshold; all others zero |

`counted` is reserved for the future STEMPy-style local-maximum event counter
and is rejected rather than being silently approximated by thresholding.

Burst buffers are allocated before DAQIRI starts, avoiding runtime allocation
on the packet path. This is intentionally memory intensive: one 128-frame
1024x3840 bucket uses 960 MiB as `uint16` or 1,920 MiB as `float32`, in both
GPU and pinned host memory, per receiver. Increase `buckets_per_capture` only
after checking the available GPU and host memory.

In the Qt console, **Apply capture settings** changes stage/path/count policy
without starting a capture and is accepted only while the burst sink is idle.
**Apply settings and arm** makes the same update and then waits for the next
eligible bucket sequence. If the state is **NOT ALLOCATED**, enable **Allocate
burst writer** under **Apply on restart** and restart acquisition first; live
controls cannot create the multi-gigabyte startup buffer allocation.

Mount the output directory when running the container:

```bash
-v /absolute/host/burst_output:/data
```

### Thinned ZeroMQ stream

`thinned_stream` publishes display products without retaining or transmitting
the full 128-frame bucket:

```yaml
thinned_stream:
  enabled: true
  start_publishing: true
  endpoint: "tcp://*:5556"
  topic_prefix: "stem"
  total_refresh_hz: 10.0
  processing_stage: "corrected"
  representative_frame_index: 64
  include_representative_frame: true
  include_bucket_sum: true
  queue_depth: 2
  threshold:
    zlp: 0.0
    core_loss: 0.0
```

The refresh rate is global. Receivers are selected round-robin, so two active
receivers at 10 Hz produce approximately 5 messages/s per receiver. A value of
zero publishes every round-robin opportunity and is intended for validation,
not live full-rate operation. The acquisition thread launches one GPU kernel
that extracts the representative frame and accumulates the bucket sum, then
copies only those enabled products to pinned memory. Both are `float32`, even
for raw input, so a 128-frame sum cannot overflow `uint16`.

The publisher has a bounded slot pool and coalesces queued products to the
newest one. A slow or disconnected viewer therefore loses display updates but
cannot backpressure frame assembly. Runtime counters report queued, published,
coalesced, and no-buffer products.

Each publication is a ZeroMQ PUB multipart message:

1. Topic: `stem/rx/<receiver>/<processing_stage>`
2. UTF-8 JSON metadata with schema `stem.thinned.v1`
3. Optional representative `[height,width]` little-endian `float32` bytes
4. Optional bucket-sum `[height,width]` little-endian `float32` bytes

With Docker `--network host`, a remote viewer connects directly to
`tcp://<IGX-management-IP>:5556`. Install `pyzmq` and NumPy on a viewer machine
and inspect the stream with:

```bash
python cpp_daqiri/scripts/inspect_thinned_stream.py \
  --endpoint tcp://192.168.10.42:5556 \
  --topic stem/ \
  --count 10
```

Use `--save-dir PATH` to save metadata and raw float32 arrays for viewer
development. ZeroMQ PUB/SUB is live-only: connect the subscriber before the
desired acquisition if the first products matter.

### Live DAQ control and Qt console

An independent ZeroMQ REP endpoint controls runtime-safe settings and stages
restart-required settings:

```yaml
control:
  enabled: true
  start_acquisition: false
  endpoint: "tcp://*:5557"
  runtime_config_path: "/tmp/stem_daqiri_runtime.yaml"
```

Runtime-safe controls are applied between bucket reservations:

- arm, disarm, or abort controlled burst capture
- change burst filename, dataset, capture count, completeness policy,
  thresholds, and bucket count up to the startup allocation
- enable/disable publication and change its stage, refresh rate, representative
  frame, topic, products, and thresholds

The burst output dtype family is fixed by its startup allocation. Changing
between network raw `uint16` and processed `float32`, increasing the maximum
burst size, enabling an unallocated output, changing ZeroMQ endpoints/queue
depth, processor kernels/calibration, continuous writer settings, or receiver
packet/NIC geometry requires restart.

`stem_daqiri_supervisor` is the persistent DAQ service. It owns the public REP
endpoint and launches `stem_daqiri_rx` as a disposable child with a private IPC
control endpoint. Consequently, the GUI remains connected while acquisition is
stopped or restarting. **Stop acquisition** drains RX and releases DPDK, NIC,
GPU, and auxiliary-output resources without exiting the supervisor; **Start
acquisition** launches a fresh child using the last active configuration.

`stage_restart` merges requested values into a complete pending YAML tree.
`restart` validates that tree and atomically writes `runtime_config_path`; only
then does RX drain outputs, shut down DAQIRI, and exit with status 75. The
supervisor waits for complete child termination before launching a new RX child
with the generated configuration. DAQIRI engine-level NIC/queue changes cannot
be preflighted while DPDK owns the interfaces; those are validated by
`daqiri_init` after relaunch and can still cause the new child to stop with its
configuration error while the supervisor remains available for inspection.

For a persistent, remotely controlled container invocation, run the installed
supervisor instead of the RX binary:

```bash
/opt/stem_daqiri/bin/stem_daqiri_supervisor \
  /run/stem_rx.yaml
```

The supervisor rewrites only the child copy of the YAML so RX binds a private
`ipc:///tmp/stem_daqiri_rx_control.ipc` endpoint; `control.endpoint` remains the
public supervisor address. With `control.start_acquisition: false`, only the
supervisor starts; the GUI launches the first RX child. Set it to `true` to
begin acquisition as soon as the service starts. `STEM_DAQIRI_START_STOPPED=1`
overrides either setting and forces an idle launch. An optional `--seconds 120`
limits each RX child to 120 seconds, but the supervisor stays online afterward;
omit it for acquisition that runs until Stop. Running `stem_daqiri_rx` directly
remains valid for noninteractive tests, but GUI Start/Stop requires the
supervisor.

The PySide6 console combines the SUB viewer with the REP controls:

```bash
python3 -m venv .venv-stem-daq
source .venv-stem-daq/bin/activate
pip install -r cpp_daqiri/gui/requirements.txt
python cpp_daqiri/gui/stem_daq_gui.py \
  --stream-endpoint tcp://127.0.0.1:15556 \
  --control-endpoint tcp://127.0.0.1:15557 \
  --max-render-hz 5
```

`--max-render-hz` limits expensive Qt image redraws without changing the DAQ
publication rate. The viewer retains only the newest received product and only
renders the selected image/profile tab. Changing the publication rate in the
GUI takes effect only after **Apply visualization settings** succeeds through
the control endpoint.

For an IGX reached through `qdaq01`, forward both ports from the local
machine:

```bash
ssh -N \
  -L 15556:igx-daq2:5556 \
  -L 15557:igx-daq2:5557 \
  -i ~/.ssh/qdaq01_key user@qdaq01
```

If `qdaq01` cannot resolve or route `igx-daq2`, use the IGX management IP
in both forwarding targets. The DAQ container must use `--network host`.
The REP protocol is intentionally lightweight and has no built-in
authentication; firewall port 5557 to the management network and normally
access it only through an SSH tunnel.

If DATA is online but CONTROL remains offline and SSH reports `connect failed:
Connection refused`, the PUB endpoint is working but nothing is listening on
IGX port 5557. Confirm that the host configuration mounted into the container
contains the `control:` block above, rebuild/restart with the control-capable
image, and check the host-network listener while RX is running:

```bash
sudo ss -ltnp | grep ':5557'
```

The application startup log also reports the enabled control endpoint. The GUI
backs failed status probes off to avoid accumulating SSH forwarding channels.

Develop and test the GUI without CUDA, DAQIRI, or detector hardware by running
the synthetic server and connecting to its default local ports:

```bash
python cpp_daqiri/gui/mock_stem_daq.py
python cpp_daqiri/gui/stem_daq_gui.py \
  --stream-endpoint tcp://127.0.0.1:5556 \
  --control-endpoint tcp://127.0.0.1:5557
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
Multi-receiver HDF5 output intentionally interleaves complete receiver batches
by arrival order through one writer thread. Use separate runs or files if
receiver provenance must be preserved.

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

## HDF5 replay

The DAQIRI receiver binary can replay finite `uint16` or `float32` HDF5
datasets with shape `[frames,H,W]` through the same `FramePipeline` and fused
CUDA processing kernels used by live network input. The host-side wrapper
generates the container config and bind-mounts all files, so no interactive
container or manual file copies are needed:

```bash
cpp_daqiri/scripts/run_daqiri_hdf5_replay.sh \
    --input /path/to/nio_15pa_spectrum_frames_float32_uncompressed.h5 \
    --dark /path/to/nio_15pa_dark_frame_float32.h5 \
    --output /path/to/nio_15pa_spectrum_processed.h5 \
    --count 256
```

Supplying `--dark` enables dark subtraction, valid-pixel masking, grouped BLR,
and two-sided dynamic masking by default. The input and dark frame must have
the same `H,W` geometry. `processor.noop` remains true unless `--reduce` is
passed, so the default output contains every fully processed input frame.
HDF5 replay validates processing and numerical output, but it is not paced like
a live 100 Gb/s stream and therefore is not a receiver keep-up test.

## Files

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Build options for hello/link-check, TX, RX, and optional HDF5 linkage |
| `common/stem_packet.h` | STEM wire layout and frame geometry |
| `common/stem_kernels.{cu,h}` | TX header update, RX header extract, gather, processor kernels |
| `tx/stem_tx_main.cpp` | paced STEM TX |
| `rx/stem_rx_main.cpp` | daqiri RX, frame assembly, output sink |
| `rx/stem_aux_output.{cpp,h}` | controlled burst and latest-only PUB outputs |
| `rx/stem_control_server.{cpp,h}` | JSON-over-ZeroMQ REP control transport |
| `gui/stem_daq_gui.py` | PySide6 live viewer and DAQ controller |
| `gui/mock_stem_daq.py` | hardware-free synthetic PUB/REP development server |
| `scripts/compare_h5_outputs.py` | pixel-level HDF5 parity comparator |
| `configs/stem_rx_igx_production.yaml` | IGX RX-only production config |
| `configs/stem_rx_igx_fpga_dual.yaml` | Dual-interface IGX FPGA production config |
| `configs/stem_rx_igx_loopback.yaml` | IGX hardware-loopback RX config |
| `configs/stem_rx_igx_loopback_hds.yaml` | IGX hardware-loopback RX config with HDS |
| `configs/stem_replay_hdf5.yaml` | finite uint16/float32 HDF5 replay config for processor parity |
| `scripts/run_daqiri_hdf5_replay.sh` | one-shot Docker wrapper for finite HDF5 processing |
| `scripts/run_daqiri_validation.sh` | repeatable HDF5, config, live, writer, and HDS validation gates |
| `configs/stem_tx_igx_loopback.yaml` | IGX hardware-loopback TX config |
| `configs/stem_{tx,rx}_spark.yaml` | two-Spark configs |
| `configs/stem_rx_spark_hds.yaml` | two-Spark RX config with HDS |
| `scripts/run_igx_loopback.sh` | single-box IGX loopback wrapper |
| `scripts/run_spark_*.sh` | Spark TX-only, TX+RX, Holoscan RX validation, and parity-sweep wrappers |
