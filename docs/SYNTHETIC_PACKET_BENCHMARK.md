# Synthetic packet benchmark

The DAQIRI receiver supports `source: synthetic` for measuring whether one GPU
can assemble and process the equivalent of one through eight independent
100-Gbit/s detector streams. It does not initialize DAQIRI, DPDK, a NIC, or
hugepages.

## What it exercises

Each logical receiver owns an immutable, GPU-resident 128-frame packet cycle.
The pool contains complete Ethernet II, IPv4, UDP, 64-byte STEM-header, and
payload bytes. During a timed run, pointers to these packets enter the same GPU
header extraction, host admission/windowing, tile placement, frame gather,
correction, reduction, burst writer, and thinned-stream code used by live
device-memory DAQIRI packets. Receivers have independent CUDA streams and
frame assemblers but share the normal processing configuration and sinks.

The default native-tile layout is:

- 8 source IDs and 120 packets per source, or 960 packets per frame
- 4,096 `uint16` samples (8,192 payload bytes) per packet
- 42 bytes Ethernet/IPv4/UDP plus a 64-byte STEM header
- 8,298 captured bytes per packet, excluding Ethernet FCS, preamble, and
  inter-packet gap
- 128 frames per processing bucket

`num_receivers` selects the number of independent logical input streams from 1
through 8. It does not split a frame across receivers. `expected_source_mask`
still selects the active source IDs within every receiver.

The synthetic packet pool is generated once before a common start barrier, so
initial allocation and payload generation are not included in the throughput
clock. Packets are then reused every 128 frames, matching the 16-bit row-number
wrap in the current STEM header. The deterministic `ramp` and `walking_dot`
patterns depend on receiver, frame, tile, and sample position, which makes
cross-receiver and placement mistakes detectable.

With all eight source IDs, each native packet pool is approximately 0.95 GiB.
Each 128-frame assembled `uint16` slot is another 0.94 GiB, and the corrected
`float32` slot is 1.88 GiB. The full correction benchmark with noop outputs is
therefore about 3.8 GiB per receiver before shared calibration and small
scratch allocations, or roughly 30 GiB for eight receivers. Continuous and
burst writers allocate additional slots/buffers; keep them disabled for the
initial GPU-capacity run and enable them in separate output-throughput tests.

## Configuration

Start from `cpp_daqiri/configs/stem_rx_synthetic.yaml`. The central controls
are:

```yaml
source: "synthetic"
num_receivers: 8

synthetic:
  mode: "packet"
  rate_mode: "limited"
  target_gbps_per_receiver: 100.0
  duration_seconds: 300.0
  buckets_per_receiver: 0
  packets_per_burst: 16384
  payload_pattern: "ramp"
  validate_output: false
  report_interval_seconds: 1.0

stem_rx:
  frames_per_tensor: 128
  expected_source_mask: 255
  header_size: 42
  payload_size: 8192
  gpu_header_extract: true
  tile_duplicate_prefix_to_simulate_payload: false
```

`rate_mode: limited` maintains one virtual producer clock per receiver. The
clock never resets when processing falls behind. `final_lag_ms` therefore
shows accumulated schedule debt rather than allowing backpressure to silently
slow the source. At eight receivers and 100 Gbit/s each, `target_Gbps` in the
aggregate summary is 800.

`rate_mode: maximum` removes pacing and measures the application's maximum
completed throughput. Use both modes: a limited run answers whether the chain
keeps up with a requested rate, while a maximum run measures available margin.

`buckets_per_receiver` is an optional deterministic stop condition. A value of
zero disables it. When both a duration and bucket count are enabled, the first
condition reached stops that receiver. `--seconds N` overrides
`duration_seconds` as it does for a network run.

`packets_per_burst` models the packet-pointer batch delivered to the receiver.
It may span at most 32 frames and is automatically clipped at a processing
bucket boundary. It affects host launch/admission overhead, so use the DAQIRI
burst size expected in the deployment when comparing systems.

Per-receiver rate overrides are optional:

```yaml
receiver0:
  target_gbps: 95.0
receiver1:
  target_gbps: 100.0
```

All receivers currently execute on CUDA device 0. To select a physical GPU,
expose it as device 0 with `CUDA_VISIBLE_DEVICES` or Docker's GPU device
selection. Multi-GPU distribution is intentionally a separate topology.

To exercise the old 3,840-sample row-packet compatibility path, use
`payload_size: 7680` and
`tile_duplicate_prefix_to_simulate_payload: true`. The generator emits 128
row offsets per active source, the receiver ignores offsets 120 through 127,
and the tile gather duplicates the first 256 samples to construct 4,096-sample
tiles. This mode is for parity testing, not the proposed native wire format.

## Running on a GPU system

The existing RX image contains the source. No privileged networking, host
networking, or hugepage mount is required unless ZeroMQ outputs are enabled:

```bash
docker run --rm -it \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PWD/cpp_daqiri/configs/stem_rx_synthetic.yaml:/run/stem.yaml:ro" \
    stem_daqiri:burst-zmq \
    /opt/stem_daqiri/bin/stem_daqiri_rx /run/stem.yaml
```

Mount the dark-frame file at the path in `processor.dark_frame_path` when
corrections are enabled. Mount `/data` when either HDF5 writer is enabled. Add
`--network host` only when exposing the control or thinned-stream ZeroMQ
endpoints.

For an 800-Gbit/s processing test, set `num_receivers: 8`, enable the exact
production processor operations, leave all writers/streams disabled initially,
and run for at least 300 seconds. Run once at `limited` 100 Gbit/s per receiver
and once at `maximum`. A successful keep-up run has:

- `completed_Gbps` close to the requested aggregate rate after startup
- `final_lag_ms` and `max_lag_ms` bounded rather than growing continuously
- zero `validation_mismatches`, incomplete batches, output-pool drops, and
  unexpected packets
- no CUDA allocation or processing errors

`packet_Gbps` includes every synthetic packet submitted to assembly.
`completed_Gbps` counts only packets belonging to completed frame tensors and
is the more useful processing-throughput number. A time-limited run can end
with `trailing_packets` from a partial, deliberately un-emitted bucket; use a
finite bucket count when an exact zero-tail test is required.

`validate_output: true` runs a full pixel-by-pixel GPU check after assembly.
It is a correctness mode and adds a complete frame-memory pass, so leave it
off for reported performance measurements.

## Validation

Host-only layout tests require only a C++17 compiler:

```bash
cmake -S cpp_daqiri/tests -B /tmp/stem-synthetic-tests
cmake --build /tmp/stem-synthetic-tests
ctest --test-dir /tmp/stem-synthetic-tests --output-on-failure
```

The installed GPU integration validator runs native, sparse/legacy,
walking-dot, corrected-output, eight-receiver, and counter-wrap cases through
the real RX binary. It also writes small HDF5 captures and compares every pixel
with an independent NumPy reference:

```bash
python3 /opt/stem_daqiri/bin/validate_synthetic_rx.py \
    --binary /opt/stem_daqiri/bin/stem_daqiri_rx
```

Use `--workdir /data/synthetic-validation` to retain generated configurations
and logs. The integration validator requires `numpy`, `h5py`, and `pyyaml` in
the Python environment.

## Scope and limitations

This is an application/GPU capacity test, not a replacement for an eight-NIC
test. It does not measure wire loss, RX CPU polling, PCIe ingress, GPUDirect
RDMA, NIC-to-GPU topology, DPDK metadata pressure, switch behavior, or Ethernet
overhead outside the captured packet. Because packets already reside in VRAM,
it also does not reproduce concurrent NIC writes competing for GPU memory
bandwidth. Final hardware qualification must add live network tests after this
benchmark establishes sufficient assembly and processing margin.
The 128-frame packet reuse can also produce more favorable cache behavior than
continuously changing NIC buffers, although the approximately 0.95-GiB pool per
receiver is much larger than GPU cache. Treat the result as a compute-capacity
gate, not as proof of end-to-end 800-Gbit/s acquisition.
