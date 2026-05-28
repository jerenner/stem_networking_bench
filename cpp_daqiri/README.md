# stem_daqiri — daqiri-based port of `stem_networking_bench`

Phased port of the Holoscan RX-only pipeline in `../cpp/` to a daqiri-based
TX + RX pipeline across two DGX Spark (GB10) nodes:

| Logical name | Hostname | IP | Role | NIC |
| --- | --- | --- | --- | --- |
| spark-stacked-01 | `spark-960b` | `10.99.99.1`  | TX | `enP2p1s0f0np0` (0002:01:00.0) |
| spark-stacked-02 | `spark-201a` | `10.99.99.2`  | RX | `enP2p1s0f0np0` (0002:01:00.0) |

The two boxes are connected directly by ConnectX-7 NICs. Each Spark has
TWO cabled NIC pairs -- `0000:01:00.0` (the kernel-managed link-local
`169.254.x.x` interface) and `0002:01:00.0` (the canonical daqiri
high-bandwidth path). We use `0002:01:00.0`: putting DPDK on the
kernel-IP-bound primary results in `flow_isolation: true` silently
dropping every packet at the application even though
`rx_packets_phy` counts the inbound traffic at the NIC. See
`benchmarks/results.md` for the full provisioning notes.

Source is on a shared NFS path
(`/srv/nfs/share/users/ccrozier/stem_networking_bench`) but each Spark
builds its own container locally.

## Phases

| Phase | What it builds | Gate |
| --- | --- | --- |
| **0** | `stem_daqiri_hello` (link-check); daqiri default bench config | Container + cmake build pass; hello runs; daqiri default TX/RX runs spark-to-spark |
| **1** | `stem_daqiri_tx` (STEM-format paced TX) + patched Holoscan RX | Container + cmake build pass; TX hits achieved Gbps within 5% of target; existing Holoscan RX assembles frames |
| **2** | `stem_daqiri_rx` (STEM frame assembly, noop processor) | Container + cmake build pass; end-to-end TX → RX, bytes within tolerance |
| **3** | Metrics, dark-correction, parity sweep vs Holoscan | daqiri matches/beats Holoscan on Gbps, drops, p50/p99 latency, and processor fps |


## Build (on each Spark, independently)

Prerequisites already in the daqiri-torch base image: cmake, ninja, g++,
patched DPDK 25.11, CUDA 13.1, daqiri at `/opt/daqiri`.

```bash
# 1) Build the base daqiri image (only needed once per Spark; the prebuilt
#    daqiri-torch:local image was constructed via this command).
cd third_party/daqiri && \
    IMAGE_TAG=daqiri-torch:local BASE_IMAGE=torch BASE_TARGET=dpdk \
    DAQIRI_MGR="dpdk" \
    scripts/build-container.sh
cd ../..

# 2) Build the stem_daqiri image. Pass STEM_DAQIRI_BUILD_TX / RX to opt in
#    to the binaries you need at each phase.
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=OFF \
    --build-arg STEM_DAQIRI_BUILD_RX=OFF \
    -t stem_daqiri:phase0 .
```

## Phase 0 gate (per Spark)

1. **Build gate** (both Sparks):

   ```bash
   docker run --rm stem_daqiri:phase0 ./stem_daqiri_hello --self-test
   docker run --rm stem_daqiri:phase0 ./stem_daqiri_hello --print-layout
   ```

   Expect: `stem_daqiri_hello: self-test ok` and a STEM layout dump that
   matches the constants in `common/stem_packet.h`.

2. **daqiri default bench (spark-to-spark)** — uses daqiri's stock
   `daqiri_bench_raw_gpudirect` binary already inside the
   `daqiri-torch:local` image, plus the topology-baked configs in
   `cpp_daqiri/benchmarks/`. This verifies the wire link before any
   STEM-specific code lands.

   Make sure hugepages are allocated on both Sparks. The Phase 3 sweep's
   expanded RX pool (262144 mbufs * 8064 B = ~2.1 GiB host-pinned) needs
   2048 hugepages = 4 GiB:

   ```bash
   sudo sysctl -w vm.nr_hugepages=2048   # 2048 * 2MB = 4 GiB
   ```

   On **spark-201a (RX)**, start first:

   ```bash
   docker run --rm -it --privileged --network host \
       --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
       -v /dev/hugepages:/dev/hugepages \
       -v "$PWD/cpp_daqiri/benchmarks:/configs" \
       daqiri-torch:local \
       /opt/daqiri/bin/daqiri_bench_raw_gpudirect \
           /configs/daqiri_default_rx_spark201a.yaml --seconds 15
   ```

   On **spark-960b (TX)**, within ~5 seconds, run:

   ```bash
   docker run --rm -it --privileged --network host \
       --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
       -v /dev/hugepages:/dev/hugepages \
       -v "$PWD/cpp_daqiri/benchmarks:/configs" \
       daqiri-torch:local \
       /opt/daqiri/bin/daqiri_bench_raw_gpudirect \
           /configs/daqiri_default_tx_spark960b.yaml --seconds 10
   ```

   Expect on the RX side something like:
   `RX complete: packets=<big number> bytes=<big number> bursts=<...>`

## Phase 1 gate (per Spark)

Build image with TX enabled:

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    -t stem_daqiri:phase1 .
```

Run via:

```bash
# on spark-stacked-02 (RX validator -- existing Holoscan pipeline)
cpp_daqiri/scripts/run_phase1_rx.sh

# on spark-stacked-01 (new daqiri TX)
cpp_daqiri/scripts/run_phase1_tx.sh --seconds 10 --rate 50
```

Pass criteria: TX prints `achieved Gbps` within 5% of `target_rate_gbps`;
Holoscan RX logs show `StemReceiverOp` ticking; no DPDK / rx_missed errors.

## Phase 2 gate (per Spark)

Build image with TX + RX enabled:

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    -t stem_daqiri:phase2 .
```

Run via:

```bash
# on spark-stacked-02
cpp_daqiri/scripts/run_phase2_rx.sh --seconds 14

# on spark-stacked-01
cpp_daqiri/scripts/run_phase2_tx.sh --seconds 10 --rate 50
```

Pass criteria: RX prints assembled frame count >= 0; bytes received within
loss tolerance; no daqiri `NO_FREE_BURST_BUFFERS` errors.

## Phase 3 parity gate (per Spark)

Build image with TX + RX + Phase 3 instrumentation enabled (same flags as
Phase 2; the latency stamping and dark-correction are activated via YAML
flags `stamp_epoch_us`, `capture_latency`, `subtract_dark`,
`apply_valid_pixel_mask`):

```bash
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    -t stem_daqiri:phase3 .
```

Run the full parity sweep:

1. **daqiri-vs-daqiri pass** -- single command from spark-stacked-01
   (the orchestrator drives the RX over SSH so RX init and TX launch
   stay aligned every iteration):

   ```bash
   cpp_daqiri/scripts/run_phase3_sweep_orchestrated.sh \
       --rates "10 25 50 80" --runs 1 --seconds 8 \
       --outdir cpp_daqiri/benchmarks/sweep_<utc>
   ```

2. **Holoscan baseline pass** -- the orchestrator only knows the daqiri
   binary today, so for the Holoscan side use the older per-side scripts:

   ```bash
   # spark-stacked-02
   cpp_daqiri/scripts/run_phase3_sweep_rx.sh --label holoscan_rx --binary holoscan

   # spark-stacked-01
   cpp_daqiri/scripts/run_phase3_sweep_tx.sh --label holoscan_tx
   ```

3. **Parse results into `cpp_daqiri/benchmarks/results.md`:**

   ```bash
   cpp_daqiri/scripts/parse_phase3_results.py \
       --daqiri-tx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
       --daqiri-rx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
       --holoscan-tx-dir cpp_daqiri/benchmarks/logs_tx_<utc> \
       --holoscan-rx-dir cpp_daqiri/benchmarks/logs_rx_<utc> \
       --duration 8
   ```

Parity gate is closed when every cell in the verdict column is all-PASS at
every rate. See [`benchmarks/results.md`](benchmarks/results.md) for the
exact rubric and tuning playbook if any metric regresses.

## Files

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Phased build (opt-in TX/RX via `-DSTEM_DAQIRI_BUILD_*=ON`) |
| `common/stem_packet.h` | STEM wire layout + frame geometry constants |
| `common/stem_daqiri_hello.cpp` | Phase 0 link-check binary |
| `common/stem_kernels.cu` | STEM CUDA kernels (TX populate, RX gather, header extract) — appears in Phase 1 |
| `common/stem_pacing.{h,cpp}` | Token-bucket pacing for paced TX — appears in Phase 1 |
| `tx/stem_tx_main.cpp` | Phase 1 TX binary |
| `rx/stem_rx_main.cpp` | Phase 2 RX binary |
| `configs/stem_tx_spark.yaml` | Phase 1 TX YAML for the two-Spark topology |
| `configs/stem_rx_spark.yaml` | Phase 2 RX YAML for the two-Spark topology |
| `benchmarks/daqiri_default_*.yaml` | Phase 0 baseline daqiri benchmark configs |
| `benchmarks/results.md` | Phase 3 parity-gate measurement table (run sweep to fill in) |
| `scripts/verify_phase.sh` | Per-phase build + smoke-test, runs on a single Spark |
| `scripts/run_phase{1,2}_{tx,rx}.sh` | Single-run smoke wrappers around the docker invocation |
| `scripts/run_phase3_sweep_{tx,rx}.sh` | Per-side loop over rates (used for the Holoscan pass) |
| `scripts/run_phase3_sweep_orchestrated.sh` | Single-host driver, drives RX over SSH; preferred for daqiri-vs-daqiri pass |
| `scripts/parse_phase3_results.py` | Parses sweep logs into `benchmarks/results.md` |
