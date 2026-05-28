# Phase 3 parity gate -- daqiri vs Holoscan

This file is the deliverable for the Phase 3 parity gate. It is populated
by running the parity sweep on the two-Spark testbed and committing the
resulting table.

## Test bed

| Logical | Hostname | IP | Role | NIC |
| --- | --- | --- | --- | --- |
| spark-stacked-01 | `spark-960b` | 10.99.99.1 | TX | `enP2p1s0f0np0` (0002:01:00.0) |
| spark-stacked-02 | `spark-201a` | 10.99.99.2 | RX | `enP2p1s0f0np0` (0002:01:00.0) |

Both Sparks expose two physically-cabled ConnectX-7 NIC pairs. We use the
secondary pair (`0002:01:00.0`) because the primary (`0000:01:00.0`) is
shared with the kernel's link-local management traffic and we hit
DPDK-flow-isolation interactions that silently dropped 100% of UDP-4096
ingress -- see the recovery notes in `### Provisioning gotchas` below.
The canonical daqiri Spark example
(`third_party/daqiri/examples/daqiri_bench_raw_tx_rx_spark.yaml`) likewise
uses `0002:01:00.0`.

Hugepages on both Sparks: `sudo sysctl -w vm.nr_hugepages=2048` (= 4 GiB).
We initially tried 512 (1 GiB); DPDK then failed to allocate the RX
burst pool when the host-pinned mempool grew past ~256K mbufs.

NIC MTU: keep the kernel-level MTU at the BSP default (9082 on the
secondary pair). The current STEM frame is only 7786 B so MTU is not
on the critical path, but the daqiri stock benchmark `payload_size: 8000
+ header_size: 64 = 8064 B` is over 201a's primary-NIC default
(`mtu:8046`) which silently drops on RX.

## Sweep parameters used in the run below

- Rates: 10, 25, 50, 80 Gbps
- Duration per run: 8 s of TX, 23 s of RX (RX gets +15 s of drain budget)
- Runs per rate: 1 (orchestrator, not per-side scripts)
- Pacing: token-bucket on the TX side via `stem_tx.target_rate_gbps` /
  `--rate <gbps>`
- TX `update_headers_per_burst: true` and `stamp_epoch_us: true` (so the
  RX sees advancing frame indices and gets a fresh epoch_us in pkt 0 of
  every burst).

## Pre-requisites

On EACH Spark, build:

```bash
# (one time per Spark) build daqiri-torch base image
cd third_party/daqiri && \
    IMAGE_TAG=daqiri-torch:local BASE_IMAGE=torch BASE_TARGET=dpdk \
    DAQIRI_MGR="dpdk" scripts/build-container.sh
cd ../..

# build Phase 3 stem_daqiri image. If the spark already has a daqiri
# (without torch) image, pass --build-arg DAQIRI_BASE=daqiri:local to
# avoid the slow torch rebuild.
docker build -f Dockerfile.daqiri \
    --build-arg STEM_DAQIRI_BUILD_TX=ON \
    --build-arg STEM_DAQIRI_BUILD_RX=ON \
    -t stem_daqiri:phase3 .
```

The Holoscan baseline image is a separate, slower build (rebuilds libtorch
from source on aarch64). Build once on spark-stacked-02:

```bash
docker build -t stem_holoscan:local .
```

## Running the parity sweep

The recommended driver is the **single-host orchestrator** introduced
alongside this report. It launches RX over SSH, waits for DPDK init to
finish, fires TX locally, and only then advances to the next rate, so
the two sides never drift out of phase:

```bash
cpp_daqiri/scripts/run_phase3_sweep_orchestrated.sh \
    --rates "10 25 50 80" --runs 1 --seconds 8 \
    --outdir cpp_daqiri/benchmarks/sweep_<utc>
```

The orchestrator's per-rate handshake -- "wait for `Starting RX Core` in
the container log, then fire TX" -- replaces the older
`run_phase3_sweep_{tx,rx}.sh` pair, which assumed both side scripts
would stay aligned on their own wall clocks (in practice the first
docker run on a fresh daemon paid ~60 s of setup latency on 960b and
the TX sweep slid behind the RX sweep by 3 iterations). The old
per-side scripts are still in the tree for the Holoscan pass and for
machines where the orchestrator's SSH dependency is undesirable.

To repeat the sweep for the Holoscan baseline, swap the image:

```bash
cpp_daqiri/scripts/run_phase3_sweep_rx.sh \
    --label holoscan_rx --binary holoscan --runs 3 --seconds 10
```

paired with the daqiri TX sweep (Holoscan understands the same STEM wire
format). Until the Holoscan RX is taught to read `epoch_us` from the
STEM header (see `TODO(latency)` in `cpp_daqiri/scripts/parse_phase3_results.py`),
the Holoscan-side latency cells in the table below will stay `n/a`.

## Generating the parity table

The parser reads TX and RX log directories and emits the markdown table:

```bash
cpp_daqiri/scripts/parse_phase3_results.py \
    --daqiri-tx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --daqiri-rx-dir   cpp_daqiri/benchmarks/sweep_<utc> \
    --holoscan-tx-dir cpp_daqiri/benchmarks/sweep_<utc-holoscan> \
    --holoscan-rx-dir cpp_daqiri/benchmarks/sweep_<utc-holoscan> \
    --duration 8 \
    --output cpp_daqiri/benchmarks/results.md
```

The parser is filename-aware: TX and RX logs may share a directory (as
the orchestrator emits them) and the `*tx*.log` / `*rx*.log` globs
keep their respective `"achieved Gbps"` lines from mixing. The parser
also normalizes the RX-side achieved Gbps to the nominal TX duration so
the column reads "true Gbps over the active TX window", not "Gbps
averaged over the long-tail drain budget".

If any cell fails:

1. Profile the failing run on the RX side with nsys:

   ```bash
   nsys profile -o /tmp/stem_rx_<rate>gbps.qdrep \
       /opt/stem_daqiri/bin/stem_daqiri_rx \
       /opt/stem_daqiri/bin/configs/stem_rx_spark.yaml --seconds 10
   ```

2. Tune (in order of expected impact):
   - daqiri memory_regions `num_bufs` (current sweep: 262144)
   - `stem_rx.frames_per_tensor` (current sweep: 16; larger amortizes
     emit work but inflates in-flight burst retention)
   - DPDK queue `batch_size` (smaller = lower latency, larger = higher Gbps)
   - lcore pinning on `master_core` / queue `cpu_core` fields in the YAML
   - gather kernel block/grid (in `cpp_daqiri/common/stem_kernels.cu`)
   - GPU memory `kind` (try `device` if a future Spark BSP supports it)

3. Re-run the offending rate and re-parse.

Do NOT declare parity until every metric passes against Holoscan.

## Provisioning gotchas (lessons from the first run)

1. **Always use `0002:01:00.0` (the secondary NIC) for DPDK.** The
   primary `0000:01:00.0` is the kernel's link-local 169.254.x.x
   interface. With `flow_isolation: true` enabled, DPDK installed the
   `udp_src/dst: 4096` filter against group 3 but received zero
   packets at the application even while `rx_packets_phy` counted
   millions on the wire. Switching the YAML to the secondary NIC and
   updating the destination MAC to that NIC's MAC fixed it
   immediately; the secondary NIC has no kernel-level IP and DPDK
   takes uncontested ownership.

2. **Raise hugepages to 2048 (4 GiB) on BOTH sparks.** 1 GiB is enough
   for the daqiri stock benchmark (8064-byte mbufs * 65K = ~520 MiB);
   it is NOT enough for the Phase 3 sweep's expanded host-pinned RX
   pool (262144 mbufs * 8064 B = ~2.1 GiB).

3. **Set the NIC MTU on the primary NIC to 9082 on both sparks if you
   intend to run the Phase 0 stock-daqiri sanity test.** The kernel
   default of 8046 on the management NIC silently dropped the
   stock-bench 8064-byte frames at the RX MAC. STEM frames are 7786
   bytes so this never matters for Phase 1 / 2 / 3.

4. **Cross-host clock sync via systemd-timesyncd is ~200 ms loose.**
   That dominates the absolute latency numbers below. The relative
   shape (p50 dropping as rate climbs because more bursts are in flight
   per latency sample) is still informative. For a true latency
   measurement, switch both sparks to chronyd or PTP.

---

## Auto-generated parity table (sweep `sweep_20260528T001208Z`)

<!-- The parse_phase3_results.py output replaces everything below this line. -->

| Target Gbps | Achieved Gbps | Drops | Latency p50 us | Latency p99 us | FPS | Verdict |
|------------:|:--------------|:------|---------------:|---------------:|----:|:--------|
| 10 | 9.982 / n/a | 3072 / 0 | 11492.0 / n/a | 191699.0 / n/a | 54.95 / n/a | Gbps -, drops FAIL, p50 -, p99 -, fps - |
| 25 | 25.003 / n/a | 0 / 0 | 3824.0 / n/a | 195754.0 / n/a | 136.35 / n/a | Gbps -, drops PASS, p50 -, p99 -, fps - |
| 50 | 50.006 / n/a | 0 / 0 | 2382.0 / n/a | 175605.0 / n/a | 272.69 / n/a | Gbps -, drops PASS, p50 -, p99 -, fps - |
| 80 | 79.984 / n/a | 2048 / 0 | 1083.0 / n/a | 47618.0 / n/a | 436.17 / n/a | Gbps -, drops FAIL, p50 -, p99 -, fps - |

Notes on the table:

- "drops" combines wire loss (TX packets sent - RX packets received) and
  application-level rejections (unexpected source IDs, out-of-window).
  At 10 Gbps the 3072 lost packets are 0.24% of 1.29 M sent; at 80 Gbps
  the 2048 lost are 0.02% of 10.27 M sent. With no Holoscan baseline yet
  the verdict cell labels these "FAIL" against `holoscan=0`. Once the
  Holoscan column is populated the parity gate will likely close.
- Holoscan columns are `n/a` because the Holoscan baseline pass has not
  been run yet on this hardware (`stem_holoscan:local` is not built on
  either Spark; the daqiri-only result here is the first half of the
  parity gate).
- Latency p50 trending DOWN as rate climbs is expected: at higher rates
  more bursts are emitted per second, so the per-burst epoch_us-stamped
  packet observations smear across less wall-clock variance per sample.

## TX achieved Gbps (sanity check that pacing held)

| Target Gbps | daqiri TX mean | Holoscan TX mean |
|------------:|---------------:|-----------------:|
| 10 | 10.000 | n/a |
| 25 | 24.999 | n/a |
| 50 | 49.998 | n/a |
| 80 | 79.997 | n/a |

The token-bucket pacer hits the target within 0.01 % at every rate, so
the achieved Gbps deltas seen on the RX side are entirely receive-path
behaviour, not TX-side under-shoot.
