# Phase 3 parity gate -- daqiri vs Holoscan

Each cell is `daqiri / holoscan` (means across runs). PASS means daqiri
matches-or-beats Holoscan on that metric.

TODO(latency): Holoscan-side p50/p99 are populated as `n/a` until the
Holoscan RX is taught to read epoch_us from STEM headers (~30 lines
in cpp/stem_receiver_op.h::add_pending_packet to capture epoch_us
for (source_id==0, row_offset==0) packets and one log line in
emit_current_assembled_batch).

| Target Gbps | Achieved Gbps | Drops | Latency p50 us | Latency p99 us | FPS | Verdict |
|------------:|:--------------|:------|---------------:|---------------:|----:|:--------|
| 10 | 9.982 / n/a | 3072 / 0 | 11492.0 / n/a | 191699.0 / n/a | 54.95 / n/a | Gbps -, drops FAIL, p50 -, p99 -, fps - |
| 25 | 25.003 / n/a | 0 / 0 | 3824.0 / n/a | 195754.0 / n/a | 136.35 / n/a | Gbps -, drops PASS, p50 -, p99 -, fps - |
| 50 | 50.006 / n/a | 0 / 0 | 2382.0 / n/a | 175605.0 / n/a | 272.69 / n/a | Gbps -, drops PASS, p50 -, p99 -, fps - |
| 80 | 79.984 / n/a | 2048 / 0 | 1083.0 / n/a | 47618.0 / n/a | 436.17 / n/a | Gbps -, drops FAIL, p50 -, p99 -, fps - |

## TX achieved Gbps (sanity check that pacing held)

| Target Gbps | daqiri TX mean | Holoscan TX mean |
|------------:|---------------:|-----------------:|
| 10 | 10.000 | n/a |
| 25 | 24.999 | n/a |
| 50 | 49.998 | n/a |
| 80 | 79.997 | n/a |
