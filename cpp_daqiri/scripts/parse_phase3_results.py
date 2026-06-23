#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
"""Parse Phase 3 parity-sweep TX/RX logs into a single markdown table.

Usage:
    parse_phase3_results.py \
        --daqiri-tx-dir   cpp_daqiri/benchmarks/logs_tx_<utc>   \
        --daqiri-rx-dir   cpp_daqiri/benchmarks/logs_rx_<utc>   \
        --holoscan-tx-dir cpp_daqiri/benchmarks/logs_tx_<utc>   \
        --holoscan-rx-dir cpp_daqiri/benchmarks/logs_rx_<utc>   \
        --output cpp_daqiri/benchmarks/results.md

The script accepts log filenames produced by run_phase3_sweep_{tx,rx}.sh:
    <label>_<rate>gbps_run<N>.log

For each rate it computes per-side mean Gbps, drops, fps, p50/p99 latency
across the runs, and emits a parity table that flags the four metrics on
which daqiri must match or beat Holoscan.

Two log formats are understood:

1. **daqiri stdout** -- key/value lines such as
       "achieved Gbps : 49.871"
       "frames assembled : 1280  (fps 127.92)"
       "latency p50/p90/p99/p999 us : 142 / 187 / 247 / 612"

2. **Holoscan stdout** -- the StemReceiverOp destructor logs
       "[info] Finished receiver with <bytes>/<packets> bytes/packets
        received, <drops> packets dropped, and <ignored> packets ignored
        from unexpected source IDs..."
   The script derives Gbps from "bytes / duration" where duration comes
   from the filename's "_<rate>gbps_run<N>.log" suffix paired with the
   sweep's --seconds value (passed in via --duration; defaults to 10).

   Holoscan and daqiri both log
       "latency p50/p90/p99/p999 us : ..."
   after reading the STEM header epoch_us field on canonical frame-start
   packets.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# daqiri stdout patterns
# ---------------------------------------------------------------------------
_TX_PATTERNS = {
    "achieved_gbps": re.compile(r"achieved Gbps\s*:\s*([\d.]+)"),
    "packets":      re.compile(r"packets sent\s*:\s*(\d+)"),
    "bytes":        re.compile(r"bytes sent\s*:\s*(\d+)"),
}

_DAQIRI_RX_PATTERNS = {
    "achieved_gbps":   re.compile(r"achieved Gbps\s*:\s*([\d.]+)"),
    "frames_assembled": re.compile(r"frames assembled\s*:\s*(\d+)"),
    "fps":             re.compile(r"\(fps\s+([\d.]+)\)"),
    "drops_src":       re.compile(r"unexpected source\s*:\s*(\d+)"),
    "drops_window":    re.compile(r"out-of-window\s*:\s*(\d+)"),
    "lat_p50":         re.compile(r"latency p50/p90/p99/p999 us\s*:\s*(-?\d+)\s*/\s*-?\d+\s*/\s*-?\d+\s*/\s*-?\d+"),
    "lat_p99":         re.compile(r"latency p50/p90/p99/p999 us\s*:\s*-?\d+\s*/\s*-?\d+\s*/\s*(-?\d+)\s*/\s*-?\d+"),
    "bytes_recv":      re.compile(r"bytes received\s*:\s*(\d+)"),
    "duration":        re.compile(r"duration\s*:\s*([\d.]+)\s*s"),
}

# ---------------------------------------------------------------------------
# Holoscan StemReceiverOp destructor pattern. Captured groups:
#   1: bytes received
#   2: packets received
#   3: packets dropped
#   4: packets ignored from unexpected source IDs
# Holoscan does not log Gbps directly; we compute it from bytes and the
# nominal duration (see --duration).
# ---------------------------------------------------------------------------
_HOLO_RX_FINISHED = re.compile(
    r"Finished receiver with\s+(\d+)/(\d+)\s+bytes/packets received,\s+"
    r"(\d+)\s+packets dropped,\s+and\s+(\d+)\s+packets ignored"
)
# Holoscan StemReceiverOp::initialize() prints "Batching {} frames" so we
# can recover frames_per_tensor, but for parity we want emitted-frame
# count. Holoscan currently does not log that in a stable line; we'd need
# to grep for "Reached frame limit" or count tick events. Leave as TODO.

_FNAME_RE = re.compile(r"_([0-9]+)gbps_run([0-9]+)\.log$")


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _grep_first(text: str, regex: re.Pattern) -> Optional[float]:
    m = regex.search(text)
    if not m:
        return None
    val = m.group(1)
    return float(val) if "." in val else float(int(val))


def _scan_daqiri(dir_: Optional[Path],
                 patterns: Dict[str, re.Pattern],
                 fname_glob: str = "*.log") -> Dict[int, Dict[str, List[float]]]:
    """Scan one log directory matching ``fname_glob``.

    The orchestrator places TX and RX logs side-by-side in the same sweep
    folder, so we filter by filename role ("*tx*.log" / "*rx*.log") to
    avoid pulling the RX's "achieved Gbps" line into TX stats and vice
    versa.
    """
    out: Dict[int, Dict[str, List[float]]] = {}
    if dir_ is None or not dir_.exists():
        return out
    for log in sorted(dir_.glob(fname_glob)):
        m = _FNAME_RE.search(log.name)
        if not m:
            continue
        rate = int(m.group(1))
        slot = out.setdefault(rate, {k: [] for k in patterns})
        text = _read(log)
        for key, regex in patterns.items():
            val = _grep_first(text, regex)
            if val is not None:
                slot[key].append(val)
    return out


def _scan_holoscan_rx(dir_: Optional[Path],
                      nominal_duration_s: float) -> Dict[int, Dict[str, List[float]]]:
    """Holoscan log scanner. Emits the same shape as _scan_daqiri so the
    rest of the parser is generic."""
    out: Dict[int, Dict[str, List[float]]] = {}
    if dir_ is None or not dir_.exists():
        return out
    for log in sorted(dir_.glob("*.log")):
        m = _FNAME_RE.search(log.name)
        if not m:
            continue
        rate = int(m.group(1))
        slot = out.setdefault(rate, {
            "achieved_gbps": [], "drops_src": [], "drops_window": [],
            "fps": [], "lat_p50": [], "lat_p99": [], "bytes_recv": [],
            "duration": [], "frames_assembled": [],
        })
        text = _read(log)
        fm = _HOLO_RX_FINISHED.search(text)
        if not fm:
            continue
        bytes_recv = float(fm.group(1))
        # group(2) packets received -- redundant, skip
        drops_src_or_dropped = float(fm.group(3))  # Holoscan calls all of these "dropped"
        drops_ignored = float(fm.group(4))
        slot["bytes_recv"].append(bytes_recv)
        slot["drops_src"].append(drops_ignored)        # unexpected-source bucket
        slot["drops_window"].append(drops_src_or_dropped)  # generic drops bucket
        slot["duration"].append(nominal_duration_s)
        slot["achieved_gbps"].append(
            bytes_recv * 8.0 / (nominal_duration_s * 1e9)
            if nominal_duration_s > 0 else 0.0)
        for key in ("lat_p50", "lat_p99"):
            val = _grep_first(text, _DAQIRI_RX_PATTERNS[key])
            if val is not None:
                slot[key].append(val)
        # No fps yet: Holoscan does not log emitted-frame rate in a stable line.
    return out


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _fmt(val: Optional[float], spec: str = "{:.3f}") -> str:
    return spec.format(val) if val is not None else "n/a"


def _cmp(d: Optional[float], h: Optional[float], lower_is_better: bool) -> str:
    if d is None or h is None:
        return "-"
    ok = (d <= h) if lower_is_better else (d >= h)
    return "PASS" if ok else "FAIL"


def _parity_row(rate: int,
                d_tx: Dict[str, List[float]],
                d_rx: Dict[str, List[float]],
                h_tx: Dict[str, List[float]],
                h_rx: Dict[str, List[float]]) -> str:
    d_gbps = _mean(d_rx.get("achieved_gbps", []))
    h_gbps = _mean(h_rx.get("achieved_gbps", []))

    # "drops" here = real wire-loss (TX pkts sent - RX pkts received),
    # plus app-level rejections (unexpected source IDs, out-of-window).
    # For Holoscan the RX log only carries the dropped/ignored buckets
    # (no TX delta), so we report those alone.
    def _wire_loss(tx: Dict[str, List[float]],
                   rx: Dict[str, List[float]]) -> float:
        tx_pkts = sum(tx.get("packets", []))
        # daqiri RX doesn't have a single "packets received" field in our
        # pattern dict but bytes_recv / packet_size is close enough; even
        # easier, we sum frames * 1024 which is the per-frame packet count.
        rx_bytes = sum(rx.get("bytes_recv", []))
        rx_pkts = rx_bytes / 7786.0 if rx_bytes else 0.0
        return max(0.0, tx_pkts - rx_pkts)

    d_drops = (_wire_loss(d_tx, d_rx) +
               sum(d_rx.get("drops_window", []) + d_rx.get("drops_src", [])))
    h_drops = sum(h_rx.get("drops_window", []) + h_rx.get("drops_src", []))
    d_p50 = _mean(d_rx.get("lat_p50", []))
    h_p50 = _mean(h_rx.get("lat_p50", []))
    d_p99 = _mean(d_rx.get("lat_p99", []))
    h_p99 = _mean(h_rx.get("lat_p99", []))
    d_fps = _mean(d_rx.get("fps", []))
    h_fps = _mean(h_rx.get("fps", []))

    return (
        f"| {rate} "
        f"| {_fmt(d_gbps)} / {_fmt(h_gbps)} "
        f"| {d_drops:.0f} / {h_drops:.0f} "
        f"| {_fmt(d_p50, '{:.1f}')} / {_fmt(h_p50, '{:.1f}')} "
        f"| {_fmt(d_p99, '{:.1f}')} / {_fmt(h_p99, '{:.1f}')} "
        f"| {_fmt(d_fps, '{:.2f}')} / {_fmt(h_fps, '{:.2f}')} "
        f"| Gbps {_cmp(d_gbps, h_gbps, False)}, drops {_cmp(float(d_drops), float(h_drops), True)}, "
        f"p50 {_cmp(d_p50, h_p50, True)}, p99 {_cmp(d_p99, h_p99, True)}, "
        f"fps {_cmp(d_fps, h_fps, False)} |"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daqiri-tx-dir",  type=Path, default=None)
    ap.add_argument("--daqiri-rx-dir",  type=Path, default=None)
    ap.add_argument("--holoscan-tx-dir", type=Path, default=None)
    ap.add_argument("--holoscan-rx-dir", type=Path, default=None)
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Nominal seconds per run, used to derive Holoscan "
                         "Gbps from bytes-received (Holoscan does not log "
                         "Gbps directly). Default 10.")
    ap.add_argument("--output", type=Path,
                    default=Path("cpp_daqiri/benchmarks/results.md"))
    args = ap.parse_args()

    daqiri_tx = _scan_daqiri(args.daqiri_tx_dir, _TX_PATTERNS,
                             fname_glob="*tx*.log")
    daqiri_rx = _scan_daqiri(args.daqiri_rx_dir, _DAQIRI_RX_PATTERNS,
                             fname_glob="*rx*.log")
    holo_tx   = _scan_daqiri(args.holoscan_tx_dir, _TX_PATTERNS,
                             fname_glob="*tx*.log")
    holo_rx   = _scan_holoscan_rx(args.holoscan_rx_dir, args.duration)

    # The daqiri RX prints "achieved Gbps" relative to its own --seconds
    # window (which the orchestrator pads beyond the TX duration so the RX
    # has time to drain). Normalize it to the TX-side duration so the
    # number aligns with how Holoscan reports the same value (bytes /
    # nominal TX duration). Without this the daqiri/holoscan column reads
    # ~33% / 100% even when daqiri caught every byte.
    if args.duration > 0:
        for rate, slot in daqiri_rx.items():
            recv_list = slot.get("bytes_recv", [])
            if recv_list:
                slot["achieved_gbps"] = [
                    b * 8.0 / (args.duration * 1e9) for b in recv_list
                ]

    rates = sorted(set(daqiri_rx) | set(holo_rx))
    lines = [
        "# Phase 3 parity gate -- daqiri vs Holoscan",
        "",
        "Each cell is `daqiri / holoscan` (means across runs). PASS means daqiri",
        "matches-or-beats Holoscan on that metric.",
        "",
        "Latency p50/p99 are parsed from the common",
        "`latency p50/p90/p99/p999 us : ...` line emitted by both RX paths.",
        "",
        "| Target Gbps | Achieved Gbps | Drops | Latency p50 us | Latency p99 us | FPS | Verdict |",
        "|------------:|:--------------|:------|---------------:|---------------:|----:|:--------|",
    ]
    for r in rates:
        lines.append(_parity_row(r,
                                  daqiri_tx.get(r, {}),
                                  daqiri_rx.get(r, {}),
                                  holo_tx.get(r, {}),
                                  holo_rx.get(r, {})))
    lines.append("")
    lines.append("## TX achieved Gbps (sanity check that pacing held)")
    lines.append("")
    lines.append("| Target Gbps | daqiri TX mean | Holoscan TX mean |")
    lines.append("|------------:|---------------:|-----------------:|")
    # Holoscan TX = daqiri TX in both passes; both columns will look the
    # same. Kept for the human-readability of the sweep procedure.
    for r in rates:
        d = _mean(daqiri_tx.get(r, {}).get("achieved_gbps", []))
        h = _mean(holo_tx.get(r, {}).get("achieved_gbps", []))
        lines.append(f"| {r} | {_fmt(d)} | {_fmt(h)} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
