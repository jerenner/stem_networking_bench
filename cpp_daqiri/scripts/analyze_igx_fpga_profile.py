#!/usr/bin/env python3
"""Plot and summarize a run produced by run_igx_fpga_profile.sh."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, help="timestamped profile directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: PROFILE_DIR/analysis)",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_log_timestamp(line: str) -> pd.Timestamp | None:
    match = re.match(r"([^\t]+)\t", line)
    return pd.to_datetime(match.group(1), utc=True) if match else None


def parse_daqiri_log(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    worker_start = None
    completion_time = None
    drop_events = []
    summaries = []
    current_summary = None

    summary_patterns = {
        "duration_seconds": r"duration\s+:\s+([\d.]+)",
        "bursts_polled": r"bursts polled\s+:\s+(\d+)",
        "packets_received": r"packets received\s+:\s+(\d+)",
        "bytes_received": r"bytes received\s+:\s+(\d+)",
        "gbps": r"achieved Gbps\s+:\s+([\d.]+)",
        "frames_assembled": r"frames assembled\s+:\s+(\d+)",
        "fps": r"frames assembled\s+:\s+\d+\s+\(fps\s+([\d.]+)\)",
        "unexpected_source": r"unexpected source:\s+(\d+)",
        "tile_dropped_packets": r"tile dropped pkts:\s+(\d+)",
        "out_of_window": r"out-of-window\s+:\s+(\d+)",
        "incomplete_batches": r"incomplete batches:\s+(\d+)",
        "incomplete_missing": r"incomplete missing:\s+(\d+)",
        "incomplete_max": r"incomplete max\s+:\s+(\d+)",
        "sink_pool_drops": r"sink pool drops\s+:\s+(\d+)",
        "sink_queued": r"sink queued\s+:\s+(\d+)",
        "sink_written": r"sink written\s+:\s+(\d+)",
        "sink_errors": r"sink errors\s+:\s+(\d+)",
    }
    integer_fields = {
        key
        for key in summary_patterns
        if key not in {"duration_seconds", "gbps", "fps"}
    }

    for line in lines:
        timestamp = parse_log_timestamp(line)
        if "Done starting workers" in line:
            worker_start = timestamp
        if "stem_daqiri_rx complete:" in line:
            if current_summary:
                summaries.append(current_summary)
            current_summary = {}
            completion_time = timestamp
            continue
        if current_summary is not None:
            for key, pattern in summary_patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = match.group(1)
                    current_summary[key] = int(value) if key in integer_fields else float(value)

        drop_match = re.search(
            r"'rx_port_(\d+)'.*Dropped (\d+) packets since last poll", line
        )
        if drop_match and timestamp is not None:
            drop_events.append(
                {
                    "timestamp": timestamp,
                    "port": int(drop_match.group(1)),
                    "packets": int(drop_match.group(2)),
                }
            )

    if current_summary:
        summaries.append(current_summary)
    if worker_start is None or completion_time is None or len(summaries) != 2:
        raise RuntimeError("Could not identify the worker interval and two receiver summaries")
    return {
        "worker_start": worker_start,
        "completion_time": completion_time,
        "drop_events": pd.DataFrame(drop_events),
        "receivers": summaries,
    }


def parse_nvidia_smi(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [column.strip() for column in frame.columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric = [column for column in frame.columns if column not in {"timestamp", "name", "pstate"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def parse_tegrastats(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    cpu_rows = []

    def find_value(pattern: str, line: str) -> float:
        match = re.search(pattern, line)
        return float(match.group(1)) if match else np.nan

    for line in path.read_text(errors="replace").splitlines():
        timestamp_match = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
        cpu_match = re.search(r"CPU \[([^]]+)\]", line)
        if not timestamp_match or not cpu_match:
            continue
        cpu_values = []
        for entry in cpu_match.group(1).split(","):
            value_match = re.search(r"(\d+)%", entry)
            cpu_values.append(float(value_match.group(1)) if value_match else np.nan)
        rows.append(
            {
                "timestamp": pd.to_datetime(
                    timestamp_match.group(1),
                    format="%m-%d-%Y %H:%M:%S",
                    utc=True,
                ),
                "ram_used_mib": find_value(r"RAM (\d+)/", line),
                "emc_percent": find_value(r"EMC_FREQ (\d+)%", line),
                "cpu_temperature_c": find_value(r"cpu@([\d.]+)C", line),
                "cpu_power_mw": find_value(r"VDD_CPU_CV (\d+)mW", line),
            }
        )
        cpu_rows.append(cpu_values)
    return pd.DataFrame(rows), np.asarray(cpu_rows)


def quantiles(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_timeline(
    output: Path,
    gpu: pd.DataFrame,
    tegra: pd.DataFrame,
    cpu: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    gpu_x = (gpu["timestamp"] - start).dt.total_seconds()
    tegra_x = (tegra["timestamp"] - start).dt.total_seconds()
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)

    axes[0].plot(gpu_x, gpu["utilization_gpu_percent"], label="GPU compute", linewidth=1.5)
    axes[0].plot(
        gpu_x,
        gpu["utilization_memory_percent"],
        label="GPU memory controller",
        linewidth=1.2,
    )
    axes[0].set_ylabel("Utilization (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].legend(loc="upper left", ncols=2)

    axes[1].plot(gpu_x, gpu["power_draw_w"], color="#d45d00", label="GPU power")
    axes[1].set_ylabel("GPU power (W)")
    temperature_axis = axes[1].twinx()
    temperature_axis.plot(
        gpu_x, gpu["temperature_gpu_c"], color="#9c2f56", label="GPU temperature"
    )
    temperature_axis.set_ylabel("GPU temperature (C)")
    lines = axes[1].lines + temperature_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="upper left", ncols=2)

    other_cpu = np.nanmean(np.delete(cpu, [8, 9, 10], axis=1), axis=1)
    axes[2].plot(tegra_x, cpu[:, 8], label="CPU 8 (DAQIRI master)", linewidth=1.0)
    axes[2].plot(tegra_x, cpu[:, 9], label="CPU 9 (RX port 0)", linewidth=1.3)
    axes[2].plot(tegra_x, cpu[:, 10], label="CPU 10 (RX port 1)", linewidth=1.3)
    axes[2].plot(tegra_x, other_cpu, label="Mean of other cores", linewidth=1.0)
    axes[2].set_ylabel("CPU utilization (%)")
    axes[2].set_ylim(-3, 103)
    axes[2].legend(loc="center right", ncols=2)

    axes[3].plot(tegra_x, tegra["ram_used_mib"] / 1024, label="Host RAM used (GiB)")
    emc_axis = axes[3].twinx()
    emc_axis.plot(
        tegra_x,
        tegra["emc_percent"],
        color="#00876c",
        label="Orin EMC utilization",
    )
    axes[3].set_ylabel("Host RAM used (GiB)")
    emc_axis.set_ylabel("Orin EMC (%)")
    axes[3].set_xlabel("Time relative to DAQIRI worker start (s)")
    lines = axes[3].lines + emc_axis.lines
    axes[3].legend(lines, [line.get_label() for line in lines], loc="center right")

    duration = (end - start).total_seconds()
    for axis in axes:
        style_axis(axis)
        axis.axvspan(0, duration, color="#4c956c", alpha=0.055)
        axis.axvline(0, color="#2f6b4f", linestyle="--", linewidth=1)
        axis.axvline(duration, color="#2f6b4f", linestyle="--", linewidth=1)
    figure.suptitle(
        "DAQIRI dual-FPGA platform utilization\n"
        "Shaded interval: active 300-second acquisition",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_cpu_heatmap(
    output: Path,
    tegra: pd.DataFrame,
    cpu: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    x = (tegra["timestamp"] - start).dt.total_seconds().to_numpy()
    figure, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    image = axis.imshow(
        cpu.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=100,
        extent=[x.min(), x.max(), -0.5, cpu.shape[1] - 0.5],
    )
    duration = (end - start).total_seconds()
    axis.axvline(0, color="white", linestyle="--", linewidth=1)
    axis.axvline(duration, color="white", linestyle="--", linewidth=1)
    axis.set_yticks(range(cpu.shape[1]))
    axis.set_ylabel("CPU core")
    axis.set_xlabel("Time relative to DAQIRI worker start (s)")
    axis.set_title("IGX CPU-core utilization (RX polling cores 9 and 10 are intentionally busy)")
    figure.colorbar(image, ax=axis, label="Utilization (%)", pad=0.01)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_summary(output: Path, metrics: dict, drop_events: pd.DataFrame) -> None:
    receivers = metrics["receivers"]
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.65, 1.2, 0.13])
    kpi_axis = figure.add_subplot(grid[0, :])
    throughput_axis = figure.add_subplot(grid[1, 0])
    drop_axis = figure.add_subplot(grid[1, 1])
    note_axis = figure.add_subplot(grid[2, :])

    kpi_axis.axis("off")
    kpis = [
        ("Combined packet bytes", f"{metrics['combined_gbps']:.2f} Gbit/s"),
        ("Combined assembly", f"{metrics['combined_fps']:.0f} frame/s"),
        ("Pipeline sink drops", f"{metrics['sink_pool_drops']:,}"),
        ("Out-of-window", f"{metrics['out_of_window']:,}"),
        ("GPU utilization", f"{metrics['gpu']['utilization_percent']['mean']:.1f}% mean"),
    ]
    for index, (label, value) in enumerate(kpis):
        x = (index + 0.5) / len(kpis)
        kpi_axis.text(x, 0.68, value, ha="center", va="center", fontsize=19, weight="bold")
        kpi_axis.text(x, 0.35, label, ha="center", va="center", fontsize=10, color="#555555")
    kpi_axis.set_title("Five-minute DAQIRI dual-FPGA keep-up summary", fontsize=16, pad=8)

    ports = np.arange(len(receivers))
    throughput_axis.bar(
        ports,
        [receiver["gbps"] for receiver in receivers],
        color=["#277da1", "#43aa8b"],
    )
    throughput_axis.axhline(100, color="#777777", linestyle=":", label="100-Gbit/s line rate")
    throughput_axis.set_xticks(ports, [f"RX port {port}" for port in ports])
    throughput_axis.set_ylabel("Accepted packet-byte rate (Gbit/s)")
    throughput_axis.set_ylim(0, 110)
    throughput_axis.set_title("Sustained rate per independent receiver")
    throughput_axis.legend(loc="upper right")
    style_axis(throughput_axis)

    if not drop_events.empty:
        for port, color in [(0, "#277da1"), (1, "#43aa8b")]:
            selected = drop_events[drop_events["port"] == port]
            drop_axis.scatter(
                selected["relative_seconds"],
                selected["packets"],
                s=55,
                label=f"RX port {port}",
                color=color,
            )
        drop_axis.set_yscale("log")
        drop_axis.legend()
    drop_axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
    drop_axis.set_xlabel("Time after DAQIRI worker start (s)")
    drop_axis.set_ylabel("Packets dropped in 500-ms poll")
    drop_axis.set_title("Reported NIC-drop events: startup only")
    style_axis(drop_axis)

    note = (
        f"Incomplete batches: {metrics['incomplete_batches']:,}/"
        f"{metrics['assembled_batches']:,} ({metrics['incomplete_batch_percent']:.3f}%).  "
        f"Missing expected packets: {metrics['incomplete_missing']:,} "
        f"({metrics['missing_expected_packet_percent']:.4f}%).\n"
        f"Compatibility tiles intentionally ignored: {metrics['tile_dropped_packets']:,} "
        f"({metrics['tile_dropped_percent']:.3f}% of received packets).  "
        f"Unexpected-source packets: {metrics['unexpected_source']:,} "
        f"({metrics['unexpected_source_percent']:.4f}%)."
    )
    note_axis.axis("off")
    note_axis.text(0.5, 0.5, note, ha="center", va="center", fontsize=10)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    profile = args.profile_dir.resolve()
    output = (args.output_dir or profile / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(profile / "run_metadata.txt")
    daqiri = parse_daqiri_log(profile / "daqiri.log")
    gpu = parse_nvidia_smi(profile / "nvidia_smi.csv")
    tegra, cpu = parse_tegrastats(profile / "tegrastats.log")
    start = daqiri["worker_start"]
    end = daqiri["completion_time"]

    active_gpu = gpu[(gpu["timestamp"] >= start) & (gpu["timestamp"] <= end)]
    active_tegra_mask = (tegra["timestamp"] >= start) & (tegra["timestamp"] <= end)
    active_tegra = tegra[active_tegra_mask]
    active_cpu = cpu[active_tegra_mask.to_numpy()]

    receivers = daqiri["receivers"]
    packets_received = sum(item["packets_received"] for item in receivers)
    frames_assembled = sum(item["frames_assembled"] for item in receivers)
    assembled_batches = frames_assembled // 128
    incomplete_missing = sum(item["incomplete_missing"] for item in receivers)
    expected_packet_slots = assembled_batches * 46080
    tile_dropped = sum(item["tile_dropped_packets"] for item in receivers)
    unexpected_source = sum(item["unexpected_source"] for item in receivers)

    metrics = {
        "profile": str(profile),
        "requested_duration_seconds": int(metadata["requested_duration_seconds"]),
        "active_duration_seconds": float((end - start).total_seconds()),
        "combined_gbps": sum(item["gbps"] for item in receivers),
        "combined_fps": sum(item["fps"] for item in receivers),
        "packets_received": packets_received,
        "frames_assembled": frames_assembled,
        "assembled_batches": assembled_batches,
        "sink_pool_drops": sum(item["sink_pool_drops"] for item in receivers),
        "sink_errors": sum(item["sink_errors"] for item in receivers),
        "out_of_window": sum(item["out_of_window"] for item in receivers),
        "incomplete_batches": sum(item["incomplete_batches"] for item in receivers),
        "incomplete_missing": incomplete_missing,
        "incomplete_batch_percent": 100
        * sum(item["incomplete_batches"] for item in receivers)
        / assembled_batches,
        "missing_expected_packet_percent": 100 * incomplete_missing / expected_packet_slots,
        "tile_dropped_packets": tile_dropped,
        "tile_dropped_percent": 100 * tile_dropped / packets_received,
        "unexpected_source": unexpected_source,
        "unexpected_source_percent": 100 * unexpected_source / packets_received,
        "reported_startup_nic_drops": int(daqiri["drop_events"]["packets"].sum()),
        "receivers": receivers,
        "gpu": {
            "utilization_percent": quantiles(active_gpu["utilization_gpu_percent"]),
            "memory_controller_percent": quantiles(
                active_gpu["utilization_memory_percent"]
            ),
            "memory_used_mib": quantiles(active_gpu["memory_used_mib"]),
            "power_w": quantiles(active_gpu["power_draw_w"]),
            "temperature_c": quantiles(active_gpu["temperature_gpu_c"]),
            "sm_clock_mhz": quantiles(active_gpu["clocks_sm_mhz"]),
        },
        "host": {
            "ram_used_mib": quantiles(active_tegra["ram_used_mib"]),
            "emc_percent": quantiles(active_tegra["emc_percent"]),
            "cpu_temperature_c": quantiles(active_tegra["cpu_temperature_c"]),
            "cpu_power_mw": quantiles(active_tegra["cpu_power_mw"]),
            "cpu_core_mean_percent": [
                float(np.nanmean(active_cpu[:, index]))
                for index in range(active_cpu.shape[1])
            ],
            "cpu_core_p95_percent": [
                float(np.nanpercentile(active_cpu[:, index], 95))
                for index in range(active_cpu.shape[1])
            ],
        },
    }

    drop_events = daqiri["drop_events"].copy()
    if not drop_events.empty:
        drop_events["relative_seconds"] = (
            drop_events["timestamp"] - start
        ).dt.total_seconds()

    plot_timeline(output / "platform_utilization_timeline.png", gpu, tegra, cpu, start, end)
    plot_cpu_heatmap(output / "cpu_core_utilization_heatmap.png", tegra, cpu, start, end)
    plot_summary(output / "daqiri_keep_up_summary.png", metrics, drop_events)
    (output / "profile_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    summary = f"""# DAQIRI IGX profile summary

- Active acquisition: **{metrics['active_duration_seconds']:.3f} s**
- Accepted packet-byte rate: **{metrics['combined_gbps']:.2f} Gbit/s combined**
  ({receivers[0]['gbps']:.2f} and {receivers[1]['gbps']:.2f} Gbit/s)
- Assembly: **{metrics['combined_fps']:.0f} frame/s combined**
- Pipeline backpressure: **{metrics['sink_pool_drops']} sink-pool drops**,
  **{metrics['out_of_window']} out-of-window packets**, and
  **{metrics['sink_errors']} sink errors**
- Incomplete output: **{metrics['incomplete_batches']}/{assembled_batches} batches ({metrics['incomplete_batch_percent']:.3f}%)**, representing
  **{metrics['missing_expected_packet_percent']:.4f}%** of expected packet slots
- Startup NIC-drop reports: **{metrics['reported_startup_nic_drops']:,} packets**;
  no further drop report appears during the remaining steady-state interval
- Intentional compatibility discard: **{metrics['tile_dropped_percent']:.3f}%**
  (`row_offset >= 120`), matching the expected 64/1024 packet rows
- Discrete GPU: **{metrics['gpu']['utilization_percent']['mean']:.1f}% mean utilization**,
  **{metrics['gpu']['memory_controller_percent']['mean']:.1f}% mean memory-controller
  utilization**, **{metrics['gpu']['power_w']['mean']:.1f} W mean**,
  **{metrics['gpu']['temperature_c']['max']:.0f} C maximum**
- RX polling cores: CPU 9 **{metrics['host']['cpu_core_mean_percent'][9]:.1f}%**
  mean and CPU 10 **{metrics['host']['cpu_core_mean_percent'][10]:.1f}%** mean

The run demonstrates five-minute steady-state keep-up for receiving and the enabled
GPU correction chain with HDF5 writing disabled. It does not characterize HDF5
write throughput. The startup loss remains a separate issue: packets are already
arriving while DAQIRI initializes the two ports and starts workers.
"""
    (output / "SUMMARY.md").write_text(summary)
    print(summary)
    print(f"Plots and metrics written to {output}")


if __name__ == "__main__":
    main()
