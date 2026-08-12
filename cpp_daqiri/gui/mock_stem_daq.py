#!/usr/bin/env python3
"""Synthetic PUB/REP server for developing the STEM DAQ GUI without hardware."""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import time

import numpy as np
import zmq


def initial_state(pub_endpoint: str, control_endpoint: str) -> dict:
    processor = {
        "noop": True,
        "subtract_dark_frame": True,
        "dark_frame_path": "/calibration/dark.h5",
        "dark_frame_dataset": "/processed",
        "apply_valid_pixel_mask": True,
        "valid_pixel_mask_dataset": "/valid_pixel_mask",
        "apply_blr_correction": True,
        "blr_rows": 30,
        "blr_zlp_width": 768,
        "blr_zlp_group_columns": 4,
        "blr_core_group_columns": 16,
        "apply_dynamic_half_column_mask": True,
        "dynamic_mask_median_window_pixels": 31,
        "dynamic_mask_threshold_ratio": 1.0,
        "dynamic_mask_threshold_offset": 500.0,
        "dynamic_mask_excluded_edge_rows": 32,
        "dynamic_mask_two_sided": True,
    }
    effective = {
        "source": "network",
        "num_receivers": 2,
        "stem_rx": {
            "frames_per_tensor": 128,
            "header_size": 42,
            "payload_size": 7680,
            "expected_source_mask": 13,
            "batch_close_slack_packets": 512,
            "gpu_header_extract": True,
            "hds": False,
            "tile_duplicate_prefix_to_simulate_payload": True,
            "capture_latency": False,
        },
        "processor": processor,
        "writer": {
            "noop": True,
            "filepath": "/data/stem_continuous.h5",
            "dataset_name": "/processed",
            "num_concurrent": 3,
        },
        "burst_writer": {
            "enabled": True,
            "start_armed": False,
            "processing_stage": "corrected",
            "filepath_template": "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5",
            "dataset_name": "/frames",
            "buckets_per_capture": 2,
            "capture_count": 10,
            "rearm_after_write": True,
            "strict_complete": False,
            "threshold": {"zlp": 0.0, "core_loss": 0.0},
        },
        "thinned_stream": {
            "enabled": True,
            "start_publishing": True,
            "processing_stage": "corrected",
            "endpoint": pub_endpoint,
            "topic_prefix": "stem",
            "total_refresh_hz": 2.0,
            "representative_frame_index": 64,
            "include_representative_frame": True,
            "include_bucket_sum": True,
            "queue_depth": 2,
            "threshold": {"zlp": 0.0, "core_loss": 0.0},
        },
        "control": {"enabled": True, "endpoint": control_endpoint},
    }
    return {
        "ok": True,
        "schema": "stem.control.v1",
        "message": "",
        "acquisition": {
            "running": True,
            "restart_pending": False,
            "restart_requested": False,
        },
        "burst_writer": {
            "capability_enabled": True,
            "armed": False,
            "busy": False,
            "output_float32": True,
            "capacity_buckets": 2,
            "processing_stage": "corrected",
            "filepath_template": effective["burst_writer"]["filepath_template"],
            "dataset_name": "/frames",
            "buckets_per_capture": 1,
            "capture_count": 10,
            "rearm_after_write": True,
            "strict_complete": False,
            "threshold": {"zlp": 0.0, "core_loss": 0.0},
            "stats": {
                "captures_started": 0,
                "captures_written": 0,
                "buckets_captured": 0,
                "buckets_skipped_busy": 0,
                "rejected_incomplete": 0,
                "aborted": 0,
                "errors": 0,
            },
        },
        "thinned_stream": {
            "capability_enabled": True,
            "publishing": True,
            "endpoint": pub_endpoint,
            "queue_depth": 2,
            "processing_stage": "corrected",
            "topic_prefix": "stem",
            "total_refresh_hz": 2.0,
            "representative_frame_index": 64,
            "include_representative_frame": True,
            "include_bucket_sum": True,
            "threshold": {"zlp": 0.0, "core_loss": 0.0},
            "stats": {
                "products_queued": 0,
                "products_published": 0,
                "products_coalesced": 0,
                "dropped_no_buffer": 0,
                "send_errors": 0,
            },
        },
        "effective_config": effective,
        "pending_config": copy.deepcopy(effective),
    }


def merge(target: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value


def handle(state: dict, request: dict) -> dict:
    command = request.get("command", "get_state")
    if command == "set_runtime":
        if "thinned_stream" in request:
            update = request["thinned_stream"]
            state["thinned_stream"].update(update)
            root = state["effective_config"]["thinned_stream"]
            root.update(update)
            state["pending_config"]["thinned_stream"].update(update)
        if "burst_writer" in request:
            update = dict(request["burst_writer"])
            action = update.pop("action", "")
            state["burst_writer"].update(update)
            state["effective_config"]["burst_writer"].update(update)
            state["pending_config"]["burst_writer"].update(update)
            if action == "arm":
                state["burst_writer"]["armed"] = True
                state["burst_writer"]["stats"]["captures_started"] += 1
            elif action in {"disarm", "abort"}:
                state["burst_writer"]["armed"] = False
        return state
    if command == "stage_restart":
        merge(state["pending_config"], request["updates"])
        state["acquisition"]["restart_pending"] = True
        return state
    if command == "discard_restart":
        state["pending_config"] = copy.deepcopy(state["effective_config"])
        state["acquisition"]["restart_pending"] = False
        return state
    if command == "restart":
        state["effective_config"] = copy.deepcopy(state["pending_config"])
        state["acquisition"]["restart_pending"] = False
        state["message"] = "mock restart complete"
        return state
    if command == "shutdown":
        state["acquisition"]["running"] = False
        return state
    if command != "get_state":
        return {"ok": False, "error": f"unknown command {command}"}
    return state


def synthetic_frame(receiver: int, batch: int, height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    center = (batch * 17 + receiver * 71) % width
    peak = 1200 * np.exp(-0.5 * ((columns - center) / 16) ** 2)
    band = 80 * np.sin(columns / 47 + batch / 11)
    split = np.where(rows < height / 2, 1.0, 0.72)
    return (peak + band + rows * 0.08) * split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pub-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(args.pub_endpoint)
    control = context.socket(zmq.REP)
    control.bind(args.control_endpoint)
    poller = zmq.Poller()
    poller.register(control, zmq.POLLIN)
    state = initial_state(args.pub_endpoint, args.control_endpoint)
    running = True

    def stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    batch = 0
    next_product = time.monotonic() + 0.3
    print(
        f"Mock STEM DAQ: PUB {args.pub_endpoint}, REP {args.control_endpoint}",
        flush=True,
    )
    while running:
        events = dict(poller.poll(20))
        if control in events:
            try:
                reply = handle(state, control.recv_json())
            except Exception as error:
                reply = {"ok": False, "error": str(error)}
            control.send_json(reply)
        if (
            state["thinned_stream"]["publishing"]
            and time.monotonic() >= next_product
        ):
            receiver = batch % 2
            frame = synthetic_frame(receiver, batch, args.height, args.width)
            bucket_sum = frame * (120 + 8 * math.sin(batch / 9))
            stage = state["thinned_stream"]["processing_stage"]
            topic = f"{state['thinned_stream']['topic_prefix']}/rx/{receiver}/{stage}"
            parts = ["metadata"]
            payloads: list[bytes] = []
            if state["thinned_stream"]["include_representative_frame"]:
                parts.append("representative")
                payloads.append(frame.astype("<f4").tobytes())
            if state["thinned_stream"]["include_bucket_sum"]:
                parts.append("sum")
                payloads.append(bucket_sum.astype("<f4").tobytes())
            metadata = {
                "schema": "stem.thinned.v1",
                "receiver_id": receiver,
                "interface_name": f"mock_rx_{receiver}",
                "batch_index": batch,
                "first_frame": batch * 128,
                "frames_in_bucket": 128,
                "complete": True,
                "received_packets": 46080,
                "expected_packets": 46080,
                "processing_stage": stage,
                "dtype": "float32",
                "byte_order": "little",
                "height": args.height,
                "width": args.width,
                "representative_frame_index": state["thinned_stream"][
                    "representative_frame_index"
                ],
                "parts": parts,
            }
            publisher.send_multipart(
                [
                    topic.encode(),
                    json.dumps(metadata).encode(),
                    *payloads,
                ]
            )
            state["thinned_stream"]["stats"]["products_queued"] += 1
            state["thinned_stream"]["stats"]["products_published"] += 1
            batch += 1
            rate = max(0.1, float(state["thinned_stream"]["total_refresh_hz"]))
            next_product = time.monotonic() + 1 / rate

    publisher.close()
    control.close()


if __name__ == "__main__":
    main()
