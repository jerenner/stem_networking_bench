#!/usr/bin/env python3
"""Inspect or record the DAQIRI thinned ZeroMQ multipart stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5556",
        help="publisher endpoint using the IGX address (default: %(default)s)",
    )
    parser.add_argument(
        "--topic",
        default="stem/",
        help="ZeroMQ subscription prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="stop after N products; 0 runs until Ctrl-C",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=0,
        help="receive timeout; 0 waits indefinitely",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="optionally save metadata JSON and raw little-endian float32 arrays",
    )
    return parser.parse_args()


def require_dependencies():
    try:
        import numpy as np
        import zmq
    except ImportError as error:
        raise SystemExit(
            "This utility requires NumPy and pyzmq (for example: pip install numpy pyzmq)"
        ) from error
    return np, zmq


def decode_message(parts: list[bytes], np):
    if len(parts) < 3:
        raise ValueError(f"expected at least 3 multipart sections, received {len(parts)}")
    topic = parts[0].decode("utf-8")
    metadata = json.loads(parts[1].decode("utf-8"))
    if metadata.get("schema") != "stem.thinned.v1":
        raise ValueError(f"unsupported schema: {metadata.get('schema')!r}")
    height = int(metadata["height"])
    width = int(metadata["width"])
    expected_bytes = height * width * np.dtype("<f4").itemsize
    payload_names = [name for name in metadata["parts"] if name != "metadata"]
    if len(payload_names) != len(parts) - 2:
        raise ValueError("metadata parts list does not match multipart payload")

    arrays = {}
    for name, payload in zip(payload_names, parts[2:], strict=True):
        if len(payload) != expected_bytes:
            raise ValueError(
                f"{name}: expected {expected_bytes} bytes, received {len(payload)}"
            )
        arrays[name] = np.frombuffer(payload, dtype="<f4").reshape(height, width).copy()
    return topic, metadata, arrays


def save_message(save_dir: Path, index: int, topic: str, metadata: dict, arrays) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"product_{index:06d}_rx{metadata['receiver_id']}_{metadata['processing_stage']}"
    payload = dict(metadata)
    payload["topic"] = topic
    (save_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, array in arrays.items():
        # Importing NumPy is deliberately deferred by require_dependencies().
        array.tofile(save_dir / f"{stem}_{name}.float32")


def main() -> None:
    args = parse_args()
    if args.count < 0 or args.timeout_ms < 0:
        raise SystemExit("--count and --timeout-ms must be non-negative")
    np, zmq = require_dependencies()

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, args.topic.encode("utf-8"))
    subscriber.setsockopt(zmq.RCVHWM, 2)
    subscriber.setsockopt(zmq.LINGER, 0)
    if args.timeout_ms:
        subscriber.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    subscriber.connect(args.endpoint)
    print(f"Connected to {args.endpoint}; subscribed to {args.topic!r}")

    received = 0
    try:
        while args.count == 0 or received < args.count:
            try:
                parts = subscriber.recv_multipart()
            except zmq.Again as error:
                raise SystemExit(
                    f"No thinned product received within {args.timeout_ms} ms"
                ) from error
            topic, metadata, arrays = decode_message(parts, np)
            received += 1
            summaries = []
            for name, array in arrays.items():
                summaries.append(
                    f"{name}: min={float(np.min(array)):.3g} "
                    f"max={float(np.max(array)):.3g} sum={float(np.sum(array)):.6g}"
                )
            print(
                f"[{received}] {topic} batch={metadata['batch_index']} "
                f"first_frame={metadata['first_frame']} "
                f"complete={metadata['complete']} | " + "; ".join(summaries)
            )
            if args.save_dir:
                save_message(args.save_dir, received, topic, metadata, arrays)
    except KeyboardInterrupt:
        print(f"\nStopped after {received} product(s)")
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
