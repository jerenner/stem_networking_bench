#!/usr/bin/env python3
"""Exercise the real synthetic RX executable without NICs; requires CUDA + HDF5.

Run inside the built container (or against a native build). Python dependencies:
numpy, h5py, pyyaml. Temporary HDF5 files/configs are removed on success.
"""

import argparse
import copy
from pathlib import Path
import re
import subprocess
import tempfile

import h5py
import numpy as np
import yaml


def reference_frame(receiver, frame, mask=255, legacy=False, pattern="ramp"):
    """Independent scalar-tile placement reference, with uint16 input semantics."""
    result = np.zeros((1024, 3840), dtype=np.uint16)
    sources = bin(mask).count("1")
    for tile in range(sources * 120):
        zlp = tile < 192
        local = tile if zlp else tile - 192
        height, width = (128, 32) if zlp else (32, 128)
        row = (local // 24) * height
        col = (local % 24) * width + (0 if zlp else 768)
        sample = np.arange(4096, dtype=np.uint32)
        if legacy:
            sample[3840:] -= 3840
        if pattern == "walking_dot":
            dot = ((frame % 128) * 31 + tile * 7 + receiver * 13) % 4096
            values = np.where(sample == dot, 20000, 100 + receiver)
        else:
            values = 64 + ((receiver * 977 + (frame % 128) * 37 + tile * 17 + sample * 5) % 4096)
        result[row:row + height, col:col + width] = values.reshape(height, width)
    return result


def parse_results(output):
    results = {}
    for line in output.splitlines():
        if line.startswith("synthetic complete "):
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
            receiver = int(fields["rx"])
            if receiver in results:
                raise AssertionError(f"Duplicate result for receiver {receiver}")
            results[receiver] = fields
    return results


def execute(binary, config, directory, name, timeout, expect_error=None):
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [str(binary), str(path)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = run.stdout + run.stderr
    (directory / f"{name}.log").write_text(output, encoding="utf-8")
    if expect_error is not None:
        if run.returncode == 0 or expect_error not in output:
            raise AssertionError(f"{name}: expected rejection containing {expect_error!r}\n{output}")
        print(f"PASS {name}: rejected invalid configuration", flush=True)
        return output
    if run.returncode:
        raise AssertionError(f"{name}: exit {run.returncode}\n{output}")
    return output


def check_results(output, config):
    results = parse_results(output)
    count = config["num_receivers"]
    if set(results) != set(range(count)):
        raise AssertionError(f"Missing receiver results: {results}\n{output}")
    frames = config["stem_rx"]["frames_per_tensor"] * config["synthetic"]["buckets_per_receiver"]
    sources = bin(config["stem_rx"]["expected_source_mask"]).count("1")
    legacy = config["stem_rx"]["tile_duplicate_prefix_to_simulate_payload"]
    for fields in results.values():
        for key in ("validation_mismatches", "incomplete", "pool_drops", "unexpected", "trailing_packets"):
            if int(fields[key]):
                raise AssertionError(f"{key} is nonzero: {fields}")
        if int(fields["frames"]) != frames:
            raise AssertionError(f"Wrong frame count: {fields}")
        if int(fields["packets"]) != frames * sources * (128 if legacy else 120):
            raise AssertionError(f"Wrong packet count: {fields}")
        if int(fields["ignored"]) != (frames * sources * 8 if legacy else 0):
            raise AssertionError(f"Wrong legacy-discard count: {fields}")


def run_case(binary, base, directory, name, timeout, *, mask=255, legacy=False,
             receivers=2, pattern="ramp", correction=False, wrap=False):
    config = copy.deepcopy(base)
    config["num_receivers"] = receivers
    config["stem_rx"].update(expected_source_mask=mask,
                             payload_size=7680 if legacy else 8192,
                             tile_duplicate_prefix_to_simulate_payload=legacy)
    config["synthetic"]["payload_pattern"] = pattern
    if wrap:
        config["synthetic"]["buckets_per_receiver"] = 65  # 130 frames crosses header wrap.
        config["burst_writer"]["enabled"] = False
    else:
        config["burst_writer"]["filepath_template"] = str(directory / f"{name}_rx{{receiver}}.h5")
    if correction:
        config["processor"].update(subtract_dark_frame=True, apply_valid_pixel_mask=True,
                                    dark_frame_path=str(directory / "calibration.h5"))
        config["burst_writer"]["processing_stage"] = "corrected"
    output = execute(binary, config, directory, name, timeout)
    check_results(output, config)
    if not wrap:
        for receiver in range(receivers):
            with h5py.File(directory / f"{name}_rx{receiver}.h5", "r") as handle:
                dataset = handle["frames"]
                if dataset.shape != (6, 1024, 3840):
                    raise AssertionError(f"Wrong saved shape: {dataset.shape}")
                if dataset.dtype != np.dtype(np.float32 if correction else np.uint16):
                    raise AssertionError(f"Wrong saved dtype: {dataset.dtype}")
                if int(dataset.attrs["receiver_id"]) != receiver or int(dataset.attrs["first_frame"]) != 0:
                    raise AssertionError("Receiver/frame metadata mismatch")
                for frame in range(6):
                    expected = reference_frame(receiver, frame, mask, legacy, pattern)
                    if correction:
                        expected = expected.astype(np.float32) - np.float32(100.5)
                        expected[100, 100] = 0  # Matches calibration valid-pixel mask.
                    np.testing.assert_array_equal(dataset[frame], expected)
            (directory / f"{name}_rx{receiver}.h5").unlink()
    print(f"PASS {name}: packets, completed frames, GPU validation"
          + (", saved pixels and metadata" if not wrap else ", 128-frame wrap"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("/opt/stem_daqiri/bin/stem_daqiri_rx"))
    parser.add_argument("--timeout", type=float, default=120, help="Timeout per executable invocation")
    parser.add_argument("--workdir", type=Path, help="Keep logs/configs in this new directory")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"Binary not found: {args.binary}")
    temporary = None
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=False)
        directory = args.workdir.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="stem-synthetic-")
        directory = Path(temporary.name)
    print(f"Validation workdir: {directory}", flush=True)
    base = {
        "source": "synthetic", "num_receivers": 2,
        "synthetic": {"mode": "packet", "rate_mode": "maximum", "duration_seconds": -1,
                      "buckets_per_receiver": 3, "packets_per_burst": 733, "validate_output": True},
        "stem_rx": {"frames_per_tensor": 2, "expected_source_mask": 255, "payload_size": 8192,
                    "gpu_header_extract": True, "tile_duplicate_prefix_to_simulate_payload": False},
        "processor": {"noop": True, "subtract_dark_frame": False, "apply_valid_pixel_mask": False,
                      "apply_blr_correction": False, "apply_dynamic_half_column_mask": False},
        "writer": {"noop": True},
        "burst_writer": {"enabled": True, "processing_stage": "raw", "dataset_name": "/frames",
                         "buckets_per_capture": 3, "capture_count": 1, "rearm_after_write": False,
                         "strict_complete": True},
    }
    try:
        # These fail during parsing, before allocating GPU memory.
        for count in (0, 9):
            bad = copy.deepcopy(base)
            bad["num_receivers"] = count
            execute(args.binary.resolve(), bad, directory, f"bad_receivers_{count}", args.timeout,
                    expect_error="num_receivers")
        bad = copy.deepcopy(base)
        bad["stem_rx"]["payload_size"] = 7680
        execute(args.binary.resolve(), bad, directory, "bad_payload", args.timeout,
                expect_error="payload_size")
        with h5py.File(directory / "calibration.h5", "w") as handle:
            handle.create_dataset("processed", data=np.full((1, 1024, 3840), 100.5, dtype=np.float32))
            valid = np.ones((1024, 3840), dtype=np.uint8)
            valid[100, 100] = 0
            handle.create_dataset("valid_pixel_mask", data=valid)
        run_case(args.binary.resolve(), base, directory, "native_dual", args.timeout)
        run_case(args.binary.resolve(), base, directory, "legacy_sparse", args.timeout, mask=13, legacy=True)
        run_case(args.binary.resolve(), base, directory, "walking_dot", args.timeout, receivers=1,
                 pattern="walking_dot")
        run_case(args.binary.resolve(), base, directory, "dark_and_static_mask", args.timeout,
                 receivers=1, correction=True)
        # Eight workers without requiring eight full-frame packet pools.
        run_case(args.binary.resolve(), base, directory, "eight_receivers_wrap", args.timeout,
                 mask=128, receivers=8, wrap=True)
        print("All synthetic GPU integration checks passed.", flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
