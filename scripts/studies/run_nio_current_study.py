#!/usr/bin/env python3
"""Discover and run the manifest-driven NiO beam-current study."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack


FILENAME_RE = re.compile(
    r"^NiO (?P<current>[0-9]+(?:pA|nA)) (?P<kind>Dark|Spectrum) (?P<index>[0-9]{4})\.dm4$",
    re.IGNORECASE,
)


def current_pa(label: str) -> int:
    value = int(re.match(r"[0-9]+", label).group())
    return value * 1000 if label.lower().endswith("na") else value


def current_key(label: str) -> str:
    return f"{current_pa(label):04d}pA"


@dataclass
class SourceFile:
    current_label: str
    current_pa: int
    current_key: str
    kind: str
    index: int
    path: str
    size_bytes: int
    frames: int
    height: int
    width: int
    dtype: str


def inspect_sources(raw_dir: Path, reader: str):
    records = []
    ignored = []
    for path in sorted(raw_dir.glob("*.dm4")):
        match = FILENAME_RE.match(path.name)
        if not match:
            ignored.append(str(path))
            continue
        label = match.group("current")
        data, _ = load_dm4(path, reader)
        stack = normalize_to_frame_stack(data, None, 960, 3840)
        records.append(
            SourceFile(
                current_label=label,
                current_pa=current_pa(label),
                current_key=current_key(label),
                kind=match.group("kind").lower(),
                index=int(match.group("index")),
                path=str(path.resolve()),
                size_bytes=path.stat().st_size,
                frames=int(stack.shape[0]),
                height=int(stack.shape[1]),
                width=int(stack.shape[2]),
                dtype=str(stack.dtype),
            )
        )
        del stack, data
    return sorted(records, key=lambda item: (item.current_pa, item.kind, item.index)), ignored


def write_manifest(output_root: Path, records, ignored):
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys()) if records else []
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    groups = {}
    for record in records:
        group = groups.setdefault(
            record.current_key,
            {
                "current_label": record.current_label,
                "current_pa": record.current_pa,
                "dark_files": [],
                "spectrum_files": [],
            },
        )
        group[f"{record.kind}_files"].append(asdict(record))
    for group in groups.values():
        for kind in ("dark", "spectrum"):
            files = group[f"{kind}_files"]
            indices = [item["index"] for item in files]
            group[f"{kind}_frames"] = sum(item["frames"] for item in files)
            group[f"{kind}_missing_indices"] = (
                sorted(set(range(1, max(indices) + 1)) - set(indices)) if indices else []
            )

    manifest = {
        "raw_dir": str(records[0].path.rsplit("/", 1)[0]) if records else None,
        "file_count": len(records),
        "ignored_files": ignored,
        "currents": dict(sorted(groups.items(), key=lambda item: item[1]["current_pa"])),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_command(command):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reader", default="rsciio")
    parser.add_argument("--currents", nargs="*", default=None, help="Optional labels or canonical keys.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=8)
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--blinker-threshold", type=float, default=500.0)
    parser.add_argument("--dynamic-threshold", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    records, ignored = inspect_sources(args.raw_dir, args.reader)
    if not records:
        raise ValueError(f"no matching DM4 files found in {args.raw_dir}")
    manifest = write_manifest(args.output_root, records, ignored)

    config = {
        "reader": args.reader,
        "read_chunk_size": args.read_chunk_size,
        "tensor_frames": args.tensor_frames,
        "frame_shape": [960, 3840],
        "edge_rows": 32,
        "blr_rows": 30,
        "zlp_width": 768,
        "zlp_period": 192,
        "zlp_group_columns": 4,
        "core_group_columns": 16,
        "blinker_std_threshold": args.blinker_threshold,
        "dynamic_mask_median_window_pixels": 31,
        "dynamic_mask_threshold_ratio": 1.0,
        "dynamic_mask_threshold_offset": args.dynamic_threshold,
        "dynamic_mask_two_sided": True,
    }
    (args.output_root / "study_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    selected = None
    if args.currents:
        selected = set()
        for value in args.currents:
            selected.add(value if value.endswith("pA") and value[:1] == "0" else current_key(value))

    completed = []
    for key, group in manifest["currents"].items():
        if selected is not None and key not in selected:
            continue
        if not group["dark_files"] or not group["spectrum_files"]:
            print(f"Skipping {key}: missing dark or spectrum files", flush=True)
            continue
        current_dir = args.output_root / "currents" / key
        dark_dir = current_dir / "dark"
        spectrum_dir = current_dir / "spectrum"
        dark_h5 = dark_dir / "dark_frame.h5"
        spectrum_h5 = spectrum_dir / "final_spectrum.h5"
        current_dir.mkdir(parents=True, exist_ok=True)
        (current_dir / "current_metadata.json").write_text(
            json.dumps(group, indent=2), encoding="utf-8"
        )

        if args.force or not dark_h5.exists():
            run_command([
                sys.executable,
                script_dir / "make_dark_frame_from_dm4.py",
                *[item["path"] for item in group["dark_files"]],
                "--output", dark_h5,
                "--output-dir", dark_dir,
                "--reader", args.reader,
                "--read-chunk-size", args.read_chunk_size,
                "--blinker-std-threshold", args.blinker_threshold,
            ])
        else:
            print(f"Reusing {dark_h5}", flush=True)

        if args.force or not spectrum_h5.exists():
            run_command([
                sys.executable,
                script_dir / "analyze_spectrum_dm4.py",
                *[item["path"] for item in group["spectrum_files"]],
                "--dark-frame", dark_h5,
                "--output-dir", spectrum_dir,
                "--reader", args.reader,
                "--tensor-frames", args.tensor_frames,
                "--read-chunk-size", args.read_chunk_size,
                "--dynamic-threshold-offset", args.dynamic_threshold,
            ])
        else:
            print(f"Reusing {spectrum_h5}", flush=True)
        completed.append(key)

    if not args.skip_comparison:
        run_command([
            sys.executable,
            script_dir / "compare_nio_current_study.py",
            "--study-root", args.output_root,
        ])

    run_summary = {
        "completed_currents": completed,
        "force": args.force,
        "comparison_generated": not args.skip_comparison,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
