#!/usr/bin/env python3
"""Stream one current's DM4 spectra into no-BLR/grouped-BLR stitch summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_dm4_to_hdf5 import load_dm4, normalize_to_frame_stack
from plot_nio_processing_analysis import configure_matplotlib_cache, subtract_imagej_blr


MODE_NAMES = ("no_blr", "grouped_blr")


def read_image(h5_file, dataset: str, np):
    data = h5_file[dataset][...]
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"expected one image at {dataset}, got {data.shape}")
    return data.astype(np.float32, copy=False)


def materialize(array, np):
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dark-frame", type=Path, required=True)
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="rsciio"
    )
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--frames-axis", type=int, default=None)
    parser.add_argument("--tensor-frames", type=int, default=128)
    parser.add_argument("--read-chunk-size", type=int, default=8)
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-period", type=int, default=192)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib_cache()
    import h5py
    import numpy as np

    if args.tensor_frames <= 0 or args.read_chunk_size <= 0:
        raise ValueError("tensor and read chunk sizes must be positive")
    if args.height % 2:
        raise ValueError("detector height must be even")
    if args.zlp_width != 4 * args.zlp_period:
        raise ValueError("stitch analysis expects four repeated ZLP reads")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dark_frame, "r") as dark_h5:
        dark = read_image(dark_h5, args.dark_dataset, np)
        valid = read_image(dark_h5, args.valid_mask_dataset, np) != 0
    if dark.shape != (args.height, args.width) or valid.shape != dark.shape:
        raise ValueError("dark frame or valid mask shape does not match requested frame shape")

    half_height = args.height // 2
    regions = (
        slice(args.edge_rows, half_height),
        slice(half_height, args.height - args.edge_rows),
    )
    file_count = len(args.inputs)
    per_file_sums = np.zeros((2, file_count, 2, args.width), dtype=np.float64)
    per_file_counts = np.zeros((file_count, 2, args.width), dtype=np.float64)
    per_file_frames = np.zeros(file_count, dtype=np.int64)
    batch_sums = []
    batch_counts = []
    batch_source_indices = []
    batch_start_frames = []
    batch_end_frames = []

    for file_index, input_path in enumerate(args.inputs):
        data, info = load_dm4(input_path, args.reader)
        stack = normalize_to_frame_stack(
            data, args.frames_axis, args.height, args.width
        )
        print(
            f"{input_path}: reader={info['reader']} stack_shape={tuple(stack.shape)}",
            flush=True,
        )
        for tensor_start in range(0, stack.shape[0], args.tensor_frames):
            tensor_end = min(stack.shape[0], tensor_start + args.tensor_frames)
            frame_count = tensor_end - tensor_start
            raw_sum = np.zeros((args.height, args.width), dtype=np.float64)
            for read_start in range(tensor_start, tensor_end, args.read_chunk_size):
                read_end = min(tensor_end, read_start + args.read_chunk_size)
                raw_sum += materialize(stack[read_start:read_end], np).sum(
                    axis=0, dtype=np.float64
                )

            no_blr = (raw_sum / frame_count).astype(np.float32) - dark
            grouped_blr = subtract_imagej_blr(
                no_blr[None],
                np,
                args.blr_rows,
                args.zlp_width,
                args.zlp_group_columns,
                args.core_group_columns,
            )[0]
            variants = (no_blr, grouped_blr)
            current_sums = np.zeros((2, 2, args.width), dtype=np.float64)
            current_counts = np.zeros((2, args.width), dtype=np.float64)
            for half_index, rows in enumerate(regions):
                region_valid = valid[rows]
                current_counts[half_index] = (
                    frame_count * np.count_nonzero(region_valid, axis=0)
                )
                for mode_index, corrected in enumerate(variants):
                    current_sums[mode_index, half_index] = (
                        np.where(region_valid, corrected[rows], 0.0).sum(
                            axis=0, dtype=np.float64
                        )
                        * frame_count
                    )

            per_file_sums[:, file_index] += current_sums
            per_file_counts[file_index] += current_counts
            per_file_frames[file_index] += frame_count
            batch_sums.append(current_sums)
            batch_counts.append(current_counts)
            batch_source_indices.append(file_index)
            batch_start_frames.append(tensor_start)
            batch_end_frames.append(tensor_end - 1)
            print(f"  batch {tensor_start}..{tensor_end - 1}", flush=True)
        del stack, data

    batch_sums = np.stack(batch_sums)
    batch_counts = np.stack(batch_counts)
    with h5py.File(args.output, "w") as output:
        output.create_dataset("per_file_sums", data=per_file_sums)
        output.create_dataset("per_file_valid_counts", data=per_file_counts)
        output.create_dataset("per_file_frame_count", data=per_file_frames)
        output.create_dataset("per_batch_sums", data=batch_sums)
        output.create_dataset("per_batch_valid_counts", data=batch_counts)
        output.create_dataset(
            "per_batch_source_index",
            data=np.asarray(batch_source_indices, dtype=np.int32),
        )
        output.create_dataset(
            "per_batch_start_frame", data=np.asarray(batch_start_frames, dtype=np.int64)
        )
        output.create_dataset(
            "per_batch_end_frame", data=np.asarray(batch_end_frames, dtype=np.int64)
        )
        output.attrs["mode_names"] = json.dumps(MODE_NAMES)
        output.attrs["source_files"] = json.dumps([str(path) for path in args.inputs])
        output.attrs["dark_frame"] = str(args.dark_frame)
        output.attrs["dynamic_mask_applied"] = False
        output.attrs["static_valid_mask_applied"] = True
        output.attrs["zlp_width"] = args.zlp_width
        output.attrs["zlp_period"] = args.zlp_period
        output.attrs["blr_rows"] = args.blr_rows
        output.attrs["zlp_group_columns"] = args.zlp_group_columns
        output.attrs["core_group_columns"] = args.core_group_columns

    summary = {
        "source_files": [str(path) for path in args.inputs],
        "source_frame_counts": per_file_frames.tolist(),
        "dark_frame": str(args.dark_frame),
        "output": str(args.output),
        "modes": list(MODE_NAMES),
        "batch_count": int(batch_sums.shape[0]),
        "dynamic_mask_applied": False,
        "static_valid_mask_applied": True,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
