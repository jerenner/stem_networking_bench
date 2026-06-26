#!/usr/bin/env python3
"""Run the Holoscan-equivalent STEM processing chain offline on HDF5 frames.

This is the canonical Python entry point for reproducing the `PyTorchProcessorOp`
math outside Holoscan. It processes input frames in tensor-sized chunks and uses
the same correction order as the runtime operator:

    float32 conversion -> optional dark subtraction -> optional grouped BLR
    -> optional valid-pixel mask -> optional dynamic half-column mask
    -> noop/pass-through or sum-output mode
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stem_analysis import ProcessorConfig, process_tensor_block
from stem_analysis.hdf5 import create_dataset_with_groups, normalize_dataset_path, read_single_image


def parse_bool(text: str) -> bool:
    value = text.lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {text!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input HDF5 frame stack.")
    parser.add_argument("output", type=Path, help="Output HDF5 file.")
    parser.add_argument("--input-dataset", default="/frames")
    parser.add_argument("--output-dataset", default="/processed")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--frames-per-tensor", type=int, default=128)
    parser.add_argument("--noop", type=parse_bool, default=True)
    parser.add_argument("--subtract-dark-frame", type=parse_bool, default=False)
    parser.add_argument("--dark-frame-path", type=Path, default=None)
    parser.add_argument("--dark-frame-dataset", default="/processed")
    parser.add_argument("--apply-valid-pixel-mask", type=parse_bool, default=False)
    parser.add_argument("--valid-pixel-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--apply-blr-correction", type=parse_bool, default=False)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--blr-zlp-width", type=int, default=768)
    parser.add_argument("--blr-zlp-group-columns", type=int, default=4)
    parser.add_argument("--blr-core-group-columns", type=int, default=16)
    parser.add_argument("--apply-dynamic-half-column-mask", type=parse_bool, default=False)
    parser.add_argument("--dynamic-mask-median-window-pixels", type=int, default=31)
    parser.add_argument("--dynamic-mask-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--dynamic-mask-threshold-offset", type=float, default=500.0)
    parser.add_argument("--dynamic-mask-excluded-edge-rows", type=int, default=32)
    parser.add_argument("--dynamic-mask-two-sided", type=parse_bool, default=True)
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"), default="none")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ProcessorConfig:
    return ProcessorConfig(
        noop=args.noop,
        subtract_dark_frame=args.subtract_dark_frame,
        apply_valid_pixel_mask=args.apply_valid_pixel_mask,
        apply_blr_correction=args.apply_blr_correction,
        blr_rows=args.blr_rows,
        blr_zlp_width=args.blr_zlp_width,
        blr_zlp_group_columns=args.blr_zlp_group_columns,
        blr_core_group_columns=args.blr_core_group_columns,
        apply_dynamic_half_column_mask=args.apply_dynamic_half_column_mask,
        dynamic_mask_median_window_pixels=args.dynamic_mask_median_window_pixels,
        dynamic_mask_threshold_ratio=args.dynamic_mask_threshold_ratio,
        dynamic_mask_threshold_offset=args.dynamic_mask_threshold_offset,
        dynamic_mask_excluded_edge_rows=args.dynamic_mask_excluded_edge_rows,
        dynamic_mask_two_sided=args.dynamic_mask_two_sided,
    )


def main() -> None:
    args = parse_args()
    if args.frames_per_tensor <= 0:
        raise ValueError("--frames-per-tensor must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")

    import h5py
    import numpy as np

    config = build_config(args)
    compression = None if args.compression == "none" else args.compression
    dark_frame = None
    valid_pixel_mask = None

    if config.subtract_dark_frame or config.apply_valid_pixel_mask:
        if args.dark_frame_path is None:
            raise ValueError("--dark-frame-path is required for dark subtraction or valid masking")
        with h5py.File(args.dark_frame_path, "r") as dark_h5:
            if config.subtract_dark_frame:
                dark_frame = read_single_image(dark_h5, args.dark_frame_dataset, np)
            if config.apply_valid_pixel_mask:
                valid_pixel_mask = read_single_image(
                    dark_h5, args.valid_pixel_mask_dataset, np
                ).astype(np.float32, copy=False)

    with h5py.File(args.input, "r") as input_h5:
        input_dataset = input_h5[normalize_dataset_path(args.input_dataset, "/frames")]
        if input_dataset.ndim != 3:
            raise ValueError(f"input dataset must have shape [frames, rows, cols], got {input_dataset.shape}")

        input_frame_count = int(input_dataset.shape[0])
        end_frame = input_frame_count if args.frames is None else min(
            input_frame_count, args.start_frame + args.frames
        )
        if args.start_frame >= end_frame:
            raise ValueError("no frames selected")

        selected_frames = end_frame - args.start_frame
        output_frame_count = selected_frames if config.noop else 0
        if not config.noop:
            output_frame_count = (selected_frames + args.frames_per_tensor - 1) // args.frames_per_tensor

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as output_h5:
            output_dataset = create_dataset_with_groups(
                output_h5,
                args.output_dataset,
                shape=(output_frame_count, input_dataset.shape[1], input_dataset.shape[2]),
                dtype=np.float32,
                chunks=(min(args.frames_per_tensor, output_frame_count), input_dataset.shape[1], input_dataset.shape[2]),
                compression=compression,
            )
            output_offset = 0
            total_masked = 0
            for start in range(args.start_frame, end_frame, args.frames_per_tensor):
                stop = min(end_frame, start + args.frames_per_tensor)
                block = input_dataset[start:stop]
                processed, zero_mask = process_tensor_block(
                    block, dark_frame, valid_pixel_mask, config, np
                )
                output_dataset[output_offset:output_offset + processed.shape[0]] = processed
                output_offset += processed.shape[0]
                total_masked += int(zero_mask.sum())

            output_dataset.attrs["description"] = "Offline NumPy reproduction of PyTorchProcessorOp output"
            output_dataset.attrs["source_file"] = str(args.input)
            output_dataset.attrs["source_dataset"] = normalize_dataset_path(args.input_dataset, "/frames")
            output_dataset.attrs["start_frame"] = args.start_frame
            output_dataset.attrs["frames"] = selected_frames
            output_dataset.attrs["frames_per_tensor"] = args.frames_per_tensor
            output_dataset.attrs["processor_config_json"] = json.dumps(asdict(config), sort_keys=True)
            output_dataset.attrs["masked_pixels_in_last_batches_sum"] = total_masked

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "frames": selected_frames,
                "output_frames": output_frame_count,
                "config": asdict(config),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

