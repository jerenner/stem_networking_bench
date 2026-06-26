#!/usr/bin/env python3
"""Run the STEM processor correction chain offline on an HDF5 frame stack.

This is a CPU/NumPy reference implementation of the PyTorchProcessorOp path used
by the Holoscan application. It is meant for analysis/debugging, not throughput:
it mirrors the same operation order so dark-frame choices and masking parameters
can be tested without rerunning on the IGX.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return "/processed"
    return dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"


def create_dataset_with_groups(h5_file, dataset_path: str, **kwargs):
    dataset_path = normalize_dataset_path(dataset_path)
    group_path, dataset_name = dataset_path.rsplit("/", 1)
    group = h5_file if not group_path else h5_file.require_group(group_path)
    return group.create_dataset(dataset_name, **kwargs)


def read_single_frame(h5_file, dataset_path: str, np):
    dataset = h5_file[normalize_dataset_path(dataset_path)]
    data = dataset[...]
    if data.ndim == 2:
        return data.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32, copy=False)
    raise ValueError(
        f"{dataset_path} must have shape [rows, cols] or [1, rows, cols], "
        f"got {data.shape}"
    )


def compute_blr_baseline(block,
                         blr_rows: int,
                         zlp_width: int,
                         zlp_group_columns: int,
                         core_group_columns: int,
                         np):
    """Match compute_blr_baseline_kernel on already dark-subtracted frames."""
    frames, height, width = block.shape
    if blr_rows <= 0 or 2 * blr_rows > height:
        raise ValueError(f"invalid blr_rows={blr_rows} for height={height}")
    if zlp_width > width:
        raise ValueError(f"zlp_width={zlp_width} exceeds frame width={width}")
    if zlp_width % zlp_group_columns != 0:
        raise ValueError("zlp_width must be divisible by zlp_group_columns")
    if (width - zlp_width) % core_group_columns != 0:
        raise ValueError("CoreLoss width must be divisible by core_group_columns")

    zlp_bins = zlp_width // zlp_group_columns
    core_bins = (width - zlp_width) // core_group_columns
    baseline = np.empty((frames, 2, zlp_bins + core_bins), dtype=np.float32)

    for half_idx, rows in enumerate((slice(0, blr_rows), slice(height - blr_rows, height))):
        edge = block[:, rows, :]
        parts = []
        if zlp_width:
            zlp = edge[:, :, :zlp_width].reshape(
                frames, blr_rows, zlp_bins, zlp_group_columns
            )
            parts.append(zlp.mean(axis=(1, 3), dtype=np.float32))
        if zlp_width < width:
            core = edge[:, :, zlp_width:].reshape(
                frames, blr_rows, core_bins, core_group_columns
            )
            parts.append(core.mean(axis=(1, 3), dtype=np.float32))
        baseline[:, half_idx, :] = np.concatenate(parts, axis=1)

    return baseline


def subtract_blr_baseline(block,
                          baseline,
                          zlp_width: int,
                          zlp_group_columns: int,
                          core_group_columns: int,
                          np) -> None:
    """Match correct_with_blr_and_mean_kernel's BLR subtraction in-place."""
    frames, height, width = block.shape
    half_height = height // 2
    zlp_bins = zlp_width // zlp_group_columns

    if zlp_width:
        top_zlp = np.repeat(baseline[:, 0, :zlp_bins], zlp_group_columns, axis=1)
        bottom_zlp = np.repeat(baseline[:, 1, :zlp_bins], zlp_group_columns, axis=1)
        block[:, :half_height, :zlp_width] -= top_zlp[:, None, :]
        block[:, half_height:, :zlp_width] -= bottom_zlp[:, None, :]

    if zlp_width < width:
        top_core = np.repeat(baseline[:, 0, zlp_bins:], core_group_columns, axis=1)
        bottom_core = np.repeat(baseline[:, 1, zlp_bins:], core_group_columns, axis=1)
        block[:, :half_height, zlp_width:] -= top_core[:, None, :]
        block[:, half_height:, zlp_width:] -= bottom_core[:, None, :]


def half_column_local_median(batch_mean,
                             median_window_pixels: int,
                             excluded_edge_rows: int,
                             np):
    """Compute the dynamic mask median exactly like the CUDA half-column kernel."""
    height, width = batch_mean.shape
    if height % 2 != 0:
        raise ValueError("dynamic half-column mask requires even frame height")
    if median_window_pixels <= 0 or median_window_pixels % 2 == 0:
        raise ValueError("median_window_pixels must be a positive odd number")
    if median_window_pixels > 129:
        raise ValueError("median_window_pixels must be <= 129 to match CUDA kernel")
    if 2 * excluded_edge_rows >= height:
        raise ValueError("excluded_edge_rows leaves no imaging pixels")

    medians = np.zeros_like(batch_mean, dtype=np.float32)
    half_height = height // 2
    radius = median_window_pixels // 2

    for half_start, half_end in (
        (excluded_edge_rows, half_height),
        (half_height, height - excluded_edge_rows),
    ):
        half_data = batch_mean[half_start:half_end]
        rows = half_data.shape[0]
        for row in range(rows):
            row_start = max(0, row - radius)
            row_end = min(rows, row + radius + 1)
            medians[half_start + row] = np.median(
                half_data[row_start:row_end], axis=0
            ).astype(np.float32)

    return medians


def apply_dynamic_and_valid_mask(block,
                                 valid_pixel_mask,
                                 apply_valid_pixel_mask: bool,
                                 apply_dynamic_half_column_mask: bool,
                                 median_window_pixels: int,
                                 threshold_ratio: float,
                                 threshold_offset: float,
                                 excluded_edge_rows: int,
                                 two_sided: bool,
                                 np):
    """Match apply_dynamic_and_valid_pixel_mask_float_kernel in-place."""
    if not apply_valid_pixel_mask and not apply_dynamic_half_column_mask:
        return np.zeros(block.shape[1:], dtype=bool)

    height, width = block.shape[1:]
    should_zero = np.zeros((height, width), dtype=bool)

    if apply_valid_pixel_mask:
        if valid_pixel_mask is None:
            raise ValueError("valid pixel mask requested but not loaded")
        should_zero |= valid_pixel_mask == 0.0

    if apply_dynamic_half_column_mask:
        batch_mean = block.mean(axis=0, dtype=np.float32)
        local_median = half_column_local_median(
            batch_mean, median_window_pixels, excluded_edge_rows, np
        )
        reference = local_median * threshold_ratio
        deviation = batch_mean - reference
        dynamic_zero = (
            np.abs(deviation) > threshold_offset
            if two_sided
            else deviation > threshold_offset
        )

        # The CUDA kernel only applies dynamic masking inside imaging rows.
        dynamic_allowed = np.zeros((height, width), dtype=bool)
        half_height = height // 2
        dynamic_allowed[excluded_edge_rows:half_height, :] = True
        dynamic_allowed[half_height:height - excluded_edge_rows, :] = True
        should_zero |= dynamic_zero & dynamic_allowed & ~should_zero

    if np.any(should_zero):
        block[:, should_zero] = 0.0
    return should_zero


def process_tensor_block(raw_block,
                         dark_frame,
                         valid_pixel_mask,
                         args,
                         np):
    """Apply the same ordered corrections as PyTorchProcessorOp."""
    block = raw_block.astype(np.float32, copy=True)

    if args.subtract_dark_frame:
        block -= dark_frame[None, :, :]

    if args.apply_blr_correction:
        baseline = compute_blr_baseline(
            block,
            args.blr_rows,
            args.blr_zlp_width,
            args.blr_zlp_group_columns,
            args.blr_core_group_columns,
            np,
        )
        subtract_blr_baseline(
            block,
            baseline,
            args.blr_zlp_width,
            args.blr_zlp_group_columns,
            args.blr_core_group_columns,
            np,
        )

    zero_mask = apply_dynamic_and_valid_mask(
        block,
        valid_pixel_mask,
        args.apply_valid_pixel_mask,
        args.apply_dynamic_half_column_mask,
        args.dynamic_mask_median_window_pixels,
        args.dynamic_mask_threshold_ratio,
        args.dynamic_mask_threshold_offset,
        args.dynamic_mask_excluded_edge_rows,
        args.dynamic_mask_two_sided,
        np,
    )

    if args.noop:
        return block.astype(np.float32, copy=False), zero_mask
    return block.sum(axis=0, dtype=np.float32)[None, :, :], zero_mask


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
    parser.add_argument(
        "--frames-per-tensor",
        type=int,
        default=128,
        help="Batch size for operations that Holoscan applies per incoming tensor.",
    )
    parser.add_argument(
        "--noop",
        type=parse_bool,
        default=True,
        help="If true, write all corrected frames. If false, write one summed image per tensor.",
    )
    parser.add_argument("--subtract-dark-frame", type=parse_bool, default=True)
    parser.add_argument("--dark-frame-path", type=Path, required=True)
    parser.add_argument("--dark-frame-dataset", default="/processed")
    parser.add_argument("--apply-valid-pixel-mask", type=parse_bool, default=False)
    parser.add_argument("--valid-pixel-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--apply-blr-correction", type=parse_bool, default=True)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--blr-zlp-width", type=int, default=768)
    parser.add_argument("--blr-zlp-group-columns", type=int, default=4)
    parser.add_argument("--blr-core-group-columns", type=int, default=16)
    parser.add_argument("--apply-dynamic-half-column-mask", type=parse_bool, default=True)
    parser.add_argument("--dynamic-mask-median-window-pixels", type=int, default=31)
    parser.add_argument("--dynamic-mask-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--dynamic-mask-threshold-offset", type=float, default=500.0)
    parser.add_argument("--dynamic-mask-excluded-edge-rows", type=int, default=32)
    parser.add_argument("--dynamic-mask-two-sided", type=parse_bool, default=True)
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="Compression for output frames. Use none to match IGX-friendly uncompressed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.frames is not None and args.frames <= 0:
        raise ValueError("--frames must be positive when provided")
    if args.frames_per_tensor <= 0:
        raise ValueError("--frames-per-tensor must be positive")

    import h5py
    import numpy as np

    compression = None if args.compression == "none" else args.compression

    with h5py.File(args.dark_frame_path, "r") as dark_h5:
        dark_frame = read_single_frame(dark_h5, args.dark_frame_dataset, np)
        valid_pixel_mask = None
        if args.apply_valid_pixel_mask:
            valid_pixel_mask = read_single_frame(
                dark_h5, args.valid_pixel_mask_dataset, np
            )

    with h5py.File(args.input, "r") as in_h5:
        raw = in_h5[normalize_dataset_path(args.input_dataset)]
        if raw.ndim != 3:
            raise ValueError(f"input dataset must have shape [frames, rows, cols], got {raw.shape}")
        total_frames, height, width = raw.shape
        if dark_frame.shape != (height, width):
            raise ValueError(
                f"dark frame shape {dark_frame.shape} does not match raw frames {(height, width)}"
            )
        if valid_pixel_mask is not None and valid_pixel_mask.shape != (height, width):
            raise ValueError(
                f"valid pixel mask shape {valid_pixel_mask.shape} does not match raw frames {(height, width)}"
            )
        if args.start_frame >= total_frames:
            raise ValueError(
                f"start frame {args.start_frame} is outside input with {total_frames} frames"
            )
        end_frame = (
            total_frames
            if args.frames is None
            else min(total_frames, args.start_frame + args.frames)
        )
        selected_frames = end_frame - args.start_frame
        if selected_frames <= 0:
            raise ValueError("no input frames selected")

        if args.noop:
            output_shape = (selected_frames, height, width)
            output_chunks = (1, height, width)
        else:
            output_shape = (
                (selected_frames + args.frames_per_tensor - 1) // args.frames_per_tensor,
                height,
                width,
            )
            output_chunks = (1, height, width)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as out_h5:
            out = create_dataset_with_groups(
                out_h5,
                args.output_dataset,
                shape=output_shape,
                dtype=np.float32,
                chunks=output_chunks,
                compression=compression,
            )
            out.attrs["description"] = (
                "Offline NumPy reproduction of PyTorchProcessorOp correction chain"
            )
            out.attrs["input_file"] = str(args.input)
            out.attrs["input_dataset"] = normalize_dataset_path(args.input_dataset)
            out.attrs["start_frame"] = args.start_frame
            out.attrs["frames"] = selected_frames
            out.attrs["frames_per_tensor"] = args.frames_per_tensor
            out.attrs["noop"] = args.noop
            out.attrs["dark_frame_path"] = str(args.dark_frame_path)
            out.attrs["dark_frame_dataset"] = normalize_dataset_path(args.dark_frame_dataset)

            zeroed_any = np.zeros((height, width), dtype=bool)
            output_index = 0
            for tensor_start in range(args.start_frame, end_frame, args.frames_per_tensor):
                tensor_end = min(end_frame, tensor_start + args.frames_per_tensor)
                raw_block = raw[tensor_start:tensor_end]
                processed, zero_mask = process_tensor_block(
                    raw_block, dark_frame, valid_pixel_mask, args, np
                )
                zeroed_any |= zero_mask

                if args.noop:
                    local_start = tensor_start - args.start_frame
                    out[local_start:local_start + processed.shape[0]] = processed
                    print(
                        f"Processed frames {tensor_start}..{tensor_end - 1} "
                        f"-> output frames {local_start}..{local_start + processed.shape[0] - 1}"
                    )
                else:
                    out[output_index:output_index + 1] = processed
                    print(
                        f"Processed frames {tensor_start}..{tensor_end - 1} "
                        f"-> summed output {output_index}"
                    )
                    output_index += 1

            mask_dataset = create_dataset_with_groups(
                out_h5,
                "/offline_zero_mask",
                data=zeroed_any[None, ...].astype(np.uint8),
                compression=compression,
            )
            mask_dataset.attrs["description"] = (
                "1 where offline dynamic and/or valid-pixel masking zeroed at least one tensor"
            )

            summary = {
                "input": str(args.input),
                "output": str(args.output),
                "selected_frames": selected_frames,
                "frame_shape": [height, width],
                "frames_per_tensor": args.frames_per_tensor,
                "noop": args.noop,
                "subtract_dark_frame": args.subtract_dark_frame,
                "apply_blr_correction": args.apply_blr_correction,
                "apply_valid_pixel_mask": args.apply_valid_pixel_mask,
                "apply_dynamic_half_column_mask": args.apply_dynamic_half_column_mask,
                "zeroed_pixel_locations": int(zeroed_any.sum()),
            }
            out_h5.attrs["offline_processing_summary"] = json.dumps(summary)
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
