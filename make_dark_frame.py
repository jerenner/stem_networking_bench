#!/usr/bin/env python3
"""Create a single averaged dark frame from an HDF5 frame stack."""

from __future__ import annotations

import argparse
from pathlib import Path


def normalize_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return "/processed"
    return dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"


def create_dataset_with_groups(h5_file, dataset_path: str, data, compression: str | None):
    dataset_path = normalize_dataset_path(dataset_path)
    group_path, dataset_name = dataset_path.rsplit("/", 1)
    group = h5_file if not group_path else h5_file.require_group(group_path)
    return group.create_dataset(dataset_name, data=data, compression=compression)


def average_frames(input_path: Path,
                   dataset_path: str,
                   start_frame: int,
                   num_frames: int | None,
                   chunk_size: int):
    import h5py
    import numpy as np

    dataset_path = normalize_dataset_path(dataset_path)
    with h5py.File(input_path, "r") as h5_file:
        dataset = h5_file[dataset_path]

        if dataset.ndim == 2:
            if start_frame != 0:
                raise ValueError("start_frame must be 0 when averaging a single 2D frame")
            frame = dataset[...].astype(np.float32)
            return frame, 1, dataset.shape

        if dataset.ndim != 3:
            raise ValueError(
                f"expected dataset {dataset_path} to have shape [frames, rows, cols] or [rows, cols], "
                f"got shape {dataset.shape}"
            )

        total_frames = dataset.shape[0]
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"start_frame {start_frame} is outside dataset with {total_frames} frames")

        end_frame = total_frames if num_frames is None else min(total_frames, start_frame + num_frames)
        frames_to_average = end_frame - start_frame
        if frames_to_average <= 0:
            raise ValueError("no frames selected for averaging")

        accumulator = np.zeros(dataset.shape[1:], dtype=np.float64)
        for offset in range(start_frame, end_frame, chunk_size):
            block_end = min(end_frame, offset + chunk_size)
            block = dataset[offset:block_end].astype(np.float64, copy=False)
            accumulator += block.sum(axis=0)

        return (accumulator / frames_to_average).astype(np.float32), frames_to_average, dataset.shape


def write_dark_frame(output_path: Path,
                     output_dataset: str,
                     dark_frame,
                     frames_averaged: int,
                     input_path: Path,
                     input_dataset: str,
                     start_frame: int,
                     input_shape,
                     compression: str | None):
    import h5py

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        # Store as [1, rows, cols] so the file has the same frame-stack convention as writer output.
        dataset = create_dataset_with_groups(
            h5_file,
            output_dataset,
            dark_frame[None, ...],
            compression=compression,
        )
        dataset.attrs["description"] = "Average dark frame"
        dataset.attrs["source_file"] = str(input_path)
        dataset.attrs["source_dataset"] = normalize_dataset_path(input_dataset)
        dataset.attrs["source_shape"] = input_shape
        dataset.attrs["start_frame"] = start_frame
        dataset.attrs["frames_averaged"] = frames_averaged
        dataset.attrs["dtype_note"] = "float32 average stored as [1, rows, cols]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average an HDF5 frame stack into a single float32 dark frame."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 file containing raw dark frames.")
    parser.add_argument("output", type=Path, help="Output HDF5 file for the averaged dark frame.")
    parser.add_argument("--input-dataset", default="/processed", help="Input dataset path.")
    parser.add_argument("--output-dataset", default="/processed", help="Output dataset path.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to average.")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to average.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Frames to read per chunk.")
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="gzip",
        help="Compression for the output dark-frame dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.frames is not None and args.frames <= 0:
        raise ValueError("--frames must be positive when provided")

    dark_frame, frames_averaged, input_shape = average_frames(
        args.input,
        args.input_dataset,
        args.start_frame,
        args.frames,
        args.chunk_size,
    )
    compression = None if args.compression == "none" else args.compression
    write_dark_frame(
        args.output,
        args.output_dataset,
        dark_frame,
        frames_averaged,
        args.input,
        args.input_dataset,
        args.start_frame,
        input_shape,
        compression,
    )

    print(
        f"Wrote {args.output} dataset {normalize_dataset_path(args.output_dataset)} "
        f"from {frames_averaged} frame(s); shape [1, {dark_frame.shape[0]}, {dark_frame.shape[1]}]."
    )


if __name__ == "__main__":
    main()
