#!/usr/bin/env python3
"""Create a blinker-aware averaged dark frame from an HDF5 frame stack."""

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


def compute_dark_statistics(input_path: Path,
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
            mean = dataset[...].astype(np.float32)
            stddev = np.zeros_like(mean, dtype=np.float32)
            return mean, stddev, 1, dataset.shape

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
            if offset == start_frame:
                sumsq = np.zeros(dataset.shape[1:], dtype=np.float64)
            sumsq += np.square(block).sum(axis=0)

        mean = accumulator / frames_to_average
        variance = (sumsq / frames_to_average) - np.square(mean)
        stddev = np.sqrt(np.maximum(variance, 0.0))
        return mean.astype(np.float32), stddev.astype(np.float32), frames_to_average, dataset.shape


def repair_blinker_pixels(dark_frame,
                          dark_stddev,
                          blinker_std_threshold: float,
                          repair_neighbors: int,
                          edge_rows: int):
    import numpy as np

    repaired = dark_frame.copy()
    blinker_mask = dark_stddev > blinker_std_threshold
    height, width = dark_frame.shape
    half_height = height // 2
    repaired_count = 0
    unrepaired_count = 0

    for y, x in np.argwhere(blinker_mask):
        neighbor_count = 0
        total = 0.0
        top_half = y < half_height
        top_edge = False
        bottom_edge = False
        y_up = int(y)
        y_down = int(y)

        while neighbor_count < repair_neighbors and not (top_edge and bottom_edge):
            y_up -= 1
            y_down += 1

            if top_half:
                if y_up < edge_rows:
                    top_edge = True
                if y_down > half_height - 1:
                    bottom_edge = True
            else:
                if y_up < half_height:
                    top_edge = True
                if y_down > height - edge_rows - 1:
                    bottom_edge = True

            if not top_edge and dark_stddev[y_up, x] < blinker_std_threshold:
                neighbor_count += 1
                total += float(dark_frame[y_up, x])

            if not bottom_edge and dark_stddev[y_down, x] < blinker_std_threshold:
                neighbor_count += 1
                total += float(dark_frame[y_down, x])

        if neighbor_count >= repair_neighbors:
            repaired[y, x] = total / neighbor_count
            repaired_count += 1
        else:
            unrepaired_count += 1

    return repaired.astype(np.float32), blinker_mask, repaired_count, unrepaired_count


def write_dark_frame(output_path: Path,
                     output_dataset: str,
                     dark_frame,
                     raw_dark_frame,
                     dark_stddev,
                     blinker_mask,
                     valid_pixel_mask,
                     frames_averaged: int,
                     input_path: Path,
                     input_dataset: str,
                     start_frame: int,
                     input_shape,
                     blinker_std_threshold: float,
                     repair_neighbors: int,
                     edge_rows: int,
                     repaired_count: int,
                     unrepaired_count: int,
                     stddev_dataset: str,
                     blinker_mask_dataset: str,
                     valid_mask_dataset: str,
                     raw_mean_dataset: str,
                     compression: str | None):
    import h5py
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        # Store as [1, rows, cols] so the file has the same frame-stack convention as writer output.
        dataset = create_dataset_with_groups(
            h5_file,
            output_dataset,
            dark_frame[None, ...],
            compression=compression,
        )
        dataset.attrs["description"] = "Blinker-repaired average dark frame"
        dataset.attrs["source_file"] = str(input_path)
        dataset.attrs["source_dataset"] = normalize_dataset_path(input_dataset)
        dataset.attrs["source_shape"] = input_shape
        dataset.attrs["start_frame"] = start_frame
        dataset.attrs["frames_averaged"] = frames_averaged
        dataset.attrs["dtype_note"] = "float32 average stored as [1, rows, cols]"
        dataset.attrs["blinker_std_threshold"] = blinker_std_threshold
        dataset.attrs["repair_neighbors"] = repair_neighbors
        dataset.attrs["edge_rows"] = edge_rows
        dataset.attrs["blinker_pixels"] = int(blinker_mask.sum())
        dataset.attrs["repaired_blinker_pixels"] = repaired_count
        dataset.attrs["unrepaired_blinker_pixels"] = unrepaired_count

        raw_dataset = create_dataset_with_groups(
            h5_file,
            raw_mean_dataset,
            raw_dark_frame[None, ...],
            compression=compression,
        )
        raw_dataset.attrs["description"] = "Raw average dark frame before blinker repair"

        stddev = create_dataset_with_groups(
            h5_file,
            stddev_dataset,
            dark_stddev[None, ...].astype(np.float32),
            compression=compression,
        )
        stddev.attrs["description"] = "Per-pixel temporal standard deviation of dark stack"
        stddev.attrs["blinker_std_threshold"] = blinker_std_threshold

        blinker = create_dataset_with_groups(
            h5_file,
            blinker_mask_dataset,
            blinker_mask[None, ...].astype(np.uint8),
            compression=compression,
        )
        blinker.attrs["description"] = "1 where temporal dark stddev exceeds threshold"

        valid = create_dataset_with_groups(
            h5_file,
            valid_mask_dataset,
            valid_pixel_mask[None, ...].astype(np.uint8),
            compression=compression,
        )
        valid.attrs["description"] = "1 for pixels considered valid during runtime masking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average an HDF5 frame stack into a blinker-aware float32 dark frame."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 file containing raw dark frames.")
    parser.add_argument("output", type=Path, help="Output HDF5 file for the averaged dark frame.")
    parser.add_argument("--input-dataset", default="/processed", help="Input dataset path.")
    parser.add_argument("--output-dataset", default="/processed", help="Output dataset path.")
    parser.add_argument("--raw-mean-dataset", default="/raw_dark_mean", help="Raw mean dataset path.")
    parser.add_argument("--stddev-dataset", default="/dark_stddev", help="Dark temporal stddev dataset path.")
    parser.add_argument("--blinker-mask-dataset", default="/blinker_mask", help="Blinker mask dataset path.")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask", help="Valid pixel mask dataset path.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to average.")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to average.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Frames to read per chunk.")
    parser.add_argument(
        "--blinker-std-threshold",
        type=float,
        default=500.0,
        help="Pixels with temporal dark stddev above this value are treated as blinkers.",
    )
    parser.add_argument(
        "--repair-neighbors",
        type=int,
        default=10,
        help="Same-column good neighbors used to repair each blinker pixel.",
    )
    parser.add_argument(
        "--edge-rows",
        type=int,
        default=32,
        help="Rows at the top/bottom edge excluded from neighbor repair search.",
    )
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
    if args.repair_neighbors <= 0:
        raise ValueError("--repair-neighbors must be positive")
    if args.edge_rows < 0:
        raise ValueError("--edge-rows must be non-negative")

    raw_dark_frame, dark_stddev, frames_averaged, input_shape = compute_dark_statistics(
        args.input,
        args.input_dataset,
        args.start_frame,
        args.frames,
        args.chunk_size,
    )
    dark_frame, blinker_mask, repaired_count, unrepaired_count = repair_blinker_pixels(
        raw_dark_frame,
        dark_stddev,
        args.blinker_std_threshold,
        args.repair_neighbors,
        args.edge_rows,
    )
    valid_pixel_mask = ~blinker_mask

    compression = None if args.compression == "none" else args.compression
    write_dark_frame(
        args.output,
        args.output_dataset,
        dark_frame,
        raw_dark_frame,
        dark_stddev,
        blinker_mask,
        valid_pixel_mask,
        frames_averaged,
        args.input,
        args.input_dataset,
        args.start_frame,
        input_shape,
        args.blinker_std_threshold,
        args.repair_neighbors,
        args.edge_rows,
        repaired_count,
        unrepaired_count,
        args.stddev_dataset,
        args.blinker_mask_dataset,
        args.valid_mask_dataset,
        args.raw_mean_dataset,
        compression,
    )

    blinker_count = int(blinker_mask.sum())
    print(
        f"Wrote {args.output} dataset {normalize_dataset_path(args.output_dataset)} "
        f"from {frames_averaged} frame(s); shape [1, {dark_frame.shape[0]}, {dark_frame.shape[1]}]."
    )
    print(
        f"Blinker threshold {args.blinker_std_threshold:g}: {blinker_count} pixel(s) flagged, "
        f"{repaired_count} repaired in the dark frame, {unrepaired_count} left as raw mean. "
        "All flagged pixels are marked invalid in /valid_pixel_mask."
    )


if __name__ == "__main__":
    main()
