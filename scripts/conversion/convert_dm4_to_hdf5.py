#!/usr/bin/env python3
"""Convert Gatan DM4 frame stacks to the HDF5 layout used by stem_networking_bench."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def normalize_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return "/frames"
    return dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"


def load_dm4_with_rsciio(path: Path):
    from rsciio.digitalmicrograph import file_reader

    try:
        objects = file_reader(str(path), lazy=True)
    except TypeError:
        objects = file_reader(str(path))

    for obj in objects:
        data = obj.get("data")
        if data is not None and getattr(data, "ndim", 0) >= 2:
            return data, {"reader": "rsciio", "metadata": obj.get("metadata", {})}
    raise ValueError(f"no image stack found in {path}")


def load_dm4_with_hyperspy(path: Path):
    import hyperspy.api as hs

    signal = hs.load(str(path), lazy=True)
    if isinstance(signal, list):
        signal = next((item for item in signal if getattr(item.data, "ndim", 0) >= 2), signal[0])
    return signal.data, {"reader": "hyperspy", "metadata": getattr(signal, "metadata", {})}


def load_dm4_with_ncempy(path: Path):
    from ncempy.io import dm

    result = dm.dmReader(str(path))
    data = result.get("data")
    if data is None:
        raise ValueError(f"ncempy did not return a data array for {path}")
    return data, {"reader": "ncempy", "metadata": {k: v for k, v in result.items() if k != "data"}}


def load_dm4(path: Path, reader: str):
    errors: list[str] = []
    readers = {
        "rsciio": load_dm4_with_rsciio,
        "hyperspy": load_dm4_with_hyperspy,
        "ncempy": load_dm4_with_ncempy,
    }
    selected_readers: Iterable[str] = readers.keys() if reader == "auto" else (reader,)

    for reader_name in selected_readers:
        try:
            return readers[reader_name](path)
        except Exception as exc:  # noqa: BLE001 - report all reader failures together.
            errors.append(f"{reader_name}: {type(exc).__name__}: {exc}")

    error_text = "\n".join(f"  - {line}" for line in errors)
    raise RuntimeError(
        "Could not read DM4 file. Install one supported reader, e.g. `pip install rosettasciio`, "
        "`pip install hyperspy`, or `pip install ncempy`.\n"
        f"Reader attempts for {path}:\n{error_text}"
    )


def choose_spatial_axes(shape: tuple[int, ...], height: int | None, width: int | None) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"expected at least 2 dimensions, got shape {shape}")

    if height is not None and width is not None:
        matches = [
            (y, x)
            for y, dim_y in enumerate(shape)
            for x, dim_x in enumerate(shape)
            if y != x and dim_y == height and dim_x == width
        ]
        if matches:
            return matches[0]
        raise ValueError(f"could not find spatial axes matching height={height}, width={width} in {shape}")

    if len(shape) >= 2 and shape[-2:] == (1024, 3840):
        return len(shape) - 2, len(shape) - 1
    if len(shape) >= 2 and shape[:2] == (1024, 3840):
        return 0, 1

    # Fall back to treating the two largest dimensions as image axes.
    sorted_axes = sorted(range(len(shape)), key=lambda axis: shape[axis], reverse=True)
    spatial = sorted(sorted_axes[:2])
    return spatial[0], spatial[1]


def normalize_to_frame_stack(data, frames_axis: int | None, height: int | None, width: int | None):
    import numpy as np

    squeezed = np.squeeze(data)
    shape = tuple(int(dim) for dim in squeezed.shape)
    if len(shape) == 2:
        return squeezed.reshape((1,) + shape)

    if frames_axis is not None:
        frames_axis = frames_axis if frames_axis >= 0 else len(shape) + frames_axis
        moved = np.moveaxis(squeezed, frames_axis, 0)
        if moved.ndim != 3:
            moved = moved.reshape((-1,) + tuple(moved.shape[-2:]))
        return moved

    row_axis, col_axis = choose_spatial_axes(shape, height, width)
    remaining_axes = [axis for axis in range(len(shape)) if axis not in (row_axis, col_axis)]
    transposed = np.transpose(squeezed, remaining_axes + [row_axis, col_axis])
    return transposed.reshape((-1, transposed.shape[-2], transposed.shape[-1]))


def as_uint16(block):
    import numpy as np

    if block.dtype == np.uint16:
        return block
    if np.issubdtype(block.dtype, np.integer):
        return np.clip(block, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return np.clip(block, 0, np.iinfo(np.uint16).max).round().astype(np.uint16)


def as_output_dtype(block, output_dtype: str):
    import numpy as np

    if output_dtype == "uint16":
        return as_uint16(block)
    if output_dtype == "float32":
        return block if block.dtype == np.float32 else block.astype(np.float32)
    raise ValueError(f"unsupported output dtype {output_dtype!r}")


def write_hdf5(inputs: list[Path],
               output: Path,
               dataset_path: str,
               reader: str,
               frames_axis: int | None,
               height: int | None,
               width: int | None,
               start_frame: int,
               max_frames_per_file: int | None,
               chunk_size: int,
               compression: str | None,
               output_dtype: str,
               inspect_only: bool) -> None:
    import h5py
    import numpy as np

    stacks = []
    total_frames = 0
    frame_shape = None

    for input_path in inputs:
        data, info = load_dm4(input_path, reader)
        stack = normalize_to_frame_stack(data, frames_axis, height, width)
        if start_frame:
            stack = stack[start_frame:]
        if max_frames_per_file is not None:
            stack = stack[:max_frames_per_file]

        stack_shape = tuple(int(dim) for dim in stack.shape)
        print(f"{input_path}: reader={info['reader']} raw_shape={tuple(data.shape)} stack_shape={stack_shape} dtype={stack.dtype}")

        if len(stack_shape) != 3:
            raise ValueError(f"normalized stack must be [frames, rows, cols], got {stack_shape}")
        if stack_shape[0] == 0:
            raise ValueError(f"no frames selected from {input_path}")
        if frame_shape is None:
            frame_shape = stack_shape[1:]
        elif frame_shape != stack_shape[1:]:
            raise ValueError(f"frame shape mismatch: expected {frame_shape}, got {stack_shape[1:]} for {input_path}")

        stacks.append((input_path, stack))
        total_frames += stack_shape[0]

    if inspect_only:
        print(
            f"Would write {total_frames} frame(s) with frame shape {frame_shape} "
            f"and dtype {output_dtype} to {output}:{dataset_path}"
        )
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = normalize_dataset_path(dataset_path)
    compression_value = None if compression == "none" else compression
    h5_dtype = np.uint16 if output_dtype == "uint16" else np.float32

    with h5py.File(output, "w") as h5_file:
        group_path, dataset_name = dataset_path.rsplit("/", 1)
        group = h5_file if not group_path else h5_file.require_group(group_path)
        dataset = group.create_dataset(
            dataset_name,
            shape=(total_frames, frame_shape[0], frame_shape[1]),
            dtype=h5_dtype,
            chunks=(min(chunk_size, total_frames), frame_shape[0], frame_shape[1]),
            compression=compression_value,
        )

        write_offset = 0
        for input_path, stack in stacks:
            for start in range(0, stack.shape[0], chunk_size):
                end = min(stack.shape[0], start + chunk_size)
                dataset[write_offset:write_offset + (end - start)] = as_output_dtype(
                    np.asarray(stack[start:end]),
                    output_dtype,
                )
                write_offset += end - start

        dataset.attrs["description"] = "DM4 frame stack converted for stem_networking_bench HDF5 replay"
        dataset.attrs["source_files"] = np.array(
            [str(path) for path in inputs],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset.attrs["start_frame"] = start_frame
        dataset.attrs["max_frames_per_file"] = -1 if max_frames_per_file is None else max_frames_per_file
        dataset.attrs["dataset_layout"] = "[frames, rows, cols]"
        dataset.attrs["output_dtype"] = output_dtype

    print(
        f"Wrote {output} dataset {dataset_path} "
        f"with shape {(total_frames, frame_shape[0], frame_shape[1])} and dtype {output_dtype}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DM4 frame stacks to HDF5 [frames, rows, cols].")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input DM4 file(s).")
    parser.add_argument("--output", required=True, type=Path, help="Output HDF5 file.")
    parser.add_argument("--dataset", default="/frames", help="Output dataset path.")
    parser.add_argument("--reader", choices=("auto", "rsciio", "hyperspy", "ncempy"), default="auto")
    parser.add_argument("--frames-axis", type=int, default=None, help="Override frame axis before normalization.")
    parser.add_argument("--height", type=int, default=None, help="Expected image height for axis inference.")
    parser.add_argument("--width", type=int, default=None, help="Expected image width for axis inference.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to include from each file.")
    parser.add_argument("--max-frames-per-file", type=int, default=None, help="Maximum frames to include per file.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Frames per HDF5 write chunk.")
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"), default="gzip")
    parser.add_argument(
        "--output-dtype",
        choices=("uint16", "float32"),
        default="uint16",
        help="Output HDF5 dataset dtype. Use float32 to preserve DM4 data outside uint16 range.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Print detected shapes without writing HDF5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_frames_per_file is not None and args.max_frames_per_file <= 0:
        raise ValueError("--max-frames-per-file must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    write_hdf5(
        args.inputs,
        args.output,
        args.dataset,
        args.reader,
        args.frames_axis,
        args.height,
        args.width,
        args.start_frame,
        args.max_frames_per_file,
        args.chunk_size,
        args.compression,
        args.output_dtype,
        args.inspect_only,
    )


if __name__ == "__main__":
    main()
