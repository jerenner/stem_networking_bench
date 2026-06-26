"""DM4 loading and frame-stack normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


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
    return data, {"reader": "ncempy", "metadata": {key: value for key, value in result.items() if key != "data"}}


def load_dm4(path: Path, reader: str = "auto"):
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


def choose_spatial_axes(shape: tuple[int, ...],
                        height: int | None,
                        width: int | None) -> tuple[int, int]:
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

    if len(shape) >= 2 and shape[-2:] in ((1024, 3840), (960, 3840)):
        return len(shape) - 2, len(shape) - 1
    if len(shape) >= 2 and shape[:2] in ((1024, 3840), (960, 3840)):
        return 0, 1

    sorted_axes = sorted(range(len(shape)), key=lambda axis: shape[axis], reverse=True)
    spatial = sorted(sorted_axes[:2])
    return spatial[0], spatial[1]


def normalize_to_frame_stack(data,
                             frames_axis: int | None = None,
                             height: int | None = None,
                             width: int | None = None):
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

