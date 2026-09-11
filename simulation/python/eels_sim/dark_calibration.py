from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

PixelRegion = tuple[int, int, int, int]


def detect_storage_lsb(dataset: h5py.Dataset, sample_frames: int = 8) -> int:
    frame_indices = np.linspace(0, dataset.shape[0] - 1, sample_frames, dtype=int)
    samples = dataset[frame_indices, ::16, ::16]
    integers = np.rint(samples[np.isfinite(samples)]).astype(np.int64)
    integers = np.abs(integers[integers != 0])
    if len(integers) == 0:
        return 1
    lsb = int(np.gcd.reduce(integers))
    return max(1, lsb)


def _frame_range_stats(
    dataset: h5py.Dataset,
    start: int,
    stop: int,
    storage_lsb: float,
    batch_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = dataset.shape[1:]
    total = np.zeros((rows, columns), dtype=np.float64)
    total_squared = np.zeros((rows, columns), dtype=np.float64)
    count = stop - start
    if count < 2:
        raise ValueError("At least two frames are required for calibration statistics")
    for batch_start in range(start, stop, batch_frames):
        batch_stop = min(stop, batch_start + batch_frames)
        block = dataset[batch_start:batch_stop]
        total += block.sum(axis=0, dtype=np.float64)
        total_squared += np.einsum("ijk,ijk->jk", block, block, dtype=np.float64)
    mean_raw = total / count
    variance_raw = np.maximum((total_squared - total * total / count) / (count - 1), 0.0)
    return (mean_raw / storage_lsb).astype(np.float32), (
        np.sqrt(variance_raw) / storage_lsb
    ).astype(np.float32)


def _common_modes(
    dataset: h5py.Dataset,
    start: int,
    stop: int,
    storage_lsb: float,
    pedestal: np.ndarray,
    valid: np.ndarray,
    batch_frames: int,
    row_stride: int,
    column_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = stop - start
    rows, columns = dataset.shape[1:]
    global_mode = np.empty(frame_count, dtype=np.float32)
    row_mode = np.empty((frame_count, rows), dtype=np.float32)
    column_mode = np.empty((frame_count, columns), dtype=np.float32)
    global_valid = valid[::row_stride, ::column_stride]
    global_pedestal = pedestal[::row_stride, ::column_stride]
    out_index = 0

    for batch_start in range(start, stop, batch_frames):
        batch_stop = min(stop, batch_start + batch_frames)
        block = dataset[batch_start:batch_stop]
        count = batch_stop - batch_start
        global_residual = (
            block[:, ::row_stride, ::column_stride] / storage_lsb - global_pedestal[None, :, :]
        )
        global_residual[:, ~global_valid] = np.nan
        global_values = np.nanmedian(global_residual, axis=(1, 2))
        global_mode[out_index : out_index + count] = global_values

        row_residual = (
            block[:, :, ::column_stride] / storage_lsb - pedestal[None, :, ::column_stride]
        )
        row_residual[:, ~valid[:, ::column_stride]] = np.nan
        row_values = np.nanmedian(row_residual, axis=2)
        row_mode[out_index : out_index + count] = row_values - global_values[:, None]

        column_residual = block[:, ::row_stride, :] / storage_lsb - pedestal[None, ::row_stride, :]
        column_residual[:, ~valid[::row_stride, :]] = np.nan
        column_values = np.nanmedian(column_residual, axis=1)
        column_mode[out_index : out_index + count] = column_values - global_values[:, None]
        out_index += count
    return global_mode, row_mode, column_mode


def _finite_quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    quantiles = np.quantile(finite, [0.1, 0.5, 0.9, 0.99])
    return {
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p99": float(quantiles[3]),
    }


def build_region_mask(
    shape: tuple[int, int], regions: list[PixelRegion] | tuple[PixelRegion, ...]
) -> np.ndarray:
    """Build a mask from half-open (row_start, row_stop, col_start, col_stop) regions."""
    mask = np.zeros(shape, dtype=bool)
    rows, columns = shape
    for row_start, row_stop, column_start, column_stop in regions:
        if not (0 <= row_start < row_stop <= rows and 0 <= column_start < column_stop <= columns):
            raise ValueError(
                "Known-defect region outside detector bounds: "
                f"({row_start}:{row_stop}, {column_start}:{column_stop}) for {shape}"
            )
        mask[row_start:row_stop, column_start:column_stop] = True
    return mask


def repair_defect_map(values: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
    """Interpolate masked pixels along rows to retain healthy spatial structure."""
    if values.shape != defect_mask.shape:
        raise ValueError("Map and defect mask shapes do not match")
    repaired = values.astype(np.float32, copy=True)
    usable = ~defect_mask & np.isfinite(repaired)
    finite_usable = repaired[usable]
    if finite_usable.size == 0:
        raise ValueError("Cannot repair a map with no healthy finite pixels")
    fallback = float(np.median(finite_usable))
    column_indices = np.arange(repaired.shape[1])
    for row_index in range(repaired.shape[0]):
        row_usable = usable[row_index]
        replace = ~row_usable
        if not np.any(replace):
            continue
        healthy_columns = column_indices[row_usable]
        if healthy_columns.size == 0:
            repaired[row_index, replace] = fallback
        elif healthy_columns.size == 1:
            repaired[row_index, replace] = repaired[row_index, healthy_columns[0]]
        else:
            repaired[row_index, replace] = np.interp(
                column_indices[replace],
                healthy_columns,
                repaired[row_index, row_usable],
            )
    return repaired


def _write_diagnostic_plot(
    output_path: Path,
    pedestal: np.ndarray,
    read_noise: np.ndarray,
    validation_bias: np.ndarray,
    valid: np.ndarray,
    global_mode: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    pedestal_limits = np.quantile(pedestal[valid], [0.01, 0.99])
    noise_limits = np.quantile(read_noise[valid], [0.01, 0.99])
    bias_limit = np.quantile(np.abs(validation_bias[valid]), 0.99)
    images = (
        axes[0, 0].imshow(
            pedestal, aspect="auto", vmin=pedestal_limits[0], vmax=pedestal_limits[1]
        ),
        axes[0, 1].imshow(read_noise, aspect="auto", vmin=noise_limits[0], vmax=noise_limits[1]),
        axes[1, 0].imshow(
            validation_bias,
            aspect="auto",
            cmap="coolwarm",
            vmin=-bias_limit,
            vmax=bias_limit,
        ),
    )
    axes[0, 0].set_title("Pedestal [ADC code]")
    axes[0, 1].set_title("Temporal noise RMS [ADC code]")
    axes[1, 0].set_title("Held-out mean minus pedestal [ADC code]")
    for axis, image in zip((axes[0, 0], axes[0, 1], axes[1, 0]), images):
        axis.set_xlabel("column")
        axis.set_ylabel("row")
        figure.colorbar(image, ax=axis, fraction=0.03)
    axes[1, 1].plot(global_mode, linewidth=1.0)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].set_title("Held-out global common mode")
    axes[1, 1].set_xlabel("validation frame")
    axes[1, 1].set_ylabel("ADC code")
    figure.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(figure)


def calibrate_dark_stack(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "frames",
    calibration_frames: int | None = None,
    validation_frames: int | None = None,
    storage_lsb: int | None = None,
    max_noise_raw: float = 500.0,
    batch_frames: int = 4,
    row_stride: int = 8,
    column_stride: int = 8,
    write_plot: bool = True,
    known_defect_regions: list[PixelRegion] | tuple[PixelRegion, ...] = (),
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with h5py.File(input_path) as source:
        dataset = source[dataset_name]
        total_frames = dataset.shape[0]
        calibration_frames = calibration_frames or total_frames // 2
        validation_frames = validation_frames or total_frames - calibration_frames
        validation_start = calibration_frames
        validation_stop = min(total_frames, validation_start + validation_frames)
        if storage_lsb is None:
            storage_lsb = detect_storage_lsb(dataset)

        pedestal, read_noise = _frame_range_stats(
            dataset, 0, calibration_frames, storage_lsb, batch_frames
        )
        validation_mean, validation_noise = _frame_range_stats(
            dataset, validation_start, validation_stop, storage_lsb, batch_frames
        )
        validation_bias = validation_mean - pedestal
        noise_limit = max_noise_raw / storage_lsb
        valid = (
            np.isfinite(pedestal)
            & np.isfinite(read_noise)
            & (read_noise > 0.0)
            & (read_noise <= noise_limit)
        )
        known_defect = build_region_mask(pedestal.shape, known_defect_regions)
        calibration_defect = known_defect | ~valid
        healthy_pedestal = repair_defect_map(pedestal, calibration_defect)
        healthy_read_noise = repair_defect_map(read_noise, calibration_defect)
        signal_efficiency = np.ones(pedestal.shape, dtype=np.float32)
        signal_efficiency[known_defect] = 0.0
        global_mode, row_mode, column_mode = _common_modes(
            dataset,
            validation_start,
            validation_stop,
            storage_lsb,
            pedestal,
            valid,
            batch_frames,
            row_stride,
            column_stride,
        )

    valid_bias = validation_bias[valid]
    summary: dict[str, object] = {
        "schema": "eels-sim-readout-calibration-v2",
        "input": str(input_path.resolve()),
        "dataset": dataset_name,
        "frame_shape": [int(pedestal.shape[0]), int(pedestal.shape[1])],
        "calibration_frames": int(calibration_frames),
        "validation_frames": int(validation_stop - validation_start),
        "storage_lsb_raw_units_per_adu": int(storage_lsb),
        "effective_adc_bits_from_observed_max": int(
            math.ceil(math.log2(float(np.nanmax(pedestal)) + 1.0))
        ),
        "max_valid_noise_raw": float(max_noise_raw),
        "max_valid_noise_adu": float(noise_limit),
        "invalid_pixels": int(valid.size - np.count_nonzero(valid)),
        "valid_fraction": float(np.mean(valid)),
        "known_defect_pixels": int(np.count_nonzero(known_defect)),
        "known_defect_regions": [list(region) for region in known_defect_regions],
        "pedestal_adu": _finite_quantiles(pedestal[valid]),
        "read_noise_adu": _finite_quantiles(read_noise[valid]),
        "validation_noise_adu": _finite_quantiles(validation_noise[valid]),
        "validation_bias_rms_adu": float(np.sqrt(np.mean(valid_bias**2))),
        "validation_bias_median_adu": float(np.median(valid_bias)),
        "global_common_mode_rms_adu": float(np.std(global_mode, ddof=1)),
        "row_common_mode_rms_adu": float(np.nanstd(row_mode)),
        "column_common_mode_rms_adu": float(np.nanstd(column_mode)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as output:
        output.attrs["schema"] = summary["schema"]
        output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        output.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        maps = output.create_group("maps")
        for name, values, units in (
            ("pedestal_adu", pedestal, "ADC code"),
            ("read_noise_adu", read_noise, "ADC code RMS"),
            ("healthy_pedestal_adu", healthy_pedestal, "ADC code"),
            ("healthy_read_noise_adu", healthy_read_noise, "ADC code RMS"),
            ("signal_efficiency", signal_efficiency, "fraction"),
            ("validation_bias_adu", validation_bias, "ADC code"),
            ("validation_noise_adu", validation_noise, "ADC code RMS"),
        ):
            dataset = maps.create_dataset(name, data=values, compression="gzip", shuffle=True)
            dataset.attrs["units"] = units
        maps.create_dataset("valid_pixel_mask", data=valid.astype(np.uint8), compression="gzip")
        maps.create_dataset(
            "known_defect_mask",
            data=known_defect.astype(np.uint8),
            compression="gzip",
        )
        temporal = output.create_group("temporal")
        temporal.create_dataset("global_common_mode_adu", data=global_mode)
        temporal.create_dataset(
            "row_common_mode_adu", data=row_mode, compression="gzip", shuffle=True
        )
        temporal.create_dataset(
            "column_common_mode_adu", data=column_mode, compression="gzip", shuffle=True
        )

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if write_plot:
        _write_diagnostic_plot(
            output_path, pedestal, read_noise, validation_bias, valid, global_mode
        )
    return summary
