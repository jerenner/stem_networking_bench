#!/usr/bin/env python3
"""Generate compact, scientifically grounded assets for the DAQIRI NiO demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("demo_config.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated asset directory.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("raw_file", "dark_file", "output_directory", "processor")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "missing demo configuration keys: {}".format(", ".join(missing))
        )
    return config


def resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def configure_matplotlib() -> None:
    cache = Path(tempfile.gettempdir()) / "stem-daqiri-demo-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))


def _finite_percentiles(data, low: float, high: float, np) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _symmetric_limit(data, percentile: float, np) -> float:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(np.abs(finite), percentile))
    return value if value > 0.0 else 1.0


def _save_detector_image(data, path: Path, title: str, mode: str, plt, np) -> None:
    from matplotlib.colors import TwoSlopeNorm

    figure, axis = plt.subplots(figsize=(12.8, 4.1), constrained_layout=True)
    figure.patch.set_facecolor("#07131b")
    axis.set_facecolor("#07131b")
    if mode == "raw":
        vmin, vmax = _finite_percentiles(data, 0.2, 99.85, np)
        image = axis.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    elif mode == "sum":
        positive_scale = max(float(np.percentile(np.abs(data), 85.0)), 1.0)
        transformed = np.arcsinh(data / positive_scale)
        limit = _symmetric_limit(transformed, 99.7, np)
        image = axis.imshow(
            transformed,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
            aspect="auto",
        )
    else:
        limit = _symmetric_limit(data, 99.7, np)
        image = axis.imshow(
            data,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
            aspect="auto",
        )
    axis.set_title(title, color="#f5f0e8", fontsize=18, loc="left", pad=10)
    axis.set_xlabel("Detector column", color="#b9c6cc")
    axis.set_ylabel("Detector row", color="#b9c6cc")
    axis.tick_params(colors="#8fa3ad", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#35505d")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.ax.tick_params(colors="#8fa3ad", labelsize=7)
    colorbar.outline.set_edgecolor("#35505d")
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def _save_mask_image(static_mask, dynamic_mask, path: Path, plt, np) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    classes = np.zeros(static_mask.shape, dtype=np.uint8)
    classes[dynamic_mask] = 1
    classes[static_mask] = 2
    cmap = ListedColormap(["#07131b", "#ffb000", "#35d0ba"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    figure, axis = plt.subplots(figsize=(12.8, 4.1), constrained_layout=True)
    figure.patch.set_facecolor("#07131b")
    axis.imshow(classes, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axis.set_title(
        "Pixels removed from every frame in this bucket",
        color="#f5f0e8",
        fontsize=18,
        loc="left",
        pad=10,
    )
    axis.set_xlabel("Detector column", color="#b9c6cc")
    axis.set_ylabel("Detector row", color="#b9c6cc")
    axis.tick_params(colors="#8fa3ad", labelsize=8)
    axis.text(
        0.01,
        -0.19,
        "orange: dynamic two-sided outlier    teal: static invalid pixel",
        transform=axis.transAxes,
        color="#b9c6cc",
        fontsize=10,
    )
    for spine in axis.spines.values():
        spine.set_color("#35505d")
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def _style_spectrum_axis(axis, boundary: int) -> None:
    axis.set_facecolor("#07131b")
    axis.axvspan(0, boundary, color="#35d0ba", alpha=0.08)
    axis.axvline(boundary, color="#f5f0e8", linewidth=1.0, alpha=0.55)
    axis.text(0.012, 0.92, "folded ZLP", transform=axis.transAxes, color="#35d0ba")
    axis.text(0.22, 0.92, "CoreLoss", transform=axis.transAxes, color="#ffb000")
    axis.set_xlabel("Collapsed detector channel", color="#b9c6cc")
    axis.set_ylabel("Mean corrected response", color="#b9c6cc")
    axis.tick_params(colors="#8fa3ad")
    axis.grid(color="#35505d", alpha=0.25, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color("#35505d")


def _save_spectrum(spectrum, path: Path, title: str, log_scale: bool, plt, np) -> None:
    figure, axis = plt.subplots(figsize=(12.8, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("#07131b")
    x = np.arange(spectrum.size)
    if log_scale:
        positive = spectrum[np.isfinite(spectrum) & (spectrum > 0)]
        floor = max(
            float(np.percentile(positive, 0.5)) if positive.size else 1e-3, 1e-6
        )
        shown = np.maximum(spectrum, floor)
        axis.semilogy(x, shown, color="#ffb000", linewidth=1.5)
    else:
        axis.plot(x, spectrum, color="#ffb000", linewidth=1.5)
        finite = spectrum[np.isfinite(spectrum)]
        if finite.size:
            low, high = np.percentile(finite, [0.5, 99.7])
            axis.set_ylim(min(0.0, float(low)), float(high) * 1.08)
    _style_spectrum_axis(axis, 192)
    axis.set_title(title, color="#f5f0e8", fontsize=19, loc="left", pad=10)
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def _write_manifest(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def generate_assets(config_path: Path, force: bool = False) -> Path:
    """Generate all demo assets and return the output directory."""

    configure_matplotlib()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from stem_analysis import ProcessorConfig
    from stem_analysis.processing import (
        apply_dynamic_and_valid_mask,
        compute_blr_baseline,
        subtract_blr_baseline,
    )
    from stem_analysis.spectra import collapsed_stitch_profile
    from stem_analysis.stitch import apply_stitch_calibration

    config_path = config_path.resolve()
    config = load_config(config_path)
    raw_path = resolve_repo_path(config["raw_file"])
    dark_path = resolve_repo_path(config["dark_file"])
    output_dir = resolve_repo_path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "demo_metadata.json"
    if manifest_path.exists() and not force:
        print("Assets already exist at {} (use --force to rebuild)".format(output_dir))
        return output_dir

    start = int(config.get("start_frame", 0))
    bucket_size = int(config.get("bucket_size", 128))
    representative_index = int(config.get("representative_frame_index", 64))
    chunk_size = int(config.get("chunk_size", 8))
    if start < 0 or bucket_size <= 0 or chunk_size <= 0:
        raise ValueError(
            "start_frame must be non-negative and bucket/chunk sizes positive"
        )
    if representative_index < 0 or representative_index >= bucket_size:
        raise ValueError("representative_frame_index must fall inside the bucket")

    processor_values = dict(config["processor"])
    processor_values.pop("noop", None)
    processor_config = ProcessorConfig(noop=True, **processor_values)
    detector = config.get("detector", {})
    edge_rows = int(detector.get("edge_rows", 32))
    zlp_width = int(detector.get("zlp_width", processor_config.blr_zlp_width))
    zlp_period = int(detector.get("zlp_period", 192))
    apply_stitch = bool(detector.get("apply_stitch", False))
    stitch_gain = detector.get("stitch_gain")

    with h5py.File(dark_path, "r") as dark_h5:
        dark = dark_h5[config.get("dark_dataset", "/processed")][...]
        if dark.ndim == 3 and dark.shape[0] == 1:
            dark = dark[0]
        dark = dark.astype(np.float32, copy=False)
        valid = dark_h5[config.get("valid_mask_dataset", "/valid_pixel_mask")][...]
        if valid.ndim == 3 and valid.shape[0] == 1:
            valid = valid[0]
        valid = valid.astype(np.float32, copy=False)

    with h5py.File(raw_path, "r") as raw_h5:
        dataset = raw_h5[config.get("raw_dataset", "/frames")]
        if dataset.ndim != 3:
            raise ValueError("raw dataset must have shape [frames, rows, columns]")
        stop = start + bucket_size
        if stop > dataset.shape[0]:
            raise ValueError(
                "selected bucket [{}:{}) exceeds {} input frames".format(
                    start, stop, dataset.shape[0]
                )
            )
        height, width = map(int, dataset.shape[1:])
        if dark.shape != (height, width) or valid.shape != (height, width):
            raise ValueError(
                "dark/mask shape does not match raw frame: raw={}, dark={}, mask={}".format(
                    (height, width), dark.shape, valid.shape
                )
            )
        if height % 2 or edge_rows < 0 or edge_rows >= height // 2:
            raise ValueError("invalid detector edge-row geometry")
        if zlp_width > width or zlp_width % zlp_period:
            raise ValueError("invalid ZLP width/period for detector width")

        raw_sum = np.zeros((height, width), dtype=np.float64)
        dark_sum = np.zeros((height, width), dtype=np.float64)
        blr_sum = np.zeros((height, width), dtype=np.float64)
        raw_representative = None
        dark_representative = None
        blr_representative = None
        representative_absolute = start + representative_index

        for chunk_start in range(start, stop, chunk_size):
            chunk_stop = min(stop, chunk_start + chunk_size)
            raw_chunk = dataset[chunk_start:chunk_stop].astype(np.float32, copy=False)
            dark_chunk = raw_chunk.astype(np.float32, copy=True)
            if processor_config.subtract_dark_frame:
                dark_chunk -= dark[None, :, :]
            blr_chunk = dark_chunk.astype(np.float32, copy=True)
            if processor_config.apply_blr_correction:
                baseline = compute_blr_baseline(
                    blr_chunk,
                    processor_config.blr_rows,
                    processor_config.blr_zlp_width,
                    processor_config.blr_zlp_group_columns,
                    processor_config.blr_core_group_columns,
                    np,
                )
                subtract_blr_baseline(
                    blr_chunk,
                    baseline,
                    processor_config.blr_zlp_width,
                    processor_config.blr_zlp_group_columns,
                    processor_config.blr_core_group_columns,
                    np,
                )

            raw_sum += raw_chunk.sum(axis=0, dtype=np.float64)
            dark_sum += dark_chunk.sum(axis=0, dtype=np.float64)
            blr_sum += blr_chunk.sum(axis=0, dtype=np.float64)
            if chunk_start <= representative_absolute < chunk_stop:
                offset = representative_absolute - chunk_start
                raw_representative = raw_chunk[offset].copy()
                dark_representative = dark_chunk[offset].copy()
                blr_representative = blr_chunk[offset].copy()

    if (
        raw_representative is None
        or dark_representative is None
        or blr_representative is None
    ):
        raise RuntimeError("representative frame was not captured")

    mean_blr = (blr_sum / bucket_size).astype(np.float32)
    mask_probe = mean_blr[None, :, :].copy()
    zero_mask = apply_dynamic_and_valid_mask(
        mask_probe,
        valid if processor_config.apply_valid_pixel_mask else None,
        processor_config,
        np,
    )
    static_mask = (
        (valid == 0.0)
        if processor_config.apply_valid_pixel_mask
        else np.zeros_like(zero_mask)
    )
    dynamic_mask = zero_mask & ~static_mask
    corrected_representative = blr_representative.copy()
    corrected_representative[zero_mask] = 0.0
    corrected_sum = blr_sum.astype(np.float32)
    corrected_sum[zero_mask] = 0.0

    imaging_rows = slice(edge_rows, height - edge_rows)
    detector_sums = corrected_sum[imaging_rows].sum(axis=0, dtype=np.float64)
    detector_counts = (~zero_mask[imaging_rows]).sum(axis=0).astype(np.float64)
    detector_counts *= bucket_size
    collapsed = collapsed_stitch_profile(
        detector_sums,
        detector_counts,
        np,
        zlp_width=zlp_width,
        zlp_period=zlp_period,
    )
    stitch_metadata = {
        "applied": False,
        "reason": "disabled; a valid gain should be calibrated from no-BLR data",
    }
    spectrum = collapsed
    if apply_stitch:
        if stitch_gain is None:
            raise ValueError(
                "detector.apply_stitch requires detector.stitch_gain from an independent no-BLR calibration"
            )
        spectrum, stitch_details = apply_stitch_calibration(
            collapsed, float(stitch_gain), np
        )
        stitch_metadata = {
            "applied": True,
            "gain": float(stitch_gain),
            **stitch_details,
        }

    _save_detector_image(
        raw_representative,
        output_dir / "raw_frame.png",
        "Recorded NiO frame - raw",
        "raw",
        plt,
        np,
    )
    _save_detector_image(
        dark_representative,
        output_dir / "dark_subtracted_frame.png",
        "After dark subtraction",
        "corrected",
        plt,
        np,
    )
    _save_detector_image(
        blr_representative,
        output_dir / "blr_corrected_frame.png",
        "After grouped BLR correction",
        "corrected",
        plt,
        np,
    )
    _save_mask_image(
        static_mask, dynamic_mask, output_dir / "mask_overlay.png", plt, np
    )
    _save_detector_image(
        corrected_representative,
        output_dir / "corrected_frame.png",
        "After static and dynamic masking",
        "corrected",
        plt,
        np,
    )
    _save_detector_image(
        corrected_sum,
        output_dir / "bucket_sum.png",
        "Sum of 128 corrected frames (symmetric compressed scale)",
        "sum",
        plt,
        np,
    )
    _save_spectrum(
        spectrum,
        output_dir / "spectrum_linear.png",
        "NiO bucket spectrum - linear scale",
        False,
        plt,
        np,
    )
    _save_spectrum(
        spectrum,
        output_dir / "spectrum_log.png",
        "NiO bucket spectrum - logarithmic scale",
        True,
        plt,
        np,
    )

    np.savez_compressed(
        output_dir / "spectrum_data.npz",
        collapsed_unstitched=collapsed.astype(np.float32),
        spectrum=spectrum.astype(np.float32),
        masked_fraction_by_column=zero_mask[imaging_rows]
        .mean(axis=0)
        .astype(np.float32),
    )
    metadata = {
        "schema": "stem.daqiri.demo.v1",
        "title": config.get("title", "STEM DAQIRI demo"),
        "source": {
            "raw_file": str(config["raw_file"]),
            "raw_dataset": config.get("raw_dataset", "/frames"),
            "dark_file": str(config["dark_file"]),
            "start_frame": start,
            "bucket_size": bucket_size,
            "representative_frame": representative_absolute,
            "label": "recorded NiO data replayed through the offline processor mirror",
        },
        "detector_shape": [height, width],
        "processor": asdict(processor_config),
        "mask": {
            "static_pixels": int(static_mask.sum()),
            "dynamic_pixels": int(dynamic_mask.sum()),
            "total_pixels": int(zero_mask.sum()),
            "fraction": float(zero_mask.mean()),
        },
        "spectrum": {
            "channels": int(spectrum.size),
            "zlp_channels": zlp_period,
            "coreloss_channels": int(spectrum.size - zlp_period),
            "stitch": stitch_metadata,
        },
        "performance": config.get("performance", {}),
        "assets": [
            "raw_frame.png",
            "dark_subtracted_frame.png",
            "blr_corrected_frame.png",
            "mask_overlay.png",
            "corrected_frame.png",
            "bucket_sum.png",
            "spectrum_linear.png",
            "spectrum_log.png",
            "spectrum_data.npz",
        ],
    }
    _write_manifest(manifest_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return output_dir


def main() -> None:
    args = parse_args()
    generate_assets(args.config, force=args.force)


if __name__ == "__main__":
    main()
