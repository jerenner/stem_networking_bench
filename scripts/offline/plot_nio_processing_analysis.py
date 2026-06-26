#!/usr/bin/env python3
"""Compare raw detector frames with dark/BLR-corrected and masked output."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "stem-networking-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def parse_frame_indices(text: str, frame_count: int) -> list[int]:
    indices = [int(value.strip()) for value in text.split(",") if value.strip()]
    invalid = [index for index in indices if index < 0 or index >= frame_count]
    if invalid:
        raise ValueError(f"frame indices outside [0, {frame_count - 1}]: {invalid}")
    return indices


def robust_limits(data, np, low: float, high: float) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def symmetric_limit(data, np, percentile: float = 99.5) -> float:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    return limit if limit > 0 else 1.0


def detector_regions(height: int, edge_rows: int) -> tuple[slice, slice]:
    half_height = height // 2
    if height % 2 != 0:
        raise ValueError(f"detector height must be even, got {height}")
    if edge_rows < 0 or edge_rows >= half_height:
        raise ValueError(f"edge_rows must be in [0, {half_height - 1}], got {edge_rows}")
    return slice(edge_rows, half_height), slice(half_height, height - edge_rows)


def subtract_imagej_blr(block,
                        np,
                        blr_rows: int,
                        zlp_width: int,
                        zlp_group_columns: int,
                        core_group_columns: int):
    """Subtract the BLR_v1.ijm edge-row/grouped-column baseline from [N,H,W]."""
    frames, height, width = block.shape
    if blr_rows <= 0 or 2 * blr_rows > height:
        raise ValueError(f"invalid blr_rows={blr_rows} for height={height}")
    if zlp_width < 0 or zlp_width > width:
        raise ValueError(f"invalid zlp_width={zlp_width} for width={width}")
    if zlp_width % zlp_group_columns != 0:
        raise ValueError("ZLP width must be divisible by zlp_group_columns")
    if (width - zlp_width) % core_group_columns != 0:
        raise ValueError("CoreLoss width must be divisible by core_group_columns")

    corrected = block.astype(np.float32, copy=True)
    half_height = height // 2
    baseline_halves = []

    for edge_block in (corrected[:, :blr_rows], corrected[:, height - blr_rows:]):
        parts = []
        if zlp_width:
            zlp_bins = zlp_width // zlp_group_columns
            zlp = edge_block[:, :, :zlp_width].reshape(
                frames, blr_rows, zlp_bins, zlp_group_columns
            )
            parts.append(np.repeat(zlp.mean(axis=(1, 3)), zlp_group_columns, axis=1))
        if zlp_width < width:
            core_width = width - zlp_width
            core_bins = core_width // core_group_columns
            core = edge_block[:, :, zlp_width:].reshape(
                frames, blr_rows, core_bins, core_group_columns
            )
            parts.append(np.repeat(core.mean(axis=(1, 3)), core_group_columns, axis=1))
        baseline_halves.append(np.concatenate(parts, axis=1))

    corrected[:, :half_height] -= baseline_halves[0][:, None, :]
    corrected[:, half_height:] -= baseline_halves[1][:, None, :]
    return corrected


def save_frame_comparison(raw_frame,
                          processed_frame,
                          raw_index: int,
                          processed_index: int,
                          output_path: Path,
                          plt,
                          TwoSlopeNorm,
                          np) -> None:
    raw_vmin, raw_vmax = robust_limits(raw_frame, np, 0.5, 99.5)
    processed_limit = symmetric_limit(processed_frame, np)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    raw_image = axes[0].imshow(
        raw_frame, cmap="magma", vmin=raw_vmin, vmax=raw_vmax, aspect="auto"
    )
    axes[0].set_title(f"Raw frame {raw_index}")
    processed_image = axes[1].imshow(
        processed_frame,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-processed_limit, vmax=processed_limit),
        aspect="auto",
    )
    axes[1].set_title(f"Processed frame {processed_index}")

    for axis in axes:
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(processed_image, ax=axes[1], fraction=0.046, pad=0.02)
    fig.suptitle("Raw versus dark-subtracted, BLR-corrected, masked data")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_mean_profiles(mean_image,
                       top_rows: slice,
                       bottom_rows: slice,
                       edge_rows: int,
                       zlp_width: int,
                       title_prefix: str,
                       value_label: str,
                       output_path: Path,
                       plt,
                       np) -> None:
    height = mean_image.shape[0]
    row_profile = mean_image.mean(axis=1)
    top_column_profile = mean_image[top_rows].mean(axis=0)
    bottom_column_profile = mean_image[bottom_rows].mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].plot(np.arange(height), row_profile, linewidth=1.0)
    axes[0].axvspan(0, edge_rows - 1, color="tab:gray", alpha=0.2)
    axes[0].axvspan(
        height - edge_rows, height - 1, color="tab:gray", alpha=0.2
    )
    axes[0].axvline(height // 2, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{title_prefix} row mean")
    axes[0].set_xlabel("Row")
    axes[0].set_ylabel(f"Mean {value_label}")
    axes[0].grid(alpha=0.2)

    axes[1].plot(top_column_profile, label="top imaging half", linewidth=0.9)
    axes[1].plot(
        bottom_column_profile, label="bottom imaging half", linewidth=0.9
    )
    axes[1].axvline(zlp_width, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{title_prefix} column mean, imaging pixels only")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel(f"Mean {value_label}")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--processed-file", type=Path, required=True)
    parser.add_argument("--dark-file", type=Path, required=True)
    parser.add_argument("--raw-dataset", default="/frames")
    parser.add_argument("--processed-dataset", default="/processed")
    parser.add_argument("--dark-dataset", default="/processed")
    parser.add_argument("--valid-mask-dataset", default="/valid_pixel_mask")
    parser.add_argument("--raw-start-frame", type=int, default=256)
    parser.add_argument("--edge-rows", type=int, default=32)
    parser.add_argument("--blr-rows", type=int, default=30)
    parser.add_argument("--zlp-width", type=int, default=768)
    parser.add_argument("--zlp-group-columns", type=int, default=4)
    parser.add_argument("--core-group-columns", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--histogram-limit", type=float, default=20000.0)
    parser.add_argument(
        "--frame-indices",
        default="0,1,2,8,16,32,64,96,128,255",
        help="Comma-separated processed-frame indices for pair plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nio_processing_analysis"),
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    configure_matplotlib_cache()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(args.raw_file, "r") as raw_h5,
        h5py.File(args.processed_file, "r") as processed_h5,
        h5py.File(args.dark_file, "r") as dark_h5,
    ):
        raw = raw_h5[args.raw_dataset]
        processed = processed_h5[args.processed_dataset]
        dark = dark_h5[args.dark_dataset][0].astype(np.float32)
        valid = dark_h5[args.valid_mask_dataset][0].astype(bool)

        if raw.ndim != 3 or processed.ndim != 3:
            raise ValueError("raw and processed datasets must have shape [frames, rows, cols]")
        if raw.shape[1:] != processed.shape[1:] or raw.shape[1:] != dark.shape:
            raise ValueError(
                f"shape mismatch: raw={raw.shape}, processed={processed.shape}, dark={dark.shape}"
            )
        if args.raw_start_frame + processed.shape[0] > raw.shape[0]:
            raise ValueError("processed frames extend beyond the aligned raw dataset")

        frame_count, height, width = processed.shape
        frame_indices = parse_frame_indices(args.frame_indices, frame_count)
        top_rows, bottom_rows = detector_regions(height, args.edge_rows)
        imaging_valid = valid.copy()
        imaging_valid[:args.edge_rows] = False
        imaging_valid[height - args.edge_rows:] = False

        for processed_index in frame_indices:
            raw_index = args.raw_start_frame + processed_index
            save_frame_comparison(
                raw[raw_index],
                processed[processed_index],
                raw_index,
                processed_index,
                args.output_dir
                / f"frame_compare_processed_{processed_index:03d}_raw_{raw_index:03d}.png",
                plt,
                TwoSlopeNorm,
                np,
            )

        sum_processed = np.zeros((height, width), dtype=np.float64)
        sum_processed_sq = np.zeros((height, width), dtype=np.float64)
        sum_pre_mask = np.zeros((height, width), dtype=np.float64)
        inferred_zero_count = np.zeros((height, width), dtype=np.uint16)

        histogram_bins = np.linspace(
            -args.histogram_limit, args.histogram_limit, 401
        )
        histograms = {
            "top_pre_mask": np.zeros(400, dtype=np.int64),
            "top_processed": np.zeros(400, dtype=np.int64),
            "bottom_pre_mask": np.zeros(400, dtype=np.int64),
            "bottom_processed": np.zeros(400, dtype=np.int64),
        }

        for start in range(0, frame_count, args.chunk_size):
            end = min(frame_count, start + args.chunk_size)
            raw_block = raw[
                args.raw_start_frame + start:args.raw_start_frame + end
            ].astype(np.float32)
            processed_block = processed[start:end].astype(np.float32)
            dark_subtracted = raw_block - dark[None, :, :]
            pre_mask = subtract_imagej_blr(
                dark_subtracted,
                np,
                args.blr_rows,
                args.zlp_width,
                args.zlp_group_columns,
                args.core_group_columns,
            )

            sum_processed += processed_block.sum(axis=0, dtype=np.float64)
            sum_processed_sq += np.square(processed_block).sum(
                axis=0, dtype=np.float64
            )
            sum_pre_mask += pre_mask.sum(axis=0, dtype=np.float64)
            inferred_zero_count += (
                (processed_block == 0.0)
                & imaging_valid[None, :, :]
                & (np.abs(pre_mask) > 1e-3)
            ).sum(axis=0).astype(np.uint16)

            for name, row_slice in (("top", top_rows), ("bottom", bottom_rows)):
                region_valid = valid[row_slice]
                pre_values = pre_mask[:, row_slice, :][:, region_valid]
                processed_values = processed_block[:, row_slice, :][:, region_valid]
                histograms[f"{name}_pre_mask"] += np.histogram(
                    pre_values, bins=histogram_bins
                )[0]
                histograms[f"{name}_processed"] += np.histogram(
                    processed_values, bins=histogram_bins
                )[0]

        mean_processed = (sum_processed / frame_count).astype(np.float32)
        mean_pre_mask = (sum_pre_mask / frame_count).astype(np.float32)
        processed_std = np.sqrt(
            np.maximum(
                sum_processed_sq / frame_count
                - np.square(mean_processed.astype(np.float64)),
                0.0,
            )
        ).astype(np.float32)

        pre_mask_limit = symmetric_limit(mean_pre_mask, np, 99.5)
        processed_limit = symmetric_limit(mean_processed, np, 99.5)
        std_vmin, std_vmax = robust_limits(processed_std, np, 0.5, 99.5)

        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        images = [
            axes[0, 0].imshow(
                mean_pre_mask,
                cmap="coolwarm",
                norm=TwoSlopeNorm(
                    vcenter=0.0, vmin=-pre_mask_limit, vmax=pre_mask_limit
                ),
                aspect="auto",
            ),
            axes[0, 1].imshow(
                mean_processed,
                cmap="coolwarm",
                norm=TwoSlopeNorm(
                    vcenter=0.0, vmin=-processed_limit, vmax=processed_limit
                ),
                aspect="auto",
            ),
            axes[1, 0].imshow(
                processed_std,
                cmap="viridis",
                vmin=std_vmin,
                vmax=std_vmax,
                aspect="auto",
            ),
            axes[1, 1].imshow(
                inferred_zero_count,
                cmap="viridis",
                vmin=0,
                vmax=max(1, int(inferred_zero_count.max())),
                aspect="auto",
            ),
        ]
        titles = [
            "Mean dark + BLR corrected, before masks",
            "Mean fully processed output",
            "Processed temporal standard deviation",
            "Inferred runtime zero count per imaging pixel",
        ]
        for axis, image, title in zip(axes.ravel(), images, titles):
            axis.set_title(title)
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        fig.savefig(args.output_dir / "summary_maps.png", dpi=180)
        plt.close(fig)

        centers = 0.5 * (histogram_bins[:-1] + histogram_bins[1:])
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for axis, half_name, row_slice in (
            (axes[0], "top", top_rows),
            (axes[1], "bottom", bottom_rows),
        ):
            axis.semilogy(
                centers,
                histograms[f"{half_name}_pre_mask"] + 1,
                label="dark + BLR corrected",
            )
            axis.semilogy(
                centers,
                histograms[f"{half_name}_processed"] + 1,
                label="fully processed",
            )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_title(
                f"{half_name.title()} detector imaging rows "
                f"{row_slice.start}..{row_slice.stop - 1}"
            )
            axis.set_xlabel("Pixel value")
            axis.set_ylabel("Count + 1")
            axis.legend()
        fig.suptitle("Imaging-pixel residual histograms by detector half")
        fig.savefig(args.output_dir / "histograms_by_detector_half.png", dpi=180)
        plt.close(fig)

        save_mean_profiles(
            mean_pre_mask,
            top_rows,
            bottom_rows,
            args.edge_rows,
            args.zlp_width,
            "Dark-subtracted + BLR-corrected pre-mask data",
            "pre-mask value",
            args.output_dir / "dark_blr_pre_mask_profiles.png",
            plt,
            np,
        )
        save_mean_profiles(
            mean_processed,
            top_rows,
            bottom_rows,
            args.edge_rows,
            args.zlp_width,
            "Fully processed data",
            "processed value",
            args.output_dir / "processed_profiles.png",
            plt,
            np,
        )

        summary = {
            "raw_file": str(args.raw_file),
            "processed_file": str(args.processed_file),
            "dark_file": str(args.dark_file),
            "raw_start_frame": args.raw_start_frame,
            "processed_frames": int(frame_count),
            "frame_shape": [int(height), int(width)],
            "edge_rows_excluded": args.edge_rows,
            "top_imaging_rows": [top_rows.start, top_rows.stop - 1],
            "bottom_imaging_rows": [bottom_rows.start, bottom_rows.stop - 1],
            "blr_rows": args.blr_rows,
            "inferred_zeroed_imaging_pixels": int(
                np.count_nonzero(inferred_zero_count)
            ),
            "max_inferred_zero_count": int(inferred_zero_count.max()),
            "mean_pre_mask": float(mean_pre_mask[imaging_valid].mean()),
            "mean_processed": float(mean_processed[imaging_valid].mean()),
            "mean_processed_stddev": float(processed_std[imaging_valid].mean()),
        }
        (args.output_dir / "processing_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
