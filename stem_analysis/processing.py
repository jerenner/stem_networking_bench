"""CPU/NumPy mirror of the Holoscan PyTorchProcessorOp correction chain."""

from __future__ import annotations

from .config import ProcessorConfig


def compute_blr_baseline(block,
                         blr_rows: int,
                         zlp_width: int,
                         zlp_group_columns: int,
                         core_group_columns: int,
                         np):
    """Compute grouped edge-row BLR baselines for an already dark-subtracted tensor."""
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

    for half_index, rows in enumerate((slice(0, blr_rows), slice(height - blr_rows, height))):
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
        baseline[:, half_index, :] = np.concatenate(parts, axis=1)

    return baseline


def subtract_blr_baseline(block,
                          baseline,
                          zlp_width: int,
                          zlp_group_columns: int,
                          core_group_columns: int,
                          np) -> None:
    """Subtract grouped BLR baselines in-place."""
    _, height, width = block.shape
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


def subtract_imagej_blr(block,
                        np,
                        blr_rows: int,
                        zlp_width: int,
                        zlp_group_columns: int,
                        core_group_columns: int):
    """Return a BLR-subtracted copy matching the ImageJ BLR_v1.ijm strategy."""
    corrected = block.astype(np.float32, copy=True)
    baseline = compute_blr_baseline(
        corrected, blr_rows, zlp_width, zlp_group_columns, core_group_columns, np
    )
    subtract_blr_baseline(
        corrected, baseline, zlp_width, zlp_group_columns, core_group_columns, np
    )
    return corrected


def half_column_local_median(batch_mean,
                             median_window_pixels: int,
                             excluded_edge_rows: int,
                             np):
    """Compute the same top/bottom half same-column median used by the CUDA mask kernel."""
    height, _ = batch_mean.shape
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
                                 config: ProcessorConfig,
                                 np):
    """Apply static valid-pixel and dynamic half-column masks in-place."""
    if not config.apply_valid_pixel_mask and not config.apply_dynamic_half_column_mask:
        return np.zeros(block.shape[1:], dtype=bool)

    height, width = block.shape[1:]
    should_zero = np.zeros((height, width), dtype=bool)

    if config.apply_valid_pixel_mask:
        if valid_pixel_mask is None:
            raise ValueError("valid pixel mask requested but not loaded")
        should_zero |= valid_pixel_mask == 0.0

    if config.apply_dynamic_half_column_mask:
        batch_mean = block.mean(axis=0, dtype=np.float32)
        local_median = half_column_local_median(
            batch_mean,
            config.dynamic_mask_median_window_pixels,
            config.dynamic_mask_excluded_edge_rows,
            np,
        )
        reference = local_median * config.dynamic_mask_threshold_ratio
        deviation = batch_mean - reference
        dynamic_zero = (
            np.abs(deviation) > config.dynamic_mask_threshold_offset
            if config.dynamic_mask_two_sided
            else deviation > config.dynamic_mask_threshold_offset
        )

        dynamic_allowed = np.zeros((height, width), dtype=bool)
        half_height = height // 2
        edge = config.dynamic_mask_excluded_edge_rows
        dynamic_allowed[edge:half_height, :] = True
        dynamic_allowed[half_height:height - edge, :] = True
        should_zero |= dynamic_zero & dynamic_allowed & ~should_zero

    if np.any(should_zero):
        block[:, should_zero] = 0.0
    return should_zero


def process_tensor_block(raw_block,
                         dark_frame,
                         valid_pixel_mask,
                         config: ProcessorConfig,
                         np):
    """Apply the same ordered corrections as the Holoscan PyTorchProcessorOp."""
    block = raw_block.astype(np.float32, copy=True)

    if config.subtract_dark_frame:
        if dark_frame is None:
            raise ValueError("dark-frame subtraction requested but no dark frame was provided")
        block -= dark_frame[None, :, :]

    if config.apply_blr_correction:
        baseline = compute_blr_baseline(
            block,
            config.blr_rows,
            config.blr_zlp_width,
            config.blr_zlp_group_columns,
            config.blr_core_group_columns,
            np,
        )
        subtract_blr_baseline(
            block,
            baseline,
            config.blr_zlp_width,
            config.blr_zlp_group_columns,
            config.blr_core_group_columns,
            np,
        )

    zero_mask = apply_dynamic_and_valid_mask(block, valid_pixel_mask, config, np)

    if config.noop:
        return block.astype(np.float32, copy=False), zero_mask
    return block.sum(axis=0, dtype=np.float32)[None, :, :], zero_mask

