"""Configuration objects for offline processor-equivalent analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessorConfig:
    """CPU-side mirror of the PyTorchProcessorOp correction knobs."""

    noop: bool = True
    subtract_dark_frame: bool = False
    apply_valid_pixel_mask: bool = False
    apply_blr_correction: bool = False
    blr_rows: int = 30
    blr_zlp_width: int = 768
    blr_zlp_group_columns: int = 4
    blr_core_group_columns: int = 16
    apply_dynamic_half_column_mask: bool = False
    dynamic_mask_median_window_pixels: int = 31
    dynamic_mask_threshold_ratio: float = 1.0
    dynamic_mask_threshold_offset: float = 500.0
    dynamic_mask_excluded_edge_rows: int = 32
    dynamic_mask_two_sided: bool = True

