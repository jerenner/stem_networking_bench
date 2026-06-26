"""Shared plotting and detector-geometry helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "stem-networking-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def detector_regions(height: int, edge_rows: int) -> tuple[slice, slice]:
    half_height = height // 2
    if height % 2 != 0:
        raise ValueError(f"detector height must be even, got {height}")
    if edge_rows < 0 or edge_rows >= half_height:
        raise ValueError(f"edge_rows must be in [0, {half_height - 1}], got {edge_rows}")
    return slice(edge_rows, half_height), slice(half_height, height - edge_rows)


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


def parse_frame_indices(text: str, frame_count: int) -> list[int]:
    indices = [int(value.strip()) for value in text.split(",") if value.strip()]
    invalid = [index for index in indices if index < 0 or index >= frame_count]
    if invalid:
        raise ValueError(f"frame indices outside [0, {frame_count - 1}]: {invalid}")
    return indices

