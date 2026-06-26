"""Spectrum construction helpers for ZLP/CoreLoss detector layouts."""

from __future__ import annotations


def safe_divide(numerator, denominator, np):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def fold_zlp(sums, counts, zlp_width: int, zlp_period: int, np):
    if zlp_width % zlp_period != 0:
        raise ValueError("zlp_width must be divisible by zlp_period")
    repeats = zlp_width // zlp_period
    folded_sums = sums[..., :zlp_width].reshape(
        *sums.shape[:-1], repeats, zlp_period
    ).sum(axis=-2)
    folded_counts = counts[..., :zlp_width].reshape(
        *counts.shape[:-1], repeats, zlp_period
    ).sum(axis=-2)
    return safe_divide(folded_sums, folded_counts, np), folded_sums, folded_counts


def collapsed_stitch_profile(sums, counts, np, zlp_width: int = 768, zlp_period: int = 192):
    """Return [summed repeated ZLP physical columns, CoreLoss output columns]."""
    if sums.ndim == 2:
        sums = sums.sum(axis=0)
        counts = counts.sum(axis=0)
    detector_profile = safe_divide(sums, counts, np)
    zlp_repeats = zlp_width // zlp_period
    zlp = detector_profile[:zlp_width].reshape(zlp_repeats, zlp_period).sum(axis=0)
    return np.concatenate([zlp, detector_profile[zlp_width:]])

