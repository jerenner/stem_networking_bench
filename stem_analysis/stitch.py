"""ZLP/CoreLoss stitch calibration and boundary repair utilities."""

from __future__ import annotations

from .spectra import safe_divide

HERMITE_LEFT_ENDPOINT = 190
HERMITE_RIGHT_ENDPOINT = 194
HERMITE_FILL_COLUMNS = (191, 192, 193)
HERMITE_LEFT_SLOPE_WINDOW = (184, 190)
HERMITE_RIGHT_SLOPE_WINDOW = (194, 200)


def estimate_linear_slope(profile, start: int, end: int, np) -> float:
    x = np.arange(start, end + 1, dtype=np.float64)
    y = profile[start:end + 1].astype(np.float64)
    valid = np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return 0.0
    x = x[valid]
    y = y[valid]
    x_centered = x - x.mean()
    denominator = np.sum(x_centered * x_centered)
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def apply_hermite_boundary_repair(profile, np):
    """Replace columns 191..193 with a C1-continuous cubic Hermite segment."""
    repaired = profile.copy()
    left = HERMITE_LEFT_ENDPOINT
    right = HERMITE_RIGHT_ENDPOINT
    y0 = float(repaired[left])
    y1 = float(repaired[right])
    if not np.isfinite(y0) or not np.isfinite(y1):
        return repaired, {
            "hermite_left_slope": float("nan"),
            "hermite_right_slope": float("nan"),
            "hermite_factor_191": float("nan"),
            "hermite_factor_192": float("nan"),
            "hermite_factor_193": float("nan"),
        }

    left_slope = estimate_linear_slope(
        repaired, HERMITE_LEFT_SLOPE_WINDOW[0], HERMITE_LEFT_SLOPE_WINDOW[1], np
    )
    right_slope = estimate_linear_slope(
        repaired, HERMITE_RIGHT_SLOPE_WINDOW[0], HERMITE_RIGHT_SLOPE_WINDOW[1], np
    )
    span = float(right - left)
    factors = {}
    for column in HERMITE_FILL_COLUMNS:
        observed = float(repaired[column])
        t = (column - left) / span
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        repaired[column] = (
            h00 * y0
            + h10 * span * left_slope
            + h01 * y1
            + h11 * span * right_slope
        )
        factors[f"hermite_factor_{column}"] = (
            float(repaired[column] / observed)
            if np.isfinite(observed) and abs(observed) > 1e-12
            else float("nan")
        )

    return repaired, {
        "hermite_left_slope": left_slope,
        "hermite_right_slope": right_slope,
        **factors,
    }


def apply_stitch_calibration(profile, gain: float, np):
    stitched = profile.copy()
    stitched[:192] *= gain
    return apply_hermite_boundary_repair(stitched, np)


def robust_log_quadratic_stitch_fit(profile,
                                    np,
                                    left_start=160,
                                    left_end=191,
                                    right_start=194,
                                    right_end=224,
                                    degree=2,
                                    huber_delta=1.345):
    """Fit a smooth log-polynomial plus a ZLP step, excluding columns 191..193."""
    left_x = np.arange(left_start, left_end)
    right_x = np.arange(right_start, right_end)
    x = np.concatenate([left_x, right_x])
    y = profile[x]
    valid = np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.size < degree + 4 or not np.any(x < 192) or not np.any(x >= 192):
        raise ValueError("insufficient positive samples for stitch fit")

    center = 191.5
    scale_x = max(right_end - left_start, 1)
    normalized_x = (x - center) / scale_x
    polynomial = np.column_stack([normalized_x**power for power in range(degree + 1)])
    left_indicator = (x < 192).astype(np.float64)
    design = np.column_stack([polynomial, -left_indicator])
    target = np.log(y)
    weights = np.ones_like(target)
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(30):
        residual = target - design @ beta
        robust_scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if robust_scale <= 1e-12:
            break
        cutoff = huber_delta * robust_scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(
            design * root_weights[:, None], target * root_weights, rcond=None
        )[0]
        if np.max(np.abs(updated - beta)) < 1e-10:
            beta = updated
            break
        beta = updated

    log_gain = float(beta[-1])
    gain = float(np.exp(log_gain))
    residual = target - design @ beta
    corrected = profile.copy()
    corrected[:192] *= gain
    hermite_corrected, hermite_metrics = apply_hermite_boundary_repair(corrected, np)

    def predict(columns):
        normalized = (np.asarray(columns) - center) / scale_x
        basis = np.column_stack([normalized**power for power in range(degree + 1)])
        return np.exp(basis @ beta[: degree + 1])

    boundary_columns = np.array([191, 192, 193])
    boundary_factors = safe_divide(predict(boundary_columns), corrected[boundary_columns], np)
    return {
        "gain": gain,
        "log_gain": log_gain,
        "fit_log_rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
        "fit_log_mad": float(np.median(np.abs(residual))),
        "fit_point_count": int(x.size),
        "positive_fraction": float(valid.mean()),
        "direct_core0_to_zlp191_before": float(profile[192] / profile[191]),
        "direct_core0_to_zlp191_after": float(corrected[192] / corrected[191]),
        "boundary_factor_191": float(boundary_factors[0]),
        "boundary_factor_192": float(boundary_factors[1]),
        "boundary_factor_193": float(boundary_factors[2]),
        **hermite_metrics,
        "corrected": corrected,
        "hermite_corrected": hermite_corrected,
        "predict": predict,
    }

