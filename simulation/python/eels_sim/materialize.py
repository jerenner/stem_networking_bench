from __future__ import annotations

from typing import Mapping

import numpy as np

from .phase_space import (
    OPTIONAL_PHASE_SPACE_DTYPES,
    PHASE_SPACE_DTYPES,
    validate_phase_space,
)

MATERIALIZATION_MODES = {"poisson", "categorical", "passthrough"}


def _poisson_indices(
    weights: np.ndarray,
    rng: np.random.Generator,
    max_electrons: int,
) -> np.ndarray:
    counts = rng.poisson(weights)
    output_count = int(counts.sum())
    if output_count > max_electrons:
        raise ValueError(
            f"Materialization produced {output_count} electrons, exceeding "
            f"the configured limit of {max_electrons}"
        )
    return np.repeat(np.arange(len(weights), dtype=np.int64), counts)


def _categorical_indices(
    electrons: Mapping[str, np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
    max_electrons: int,
) -> tuple[np.ndarray, int]:
    if "parent_electron_id" not in electrons:
        raise ValueError("Categorical materialization requires parent_electron_id")
    parents = np.asarray(electrons["parent_electron_id"], dtype=np.uint64)
    frame_ids = np.asarray(electrons["frame_id"], dtype=np.uint64)
    selected: list[int] = []
    dropped = 0
    order = np.argsort(parents, kind="stable")
    sorted_parents = parents[order]
    unique_parents, starts = np.unique(sorted_parents, return_index=True)
    stops = np.append(starts[1:], len(order))
    for parent, start, stop in zip(unique_parents, starts, stops):
        candidates = order[start:stop]
        if len(np.unique(frame_ids[candidates])) != 1:
            raise ValueError("All branches of one parent must have the same frame_id")
        probabilities = weights[candidates]
        total_probability = float(probabilities.sum())
        if total_probability > 1.0 + 1.0e-9:
            raise ValueError(
                f"Branch probabilities for parent {int(parent)} sum to "
                f"{total_probability:.12g}, greater than one"
            )
        draw = rng.random()
        if draw >= min(total_probability, 1.0):
            dropped += 1
            continue
        choice = int(np.searchsorted(np.cumsum(probabilities), draw, side="right"))
        selected.append(int(candidates[min(choice, len(candidates) - 1)]))
    if len(selected) > max_electrons:
        raise ValueError(
            f"Materialization produced {len(selected)} electrons, exceeding "
            f"the configured limit of {max_electrons}"
        )
    return np.asarray(selected, dtype=np.int64), dropped


def materialize_phase_space(
    electrons: Mapping[str, np.ndarray],
    mode: str,
    random_seed: int,
    max_electrons: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Sample weighted phase space into unit-weight individual electrons."""
    input_count = validate_phase_space(electrons)
    if mode not in MATERIALIZATION_MODES:
        raise ValueError(f"mode must be one of {sorted(MATERIALIZATION_MODES)}")
    if max_electrons <= 0:
        raise ValueError("max_electrons must be positive")
    weights = np.asarray(electrons["weight"], dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    dropped_parent_electrons = 0
    if mode == "passthrough":
        if not np.allclose(weights, 1.0):
            raise ValueError("Passthrough materialization requires unit weights")
        if input_count > max_electrons:
            raise ValueError(
                f"Materialization received {input_count} electrons, exceeding "
                f"the configured limit of {max_electrons}"
            )
        selected = np.arange(input_count, dtype=np.int64)
    elif mode == "poisson":
        selected = _poisson_indices(weights, rng, max_electrons)
    else:
        selected, dropped_parent_electrons = _categorical_indices(
            electrons, weights, rng, max_electrons
        )
    if len(selected) == 0:
        raise ValueError("Materialization produced zero electrons")

    output = {
        name: np.asarray(electrons[name], dtype=dtype)[selected].copy()
        for name, dtype in PHASE_SPACE_DTYPES.items()
    }
    for name, dtype in OPTIONAL_PHASE_SPACE_DTYPES.items():
        if name in electrons and name != "source_record_id":
            output[name] = np.asarray(electrons[name], dtype=dtype)[selected].copy()
    output["source_record_id"] = np.asarray(electrons["electron_id"], dtype=np.uint64)[selected]
    output["electron_id"] = np.arange(len(selected), dtype=np.uint64)
    output["weight"] = np.ones(len(selected), dtype=np.float64)
    validate_phase_space(output)

    frames, frame_counts = np.unique(output["frame_id"], return_counts=True)
    summary: dict[str, object] = {
        "mode": mode,
        "random_seed": int(random_seed),
        "input_records": int(input_count),
        "input_weight_sum": float(weights.sum()),
        "output_electrons": int(len(selected)),
        "output_frames_with_electrons": int(len(frames)),
        "minimum_electrons_per_nonempty_frame": int(frame_counts.min()),
        "maximum_electrons_per_nonempty_frame": int(frame_counts.max()),
        "dropped_parent_electrons": int(dropped_parent_electrons),
    }
    return output, summary
