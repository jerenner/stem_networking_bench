from __future__ import annotations

import copy
import csv
import json
import math
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .abtem_adapter import AbTEMBackend, AngularDistribution

AXIS_ORDER = (
    "lateral_repetitions",
    "thickness_repetitions",
    "potential_sampling_A",
    "max_scattering_angle_mrad",
    "double_channel",
)

AXIS_LABELS = {
    "lateral_repetitions": "lateral cell",
    "thickness_repetitions": "thickness",
    "potential_sampling_A": "real-space sampling",
    "max_scattering_angle_mrad": "angular cutoff",
    "double_channel": "channeling model",
}


@dataclass(frozen=True)
class ConvergenceCase:
    case_id: str
    axis: str | None
    value: object
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ConvergenceDefinition:
    beam_energy_eV: float
    radial_bins: int
    histogram_max_angle_mrad: float
    comparison_collection_angle_mrad: float | None
    baseline: Mapping[str, object]
    cases: tuple[ConvergenceCase, ...]
    axis_case_ids: Mapping[str, tuple[str, ...]]
    reference_case_ids: Mapping[str, str]
    combined_case_ids: Mapping[str, str]


def _as_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_as_json(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return False
        return list(left) == list(right)
    return left == right


def _positive_int(value: object, name: str) -> int:
    number = int(value)
    if number != value or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _repetitions(parameters: Mapping[str, object]) -> list[int]:
    values = parameters.get("repetitions", [1, 1, 1])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("baseline.repetitions must be a three-element array")
    result = [_positive_int(value, "baseline.repetitions entry") for value in values]
    if len(result) != 3:
        raise ValueError("baseline.repetitions must be a three-element array")
    return result


def _core_loss(parameters: Mapping[str, object]) -> dict[str, object]:
    value = parameters.get("core_loss", {})
    if not isinstance(value, Mapping):
        raise ValueError("baseline.core_loss must be a TOML table")
    return copy.deepcopy(dict(value))


def _baseline_axis_value(axis: str, baseline: Mapping[str, object]) -> object:
    repetitions = _repetitions(baseline)
    if axis == "lateral_repetitions":
        return repetitions[:2]
    if axis == "thickness_repetitions":
        return repetitions[2]
    if axis == "potential_sampling_A":
        return float(baseline[axis])
    if axis == "max_scattering_angle_mrad":
        return float(baseline[axis])
    if axis == "double_channel":
        return bool(_core_loss(baseline).get("double_channel", False))
    raise KeyError(axis)


def _validated_axis_values(axis: str, values: object) -> list[object]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"axes.{axis} must be a nonempty TOML array")
    if axis == "lateral_repetitions":
        result: list[object] = []
        for value in values:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("axes.lateral_repetitions entries must have two integers")
            result.append([_positive_int(item, "lateral repetition") for item in value])
        return result
    if axis == "thickness_repetitions":
        return [_positive_int(value, "thickness repetition") for value in values]
    if axis in {"potential_sampling_A", "max_scattering_angle_mrad"}:
        result = [float(value) for value in values]
        if any(not math.isfinite(value) or value <= 0.0 for value in result):
            raise ValueError(f"axes.{axis} values must be finite and positive")
        return result
    if axis == "double_channel":
        if any(not isinstance(value, bool) for value in values):
            raise ValueError("axes.double_channel values must be true or false")
        return list(values)
    raise KeyError(axis)


def _case_parameters(baseline: Mapping[str, object], axis: str, value: object) -> dict[str, object]:
    parameters = copy.deepcopy(dict(baseline))
    repetitions = _repetitions(parameters)
    if axis == "lateral_repetitions":
        assert isinstance(value, list)
        repetitions[:2] = [int(value[0]), int(value[1])]
        parameters["repetitions"] = repetitions
    elif axis == "thickness_repetitions":
        repetitions[2] = int(value)
        parameters["repetitions"] = repetitions
    elif axis in {"potential_sampling_A", "max_scattering_angle_mrad"}:
        parameters[axis] = float(value)
    elif axis == "double_channel":
        core_loss = _core_loss(parameters)
        core_loss["double_channel"] = bool(value)
        parameters["core_loss"] = core_loss
    else:
        raise KeyError(axis)
    return parameters


def _value_slug(axis: str, value: object) -> str:
    if axis == "lateral_repetitions":
        assert isinstance(value, list)
        return f"{int(value[0])}x{int(value[1])}"
    if axis == "thickness_repetitions":
        return f"z{int(value)}"
    if axis == "potential_sampling_A":
        return f"{float(value):g}A".replace(".", "p")
    if axis == "max_scattering_angle_mrad":
        return f"{float(value):g}mrad".replace(".", "p")
    if axis == "double_channel":
        return "double" if value else "single"
    raise KeyError(axis)


def _reference_value(axis: str, values: Sequence[object], baseline: object) -> object:
    if axis == "lateral_repetitions":
        return max(values, key=lambda item: int(item[0]) * int(item[1]))  # type: ignore[index]
    if axis == "thickness_repetitions":
        return baseline
    if axis == "potential_sampling_A":
        return min(float(value) for value in values)
    if axis == "max_scattering_angle_mrad":
        return max(float(value) for value in values)
    if axis == "double_channel":
        return True if True in values else baseline
    raise KeyError(axis)


def _combined_cases(
    document: Mapping[str, object], baseline: Mapping[str, object]
) -> tuple[list[ConvergenceCase], dict[str, str]]:
    combined = document.get("combined_check")
    if combined is None:
        return [], {}
    if not isinstance(combined, Mapping):
        raise ValueError("combined_check must be a TOML table")
    roles = ("reference", "confirmation")
    unknown_roles = set(combined) - set(roles)
    if unknown_roles:
        raise ValueError(f"Unknown combined_check entries: {sorted(unknown_roles)}")
    missing_roles = set(roles) - set(combined)
    if missing_roles:
        raise ValueError(f"Missing combined_check entries: {sorted(missing_roles)}")

    cases: list[ConvergenceCase] = []
    case_ids: dict[str, str] = {}
    allowed = {"case_id", *AXIS_ORDER}
    for role in roles:
        settings = combined[role]
        if not isinstance(settings, Mapping):
            raise ValueError(f"combined_check.{role} must be a TOML table")
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unknown combined_check.{role} entries: {sorted(unknown)}")
        configured_axes = [axis for axis in AXIS_ORDER if axis in settings]
        if not configured_axes:
            raise ValueError(f"combined_check.{role} must override at least one convergence axis")
        case_id = str(settings.get("case_id", f"combined_{role}"))
        if not case_id or any(character.isspace() for character in case_id):
            raise ValueError(f"combined_check.{role}.case_id must be nonempty without whitespace")
        parameters = copy.deepcopy(dict(baseline))
        overrides: dict[str, object] = {}
        for axis in configured_axes:
            value = _validated_axis_values(axis, [settings[axis]])[0]
            parameters = _case_parameters(parameters, axis, value)
            overrides[axis] = _as_json(value)
        cases.append(
            ConvergenceCase(
                case_id=case_id,
                axis="combined_check",
                value={"role": role, "overrides": overrides},
                parameters=parameters,
            )
        )
        case_ids[role] = case_id
    return cases, case_ids


def load_convergence_definition(path: Path) -> ConvergenceDefinition:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    study = document.get("study", {})
    baseline = document.get("baseline")
    axes = document.get("axes")
    if not isinstance(study, Mapping):
        raise ValueError("study must be a TOML table")
    if not isinstance(baseline, Mapping):
        raise ValueError("baseline must be a TOML table")
    if not isinstance(axes, Mapping):
        raise ValueError("axes must be a TOML table")
    unknown_axes = set(axes) - set(AXIS_ORDER)
    if unknown_axes:
        raise ValueError(f"Unknown convergence axes: {sorted(unknown_axes)}")
    missing_axes = set(AXIS_ORDER) - set(axes)
    if missing_axes:
        raise ValueError(f"Missing convergence axes: {sorted(missing_axes)}")

    beam_energy_eV = float(study.get("beam_energy_eV", 300_000.0))
    radial_bins = _positive_int(study.get("radial_bins", 120), "study.radial_bins")
    if beam_energy_eV <= 0.0 or not math.isfinite(beam_energy_eV):
        raise ValueError("study.beam_energy_eV must be finite and positive")
    if radial_bins < 8:
        raise ValueError("study.radial_bins must be at least 8")

    baseline_copy = copy.deepcopy(dict(baseline))
    _repetitions(baseline_copy)
    combined_cases, combined_case_ids = _combined_cases(document, baseline_copy)
    axis_values = {axis: _validated_axis_values(axis, axes[axis]) for axis in AXIS_ORDER}
    maximum_configured_angle = max(
        float(_baseline_axis_value("max_scattering_angle_mrad", baseline_copy)),
        *(float(value) for value in axis_values["max_scattering_angle_mrad"]),
        *(float(case.parameters["max_scattering_angle_mrad"]) for case in combined_cases),
    )
    histogram_max_angle_mrad = float(
        study.get("histogram_max_angle_mrad", maximum_configured_angle)
    )
    if (
        not math.isfinite(histogram_max_angle_mrad)
        or histogram_max_angle_mrad < maximum_configured_angle
    ):
        raise ValueError("study.histogram_max_angle_mrad must cover every angular cutoff")
    collection_value = study.get("comparison_collection_angle_mrad")
    comparison_collection_angle_mrad = None if collection_value is None else float(collection_value)
    if comparison_collection_angle_mrad is not None and (
        not math.isfinite(comparison_collection_angle_mrad)
        or comparison_collection_angle_mrad <= 0.0
        or comparison_collection_angle_mrad > histogram_max_angle_mrad
    ):
        raise ValueError(
            "study.comparison_collection_angle_mrad must be positive and no "
            "larger than the histogram range"
        )

    cases = [ConvergenceCase("baseline", None, None, baseline_copy)]
    axis_case_ids: dict[str, tuple[str, ...]] = {}
    reference_case_ids: dict[str, str] = {}
    for axis in AXIS_ORDER:
        baseline_value = _baseline_axis_value(axis, baseline_copy)
        values = axis_values[axis]
        ids: list[str] = []
        value_to_id: list[tuple[object, str]] = []
        for value in values:
            if _same_value(value, baseline_value):
                case_id = "baseline"
            else:
                case_id = f"{axis}_{_value_slug(axis, value)}"
                cases.append(
                    ConvergenceCase(
                        case_id,
                        axis,
                        _as_json(value),
                        _case_parameters(baseline_copy, axis, value),
                    )
                )
            ids.append(case_id)
            value_to_id.append((value, case_id))
        if "baseline" not in ids:
            raise ValueError(f"axes.{axis} must include its baseline value {baseline_value!r}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"axes.{axis} contains duplicate values")
        axis_case_ids[axis] = tuple(ids)
        reference_value = _reference_value(axis, values, baseline_value)
        reference_case_ids[axis] = next(
            case_id for value, case_id in value_to_id if _same_value(value, reference_value)
        )

    cases.extend(combined_cases)

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Convergence axes produced duplicate case identifiers")
    return ConvergenceDefinition(
        beam_energy_eV=beam_energy_eV,
        radial_bins=radial_bins,
        histogram_max_angle_mrad=histogram_max_angle_mrad,
        comparison_collection_angle_mrad=comparison_collection_angle_mrad,
        baseline=baseline_copy,
        cases=tuple(cases),
        axis_case_ids=axis_case_ids,
        reference_case_ids=reference_case_ids,
        combined_case_ids=combined_case_ids,
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile, cumulative, sorted_values))


def _radial_metrics(radius: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "mean_radial_mrad": float(np.sum(probability * radius)),
        "rms_radial_mrad": float(np.sqrt(np.sum(probability * radius**2))),
        "p50_radial_mrad": _weighted_quantile(radius, probability, 0.50),
        "p90_radial_mrad": _weighted_quantile(radius, probability, 0.90),
        "p99_radial_mrad": _weighted_quantile(radius, probability, 0.99),
    }


def _distribution_metrics(
    distribution: AngularDistribution,
    edges: np.ndarray,
    collection_angle_mrad: float | None,
) -> tuple[dict[str, float], list[float], dict[str, object] | None]:
    radius = np.hypot(distribution.theta_x_mrad, distribution.theta_y_mrad)
    probability = distribution.probability
    histogram = np.histogram(radius, bins=edges, weights=probability)[0]
    histogram_sum = float(histogram.sum())
    if histogram_sum <= 0.0:
        raise ValueError("Angular distribution is outside the convergence histogram")
    histogram /= histogram_sum
    metrics = {
        **_radial_metrics(radius, probability),
        "centroid_x_mrad": float(np.sum(probability * distribution.theta_x_mrad)),
        "centroid_y_mrad": float(np.sum(probability * distribution.theta_y_mrad)),
    }
    collection = None
    if collection_angle_mrad is not None:
        mask = radius <= collection_angle_mrad + 1.0e-12
        fraction = float(probability[mask].sum())
        if fraction <= 0.0:
            raise ValueError("Angular distribution has no weight inside collection angle")
        collected_probability = probability[mask] / fraction
        collected_histogram = np.histogram(radius[mask], bins=edges, weights=collected_probability)[
            0
        ]
        collected_histogram /= collected_histogram.sum()
        collection = {
            "angle_mrad": collection_angle_mrad,
            "fraction": fraction,
            "metrics": _radial_metrics(radius[mask], collected_probability),
            "radial_histogram": collected_histogram.tolist(),
        }
    return metrics, histogram.tolist(), collection


def _run_case(
    backend: AbTEMBackend,
    definition: ConvergenceDefinition,
    case: ConvergenceCase,
    edges: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    model = backend.calculate(definition.beam_energy_eV, case.parameters)
    runtime_seconds = time.perf_counter() - started
    elastic_metrics, elastic_histogram, elastic_collection = _distribution_metrics(
        model.elastic, edges, definition.comparison_collection_angle_mrad
    )
    core_metrics = None
    core_histogram = None
    core_collection = None
    if model.core is not None:
        core_metrics, core_histogram, core_collection = _distribution_metrics(
            model.core, edges, definition.comparison_collection_angle_mrad
        )
    metadata = dict(model.metadata or {})
    thickness_A = float(model.specimen_thickness_A)
    elastic_captured_intensity = metadata.get("elastic_captured_intensity")
    elastic_captured_probability = (
        float(elastic_captured_intensity) if elastic_captured_intensity is not None else 1.0
    )
    return {
        "case_id": case.case_id,
        "axis": case.axis,
        "value": _as_json(case.value),
        "status": "ok",
        "parameters": _as_json(case.parameters),
        "runtime_seconds": runtime_seconds,
        "specimen_thickness_A": thickness_A,
        "atom_count": metadata.get("atom_count"),
        "potential_gpts": metadata.get("potential_gpts"),
        "potential_slices": metadata.get("potential_slices"),
        "elastic_captured_intensity": elastic_captured_intensity,
        "core_probability": float(model.core_probability),
        "core_probability_per_A": (
            float(model.core_probability / thickness_A) if thickness_A > 0.0 else None
        ),
        "core_probability_within_collection": (
            float(model.core_probability * core_collection["fraction"])
            if core_collection is not None
            else None
        ),
        "elastic_probability_within_collection": (
            float(
                (1.0 - model.core_probability)
                * elastic_captured_probability
                * elastic_collection["fraction"]
            )
            if elastic_collection is not None
            else None
        ),
        "elastic": {
            "metrics": elastic_metrics,
            "radial_histogram": elastic_histogram,
            "collection": elastic_collection,
        },
        "core": (
            {
                "metrics": core_metrics,
                "radial_histogram": core_histogram,
                "collection": core_collection,
            }
            if core_metrics is not None
            else None
        ),
        "backend_metadata": _as_json(metadata),
    }


def _relative_difference(value: object, reference: object) -> float | None:
    if value is None or reference is None:
        return None
    value_float = float(value)
    reference_float = float(reference)
    if reference_float == 0.0:
        return None
    return (value_float - reference_float) / reference_float


def _distribution_view(
    result: Mapping[str, object], branch: str, within_collection: bool
) -> Mapping[str, object] | None:
    result_branch = result.get(branch)
    if not isinstance(result_branch, Mapping):
        return None
    if not within_collection:
        return result_branch
    collection = result_branch.get("collection")
    return collection if isinstance(collection, Mapping) else None


def _total_variation(
    result: Mapping[str, object],
    reference: Mapping[str, object],
    branch: str,
    within_collection: bool = False,
) -> float | None:
    result_branch = _distribution_view(result, branch, within_collection)
    reference_branch = _distribution_view(reference, branch, within_collection)
    if not isinstance(result_branch, Mapping) or not isinstance(reference_branch, Mapping):
        return None
    left = np.asarray(result_branch["radial_histogram"], dtype=np.float64)
    right = np.asarray(reference_branch["radial_histogram"], dtype=np.float64)
    return float(0.5 * np.abs(left - right).sum())


def _radial_wasserstein(
    result: Mapping[str, object],
    reference: Mapping[str, object],
    branch: str,
    bin_width_mrad: float,
    within_collection: bool = False,
) -> float | None:
    result_branch = _distribution_view(result, branch, within_collection)
    reference_branch = _distribution_view(reference, branch, within_collection)
    if not isinstance(result_branch, Mapping) or not isinstance(reference_branch, Mapping):
        return None
    left = np.asarray(result_branch["radial_histogram"], dtype=np.float64)
    right = np.asarray(reference_branch["radial_histogram"], dtype=np.float64)
    return float(np.abs(np.cumsum(left - right)).sum() * bin_width_mrad)


def _comparison(
    result: Mapping[str, object],
    reference: Mapping[str, object],
    bin_width_mrad: float,
) -> dict[str, float | None] | None:
    if result.get("status") != "ok" or reference.get("status") != "ok":
        return None
    return {
        "relative_core_probability": _relative_difference(
            result.get("core_probability"), reference.get("core_probability")
        ),
        "relative_core_probability_per_A": _relative_difference(
            result.get("core_probability_per_A"),
            reference.get("core_probability_per_A"),
        ),
        "relative_core_probability_within_collection": _relative_difference(
            result.get("core_probability_within_collection"),
            reference.get("core_probability_within_collection"),
        ),
        "relative_elastic_probability_within_collection": _relative_difference(
            result.get("elastic_probability_within_collection"),
            reference.get("elastic_probability_within_collection"),
        ),
        "relative_elastic_captured_intensity": _relative_difference(
            result.get("elastic_captured_intensity"),
            reference.get("elastic_captured_intensity"),
        ),
        "elastic_radial_total_variation": _total_variation(result, reference, "elastic"),
        "core_radial_total_variation": _total_variation(result, reference, "core"),
        "elastic_radial_wasserstein_mrad": _radial_wasserstein(
            result, reference, "elastic", bin_width_mrad
        ),
        "core_radial_wasserstein_mrad": _radial_wasserstein(
            result, reference, "core", bin_width_mrad
        ),
        "elastic_collection_radial_total_variation": _total_variation(
            result, reference, "elastic", within_collection=True
        ),
        "core_collection_radial_total_variation": _total_variation(
            result, reference, "core", within_collection=True
        ),
        "elastic_collection_radial_wasserstein_mrad": _radial_wasserstein(
            result,
            reference,
            "elastic",
            bin_width_mrad,
            within_collection=True,
        ),
        "core_collection_radial_wasserstein_mrad": _radial_wasserstein(
            result,
            reference,
            "core",
            bin_width_mrad,
            within_collection=True,
        ),
        "relative_elastic_rms_radial": _relative_difference(
            _metric(result, "elastic", "rms_radial_mrad"),
            _metric(reference, "elastic", "rms_radial_mrad"),
        ),
        "relative_core_rms_radial": _relative_difference(
            _metric(result, "core", "rms_radial_mrad"),
            _metric(reference, "core", "rms_radial_mrad"),
        ),
        "relative_elastic_collection_rms_radial": _relative_difference(
            _collection_metric(result, "elastic", "rms_radial_mrad"),
            _collection_metric(reference, "elastic", "rms_radial_mrad"),
        ),
        "relative_core_collection_rms_radial": _relative_difference(
            _collection_metric(result, "core", "rms_radial_mrad"),
            _collection_metric(reference, "core", "rms_radial_mrad"),
        ),
    }


def _axis_value(axis: str, case: ConvergenceCase, baseline: Mapping[str, object]) -> object:
    return _baseline_axis_value(axis, baseline) if case.case_id == "baseline" else case.value


def _assemble_report(
    definition: ConvergenceDefinition,
    results: list[dict[str, object]],
    edges: np.ndarray,
    config_path: Path,
) -> dict[str, object]:
    by_id = {str(result["case_id"]): result for result in results}
    baseline = by_id["baseline"]
    bin_width_mrad = float(edges[1] - edges[0])
    for result in results:
        result["comparison_to_baseline"] = _comparison(result, baseline, bin_width_mrad)

    case_by_id = {case.case_id: case for case in definition.cases}
    axes: dict[str, object] = {}
    for axis in AXIS_ORDER:
        reference_id = definition.reference_case_ids[axis]
        reference = by_id[reference_id]
        points = []
        for case_id in definition.axis_case_ids[axis]:
            case = case_by_id[case_id]
            result = by_id[case_id]
            points.append(
                {
                    "case_id": case_id,
                    "value": _as_json(_axis_value(axis, case, definition.baseline)),
                    "status": result["status"],
                    "comparison_to_reference": _comparison(result, reference, bin_width_mrad),
                }
            )
        axes[axis] = {
            "label": AXIS_LABELS[axis],
            "reference_case_id": reference_id,
            "interpretation": (
                "physical thickness trend; the baseline is a comparison point, "
                "not a numerical convergence limit"
                if axis == "thickness_repetitions"
                else "one-at-a-time numerical/model convergence"
            ),
            "points": points,
        }

    combined_check = None
    if definition.combined_case_ids:
        reference_id = definition.combined_case_ids["reference"]
        confirmation_id = definition.combined_case_ids["confirmation"]
        reference = by_id[reference_id]
        confirmation = by_id[confirmation_id]
        combined_check = {
            "reference_case_id": reference_id,
            "confirmation_case_id": confirmation_id,
            "comparison_direction": "confirmation relative to reference",
            "confirmation_relative_to_reference": _comparison(
                confirmation, reference, bin_width_mrad
            ),
            "points": [
                {
                    "case_id": reference_id,
                    "role": "reference",
                    "status": reference["status"],
                    "comparison_to_reference": _comparison(reference, reference, bin_width_mrad),
                },
                {
                    "case_id": confirmation_id,
                    "role": "confirmation",
                    "status": confirmation["status"],
                    "comparison_to_reference": _comparison(confirmation, reference, bin_width_mrad),
                },
            ],
            "held_constant": {
                "beam_energy_eV": definition.beam_energy_eV,
                "thickness_repetitions": _repetitions(definition.baseline)[2],
                "specimen_thickness_A": reference.get("specimen_thickness_A"),
            },
        }

    return {
        "schema": "eels-sim-abtem-convergence-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(Path(config_path).resolve()),
        "study": {
            "beam_energy_eV": definition.beam_energy_eV,
            "radial_bins": definition.radial_bins,
            "histogram_max_angle_mrad": definition.histogram_max_angle_mrad,
            "comparison_collection_angle_mrad": (definition.comparison_collection_angle_mrad),
            "method": (
                "one parameter at a time around the baseline, plus an explicit "
                "combined reference and larger/finer confirmation"
                if combined_check is not None
                else "one parameter at a time around the baseline"
            ),
            "radial_histogram_edges_mrad": edges.tolist(),
        },
        "baseline_parameters": _as_json(definition.baseline),
        "axes": axes,
        "combined_check": combined_check,
        "cases": results,
        "summary": {
            "case_count": len(results),
            "successful_cases": sum(result["status"] == "ok" for result in results),
            "failed_cases": sum(result["status"] != "ok" for result in results),
        },
    }


def _metric(result: Mapping[str, object], branch: str, name: str) -> object:
    branch_value = result.get(branch)
    if not isinstance(branch_value, Mapping):
        return None
    metrics = branch_value.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return metrics.get(name)


def _collection_metric(result: Mapping[str, object], branch: str, name: str) -> object:
    branch_value = result.get(branch)
    if not isinstance(branch_value, Mapping):
        return None
    collection = branch_value.get("collection")
    if not isinstance(collection, Mapping):
        return None
    metrics = collection.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return metrics.get(name)


def _csv_result_row(
    axis: str,
    axis_value: object,
    result: Mapping[str, object],
    reference_id: str,
    comparison: Mapping[str, object],
) -> dict[str, object]:
    return {
        "axis": axis,
        "axis_value": json.dumps(axis_value, separators=(",", ":")),
        "case_id": result["case_id"],
        "reference_case_id": reference_id,
        "status": result["status"],
        "runtime_seconds": result.get("runtime_seconds"),
        "specimen_thickness_A": result.get("specimen_thickness_A"),
        "atom_count": result.get("atom_count"),
        "potential_gpts": json.dumps(result.get("potential_gpts")),
        "core_probability": result.get("core_probability"),
        "core_probability_per_A": result.get("core_probability_per_A"),
        "core_probability_within_collection": result.get("core_probability_within_collection"),
        "elastic_probability_within_collection": result.get(
            "elastic_probability_within_collection"
        ),
        "elastic_captured_intensity": result.get("elastic_captured_intensity"),
        "elastic_rms_radial_mrad": _metric(result, "elastic", "rms_radial_mrad"),
        "elastic_p90_radial_mrad": _metric(result, "elastic", "p90_radial_mrad"),
        "core_rms_radial_mrad": _metric(result, "core", "rms_radial_mrad"),
        "core_p90_radial_mrad": _metric(result, "core", "p90_radial_mrad"),
        "elastic_collection_rms_radial_mrad": _collection_metric(
            result, "elastic", "rms_radial_mrad"
        ),
        "core_collection_rms_radial_mrad": _collection_metric(result, "core", "rms_radial_mrad"),
        "relative_core_probability": comparison.get("relative_core_probability"),
        "relative_core_probability_per_A": comparison.get("relative_core_probability_per_A"),
        "relative_core_probability_within_collection": comparison.get(
            "relative_core_probability_within_collection"
        ),
        "relative_elastic_probability_within_collection": comparison.get(
            "relative_elastic_probability_within_collection"
        ),
        "relative_elastic_captured_intensity": comparison.get(
            "relative_elastic_captured_intensity"
        ),
        "elastic_radial_total_variation": comparison.get("elastic_radial_total_variation"),
        "core_radial_total_variation": comparison.get("core_radial_total_variation"),
        "elastic_radial_wasserstein_mrad": comparison.get("elastic_radial_wasserstein_mrad"),
        "core_radial_wasserstein_mrad": comparison.get("core_radial_wasserstein_mrad"),
        "relative_elastic_rms_radial": comparison.get("relative_elastic_rms_radial"),
        "relative_core_rms_radial": comparison.get("relative_core_rms_radial"),
        "elastic_collection_radial_total_variation": comparison.get(
            "elastic_collection_radial_total_variation"
        ),
        "core_collection_radial_total_variation": comparison.get(
            "core_collection_radial_total_variation"
        ),
        "elastic_collection_radial_wasserstein_mrad": comparison.get(
            "elastic_collection_radial_wasserstein_mrad"
        ),
        "core_collection_radial_wasserstein_mrad": comparison.get(
            "core_collection_radial_wasserstein_mrad"
        ),
        "relative_elastic_collection_rms_radial": comparison.get(
            "relative_elastic_collection_rms_radial"
        ),
        "relative_core_collection_rms_radial": comparison.get(
            "relative_core_collection_rms_radial"
        ),
        "error": result.get("error"),
    }


def _write_csv(path: Path, report: Mapping[str, object]) -> None:
    results = {str(result["case_id"]): result for result in report["cases"]}  # type: ignore[index]
    fields = [
        "axis",
        "axis_value",
        "case_id",
        "reference_case_id",
        "status",
        "runtime_seconds",
        "specimen_thickness_A",
        "atom_count",
        "potential_gpts",
        "core_probability",
        "core_probability_per_A",
        "core_probability_within_collection",
        "elastic_probability_within_collection",
        "elastic_captured_intensity",
        "elastic_rms_radial_mrad",
        "elastic_p90_radial_mrad",
        "core_rms_radial_mrad",
        "core_p90_radial_mrad",
        "elastic_collection_rms_radial_mrad",
        "core_collection_rms_radial_mrad",
        "relative_core_probability",
        "relative_core_probability_per_A",
        "relative_core_probability_within_collection",
        "relative_elastic_probability_within_collection",
        "relative_elastic_captured_intensity",
        "elastic_radial_total_variation",
        "core_radial_total_variation",
        "elastic_radial_wasserstein_mrad",
        "core_radial_wasserstein_mrad",
        "relative_elastic_rms_radial",
        "relative_core_rms_radial",
        "elastic_collection_radial_total_variation",
        "core_collection_radial_total_variation",
        "elastic_collection_radial_wasserstein_mrad",
        "core_collection_radial_wasserstein_mrad",
        "relative_elastic_collection_rms_radial",
        "relative_core_collection_rms_radial",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for axis, axis_data in report["axes"].items():  # type: ignore[union-attr]
            reference_id = axis_data["reference_case_id"]
            for point in axis_data["points"]:
                result = results[point["case_id"]]
                comparison = point["comparison_to_reference"] or {}
                writer.writerow(
                    _csv_result_row(axis, point["value"], result, reference_id, comparison)
                )
        combined_check = report.get("combined_check")
        if isinstance(combined_check, Mapping):
            reference_id = str(combined_check["reference_case_id"])
            for point in combined_check["points"]:
                result = results[point["case_id"]]
                comparison = point["comparison_to_reference"] or {}
                writer.writerow(
                    _csv_result_row(
                        "combined_check",
                        point["role"],
                        result,
                        reference_id,
                        comparison,
                    )
                )


def _plot_report(path: Path, report: Mapping[str, object]) -> None:
    import matplotlib.pyplot as plt

    results = {str(result["case_id"]): result for result in report["cases"]}  # type: ignore[index]
    thickness = report["axes"]["thickness_repetitions"]  # type: ignore[index]
    thickness_points = [
        (point, results[point["case_id"]])
        for point in thickness["points"]
        if results[point["case_id"]]["status"] == "ok"
    ]
    thickness_points.sort(key=lambda item: item[1]["specimen_thickness_A"])
    thickness_A = [item[1]["specimen_thickness_A"] for item in thickness_points]
    probabilities = [item[1]["core_probability"] for item in thickness_points]
    probability_per_A = [item[1]["core_probability_per_A"] for item in thickness_points]

    numerical_points = []
    for axis in AXIS_ORDER:
        if axis == "thickness_repetitions":
            continue
        axis_data = report["axes"][axis]  # type: ignore[index]
        for point in axis_data["points"]:
            result = results[point["case_id"]]
            comparison = point["comparison_to_reference"]
            if result["status"] != "ok" or comparison is None:
                continue
            value = point["value"]
            if axis == "lateral_repetitions":
                value_label = f"{value[0]}x{value[1]}"
            elif axis == "potential_sampling_A":
                value_label = f"{value:g} A"
            elif axis == "max_scattering_angle_mrad":
                value_label = f"{value:g} mrad"
            else:
                value_label = "double" if value else "single"
            numerical_points.append((f"{AXIS_LABELS[axis]}: {value_label}", result, comparison))

    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    axes[0, 0].plot(thickness_A, probabilities, "o-", color="#375a7f")
    axes[0, 0].set(
        xlabel="specimen thickness [A]",
        ylabel="integrated core-loss probability",
        title="Physical thickness series",
    )
    axes[0, 1].plot(thickness_A, probability_per_A, "o-", color="#00876c")
    axes[0, 1].set(
        xlabel="specimen thickness [A]",
        ylabel="core probability / A",
        title="Thickness-normalized core-loss yield",
    )

    labels = [item[0] for item in numerical_points]
    positions = np.arange(len(labels))
    relative_core = [
        (
            100.0 * item[2]["relative_core_probability"]
            if item[2]["relative_core_probability"] is not None
            else np.nan
        )
        for item in numerical_points
    ]
    elastic_wasserstein = [item[2]["elastic_radial_wasserstein_mrad"] for item in numerical_points]
    core_wasserstein = [item[2]["core_radial_wasserstein_mrad"] for item in numerical_points]
    axes[1, 0].bar(positions, relative_core, color="#bc5090")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(ylabel="difference from reference [%]", title="Core probability")
    axes[1, 1].bar(positions, elastic_wasserstein, color="#ffa600")
    axes[1, 1].set(ylabel="radial Wasserstein-1 [mrad]", title="Elastic shape")
    axes[2, 0].bar(positions, core_wasserstein, color="#58508d")
    axes[2, 0].set(ylabel="radial Wasserstein-1 [mrad]", title="Core-loss shape")

    relative_core_rms = [
        (
            100.0 * item[2]["relative_core_rms_radial"]
            if item[2]["relative_core_rms_radial"] is not None
            else np.nan
        )
        for item in numerical_points
    ]
    axes[2, 1].bar(positions, relative_core_rms, color="#2f4b7c")
    axes[2, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[2, 1].set(ylabel="difference from reference [%]", title="Core radial RMS")
    for axis in axes[1:, :].ravel():
        axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[0, :]:
        axis.grid(alpha=0.25)
    figure.suptitle("abTEM NiO specimen convergence study", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_combined_check(path: Path, report: Mapping[str, object]) -> None:
    import matplotlib.pyplot as plt

    combined = report.get("combined_check")
    if not isinstance(combined, Mapping):
        raise ValueError("The convergence report has no combined check")
    results = {str(result["case_id"]): result for result in report["cases"]}  # type: ignore[index]
    reference = results[str(combined["reference_case_id"])]
    confirmation = results[str(combined["confirmation_case_id"])]
    if reference["status"] != "ok" or confirmation["status"] != "ok":
        raise ValueError("Both combined-check cases must succeed before plotting")
    comparison = combined["confirmation_relative_to_reference"]
    if not isinstance(comparison, Mapping):
        raise ValueError("The combined-check comparison is unavailable")

    edges = np.asarray(
        report["study"]["radial_histogram_edges_mrad"], dtype=np.float64  # type: ignore[index]
    )
    collection_angle = report["study"].get(  # type: ignore[union-attr]
        "comparison_collection_angle_mrad"
    )
    if collection_angle is None:
        raise ValueError("A combined-check plot requires comparison_collection_angle_mrad")
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = np.diff(edges)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    labels = ("combined reference", "larger/finer confirmation")
    colors = ("#375a7f", "#bc5090")
    for axis, branch, title in (
        (axes[0, 0], "elastic", "Elastic radial distribution"),
        (axes[0, 1], "core", "O-K radial distribution"),
    ):
        for result, label, color in zip((reference, confirmation), labels, colors):
            histogram = np.asarray(result[branch]["radial_histogram"], dtype=np.float64)
            axis.step(
                centers,
                histogram / bin_widths,
                where="mid",
                label=label,
                color=color,
                linewidth=1.5,
            )
        axis.set(
            xlabel="radial scattering angle [mrad]",
            ylabel="probability density [1/mrad]",
            title=title,
        )
        axis.grid(alpha=0.25)
        if collection_angle is not None:
            axis.axvline(
                float(collection_angle),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="collection angle",
            )
        axis.legend(fontsize=8)

    relative_names = (
        "full core probability",
        "accepted core probability",
        "accepted elastic RMS",
        "accepted core RMS",
    )
    relative_keys = (
        "relative_core_probability",
        "relative_core_probability_within_collection",
        "relative_elastic_collection_rms_radial",
        "relative_core_collection_rms_radial",
    )
    relative_values = [
        100.0 * float(comparison[key]) if comparison.get(key) is not None else np.nan
        for key in relative_keys
    ]
    positions = np.arange(len(relative_names))
    axes[1, 0].bar(positions, relative_values, color="#00876c")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(
        ylabel="confirmation minus reference [%]",
        title="Scalar confirmation changes",
    )
    axes[1, 0].set_xticks(positions, relative_names, rotation=30, ha="right", fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    wasserstein_names = ("elastic", "O-K core loss")
    full_wasserstein = [
        comparison["elastic_radial_wasserstein_mrad"],
        comparison["core_radial_wasserstein_mrad"],
    ]
    collection_wasserstein = [
        comparison["elastic_collection_radial_wasserstein_mrad"],
        comparison["core_collection_radial_wasserstein_mrad"],
    ]
    wasserstein_positions = np.arange(2)
    bar_width = 0.36
    axes[1, 1].bar(
        wasserstein_positions - bar_width / 2.0,
        full_wasserstein,
        width=bar_width,
        color="#ffa600",
        label="full simulated range",
    )
    axes[1, 1].bar(
        wasserstein_positions + bar_width / 2.0,
        collection_wasserstein,
        width=bar_width,
        color="#58508d",
        label=f"within {float(collection_angle):g} mrad",
    )
    axes[1, 1].set(
        ylabel="radial Wasserstein-1 [mrad]",
        title="Distribution distance",
    )
    axes[1, 1].set_xticks(wasserstein_positions, wasserstein_names)
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("abTEM NiO combined-reference confirmation", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_abtem_convergence(
    config_path: Path,
    output_json: Path,
    output_csv: Path | None = None,
    output_plot: Path | None = None,
    output_combined_plot: Path | None = None,
    backend: AbTEMBackend | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    if backend is None:
        from .abtem_adapter import LiveAbTEMBackend

        backend = LiveAbTEMBackend()
    definition = load_convergence_definition(config_path)
    edges = np.linspace(
        0.0,
        definition.histogram_max_angle_mrad,
        definition.radial_bins + 1,
    )
    results: list[dict[str, object]] = []
    for index, case in enumerate(definition.cases, start=1):
        if progress is not None:
            progress(f"[{index}/{len(definition.cases)}] {case.case_id}")
        try:
            results.append(_run_case(backend, definition, case, edges))
        except Exception as error:  # preserve a long study even if one point fails
            results.append(
                {
                    "case_id": case.case_id,
                    "axis": case.axis,
                    "value": _as_json(case.value),
                    "status": "error",
                    "parameters": _as_json(case.parameters),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if progress is not None:
                progress(f"  failed: {type(error).__name__}: {error}")

    report = _assemble_report(definition, results, edges, config_path)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(_as_json(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
    if output_csv is not None:
        _write_csv(Path(output_csv), report)
    if output_plot is not None:
        _plot_report(Path(output_plot), report)
    if output_combined_plot is not None:
        _plot_combined_check(Path(output_combined_plot), report)
    return report
