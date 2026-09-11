from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import TransportConfig
from .io import read_transport_output
from .transport import geometry_from_run_info, transport_deposits


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(np.max(values)),
    }


def _cluster_sigma(charge: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(charge.shape[1:])
    sigmas = []
    for frame in charge:
        total = float(frame.sum())
        if total <= 0.0:
            sigmas.append(0.0)
            continue
        mean_row = float((frame * rows).sum() / total)
        mean_column = float((frame * columns).sum() / total)
        variance_row = float((frame * (rows - mean_row) ** 2).sum() / total)
        variance_column = float((frame * (columns - mean_column) ** 2).sum() / total)
        sigmas.append(np.sqrt(0.5 * (variance_row + variance_column)))
    return np.asarray(sigmas)


def summarize_legacy(
    path: str | Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    import pandas as pd

    frame = pd.read_pickle(path)
    event_count = int(frame["event"].max()) + 1
    total_pairs = (
        frame.groupby("event")["counts"].sum().reindex(range(event_count), fill_value=0).to_numpy()
    )
    sigmas = []
    for _, event in frame.groupby("event"):
        weights = event["counts"].to_numpy(dtype=float)
        rows = event["row"].to_numpy(dtype=float)
        columns = event["col"].to_numpy(dtype=float)
        mean_row = np.average(rows, weights=weights)
        mean_column = np.average(columns, weights=weights)
        variance = 0.5 * (
            np.average((rows - mean_row) ** 2, weights=weights)
            + np.average((columns - mean_column) ** 2, weights=weights)
        )
        sigmas.append(np.sqrt(variance))
    sigma_by_event = np.zeros(event_count)
    sigma_by_event[np.sort(frame["event"].unique())] = sigmas
    summary = {
        "events": event_count,
        "nonzero_events": int(np.count_nonzero(total_pairs)),
        "total_pairs": _distribution(total_pairs),
        "equivalent_deposited_energy_eV": _distribution(total_pairs * 3.6),
        "cluster_sigma_pixels": _distribution(sigma_by_event),
    }
    return summary, total_pairs, sigma_by_event


def summarize_new(
    base_path: str | Path,
    transport_config: TransportConfig,
    random_seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    tables = read_transport_output(base_path)
    geometry = geometry_from_run_info(tables["run_info"])
    event_energy = tables["events"]["total_edep_eV"]
    expected_pairs = event_energy / transport_config.pair_creation_energy_eV
    charge, _ = transport_deposits(
        tables["deposits"],
        geometry,
        transport_config,
        primaries_per_frame=1,
        random_seed=random_seed,
    )
    sigma = _cluster_sigma(charge)
    summary = {
        "events": int(len(event_energy)),
        "nonzero_events": int(np.count_nonzero(event_energy)),
        "total_pairs": _distribution(expected_pairs),
        "deposited_energy_eV": _distribution(event_energy),
        "cluster_sigma_pixels": _distribution(sigma),
    }
    return summary, expected_pairs, sigma


def compare_response_libraries(
    legacy_100: str | Path,
    legacy_300: str | Path,
    new_100: str | Path,
    new_300: str | Path,
    output_json: str | Path,
    transport_config: TransportConfig,
    random_seed: int = 12345,
) -> dict[str, object]:
    legacy100, legacy100_pairs, legacy100_sigma = summarize_legacy(legacy_100)
    legacy300, legacy300_pairs, legacy300_sigma = summarize_legacy(legacy_300)
    new100, new100_pairs, new100_sigma = summarize_new(new_100, transport_config, random_seed)
    new300, new300_pairs, new300_sigma = summarize_new(new_300, transport_config, random_seed + 1)
    result = {
        "legacy_100keV": legacy100,
        "legacy_300keV": legacy300,
        "new_100keV": new100,
        "new_300keV": new300,
        "median_pair_ratio_new_to_legacy": {
            "100keV": new100["total_pairs"]["median"] / legacy100["total_pairs"]["median"],
            "300keV": new300["total_pairs"]["median"] / legacy300["total_pairs"]["median"],
        },
        "notes": [
            "Legacy counts use 3.6 eV per pair; the new model uses the configured 3.64 eV.",
            "The legacy and new geometries are not yet identical, so ratios are diagnostics rather than fit results.",
        ],
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, (energy, old_pairs, new_pairs, old_sigma, new_sigma) in enumerate(
        (
            (100, legacy100_pairs, new100_pairs, legacy100_sigma, new100_sigma),
            (300, legacy300_pairs, new300_pairs, legacy300_sigma, new300_sigma),
        )
    ):
        axes[0, column].hist(old_pairs, bins=30, alpha=0.6, density=True, label="legacy")
        axes[0, column].hist(new_pairs, bins=30, alpha=0.6, density=True, label="new")
        axes[0, column].set_title(f"{energy} keV total charge")
        axes[0, column].set_xlabel("e-h pairs")
        axes[0, column].set_ylabel("density")
        axes[0, column].legend()
        axes[1, column].hist(old_sigma, bins=25, alpha=0.6, density=True, label="legacy")
        axes[1, column].hist(new_sigma, bins=25, alpha=0.6, density=True, label="new")
        axes[1, column].set_title(f"{energy} keV cluster width")
        axes[1, column].set_xlabel("RMS pixels")
        axes[1, column].set_ylabel("density")
        axes[1, column].legend()
    figure.savefig(output_json.with_suffix(".png"), dpi=160)
    plt.close(figure)
    return result
