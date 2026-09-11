from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

PHASE_SPACE_SCHEMA = "eels-sim-phase-space-v1"
LOSS_CHANNELS = {
    0: "zero_loss",
    1: "phonon",
    2: "plasmon",
    3: "core_loss",
    4: "other",
}
PHASE_SPACE_DTYPES = {
    "frame_id": np.dtype(np.uint64),
    "electron_id": np.dtype(np.uint64),
    "x_um": np.dtype(np.float64),
    "y_um": np.dtype(np.float64),
    "z_um": np.dtype(np.float64),
    "dir_x": np.dtype(np.float64),
    "dir_y": np.dtype(np.float64),
    "dir_z": np.dtype(np.float64),
    "kinetic_energy_eV": np.dtype(np.float64),
    "time_ns": np.dtype(np.float64),
    "weight": np.dtype(np.float64),
    "loss_channel": np.dtype(np.int32),
}
OPTIONAL_PHASE_SPACE_DTYPES = {
    "parent_electron_id": np.dtype(np.uint64),
    "branch_id": np.dtype(np.uint32),
    "source_record_id": np.dtype(np.uint64),
    "spectral_component_id": np.dtype(np.uint16),
    "plural_order": np.dtype(np.uint16),
}


def scatter_directions(
    incident: np.ndarray,
    theta_x_mrad: np.ndarray,
    theta_y_mrad: np.ndarray,
) -> np.ndarray:
    """Rotate local paraxial angular samples around incident directions."""
    incident = np.asarray(incident, dtype=np.float64)
    if incident.ndim != 2 or incident.shape[1] != 3:
        raise ValueError("incident directions must have shape [N, 3]")
    theta_x = np.asarray(theta_x_mrad, dtype=np.float64)
    theta_y = np.asarray(theta_y_mrad, dtype=np.float64)
    if theta_x.shape != (len(incident),) or theta_y.shape != (len(incident),):
        raise ValueError("angular samples must have one value per direction")
    lab_x = np.broadcast_to(np.array([1.0, 0.0, 0.0]), incident.shape)
    basis_x = lab_x - (incident * lab_x).sum(axis=1)[:, None] * incident
    basis_norm = np.linalg.norm(basis_x, axis=1)
    singular = basis_norm < 1.0e-10
    if np.any(singular):
        lab_y = np.array([0.0, 1.0, 0.0])
        basis_x[singular] = (
            lab_y - (incident[singular] * lab_y).sum(axis=1)[:, None] * incident[singular]
        )
        basis_norm[singular] = np.linalg.norm(basis_x[singular], axis=1)
    basis_x /= basis_norm[:, None]
    basis_y = np.cross(incident, basis_x)
    slopes_x = np.tan(theta_x * 1.0e-3)
    slopes_y = np.tan(theta_y * 1.0e-3)
    outgoing = incident + slopes_x[:, None] * basis_x + slopes_y[:, None] * basis_y
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    return outgoing


def validate_phase_space(electrons: Mapping[str, np.ndarray]) -> int:
    missing = set(PHASE_SPACE_DTYPES) - set(electrons)
    if missing:
        raise ValueError(f"Phase-space table is missing fields: {sorted(missing)}")
    lengths = {len(np.asarray(electrons[name])) for name in PHASE_SPACE_DTYPES}
    if len(lengths) != 1:
        raise ValueError("All phase-space fields must have the same length")
    count = lengths.pop()
    if count == 0:
        raise ValueError("Phase-space table is empty")

    floating_fields = (
        "x_um",
        "y_um",
        "z_um",
        "dir_x",
        "dir_y",
        "dir_z",
        "kinetic_energy_eV",
        "time_ns",
        "weight",
    )
    if any(not np.all(np.isfinite(electrons[name])) for name in floating_fields):
        raise ValueError("Phase-space floating-point fields must be finite")
    direction_norm = np.sqrt(
        np.asarray(electrons["dir_x"]) ** 2
        + np.asarray(electrons["dir_y"]) ** 2
        + np.asarray(electrons["dir_z"]) ** 2
    )
    if not np.allclose(direction_norm, 1.0, rtol=1.0e-7, atol=1.0e-9):
        raise ValueError("Phase-space direction vectors must be normalized")
    if np.any(np.asarray(electrons["kinetic_energy_eV"]) <= 0.0):
        raise ValueError("Kinetic energies must be positive")
    if np.any(np.asarray(electrons["time_ns"]) < 0.0):
        raise ValueError("Times must be nonnegative within each frame")
    if np.any(np.asarray(electrons["weight"]) <= 0.0):
        raise ValueError("Statistical weights must be positive")
    electron_ids = np.asarray(electrons["electron_id"])
    if len(np.unique(electron_ids)) != count:
        raise ValueError("electron_id must be globally unique")
    channels = np.asarray(electrons["loss_channel"])
    if np.any(~np.isin(channels, list(LOSS_CHANNELS))):
        raise ValueError(f"loss_channel must be one of {sorted(LOSS_CHANNELS)}")
    for name in set(electrons) & set(OPTIONAL_PHASE_SPACE_DTYPES):
        if len(np.asarray(electrons[name])) != count:
            raise ValueError(f"Optional phase-space field {name} has the wrong length")
    return count


def _typed_phase_space(
    electrons: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    validate_phase_space(electrons)
    typed = {
        name: np.asarray(electrons[name], dtype=dtype) for name, dtype in PHASE_SPACE_DTYPES.items()
    }
    typed.update(
        {
            name: np.asarray(electrons[name], dtype=dtype)
            for name, dtype in OPTIONAL_PHASE_SPACE_DTYPES.items()
            if name in electrons
        }
    )
    return typed


def write_phase_space_hdf5(
    path: str | Path,
    electrons: Mapping[str, np.ndarray],
    metadata: Mapping[str, object] | None = None,
) -> None:
    typed = _typed_phase_space(electrons)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = PHASE_SPACE_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["loss_channel_json"] = json.dumps(LOSS_CHANNELS, sort_keys=True)
        if metadata is not None:
            h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            if "weight_semantics" in metadata:
                h5.attrs["weight_semantics"] = str(metadata["weight_semantics"])
        group = h5.create_group("electrons")
        group.attrs["coordinate_system"] = "right-handed; beam travels in +z"
        group.attrs["position_units"] = "um"
        group.attrs["energy_units"] = "eV"
        group.attrs["time_units"] = "ns"
        for name, values in typed.items():
            group.create_dataset(name, data=values, compression="gzip", shuffle=True)


def read_phase_space_hdf5(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as h5:
        if h5.attrs.get("schema") != PHASE_SPACE_SCHEMA:
            raise ValueError(f"Unsupported phase-space schema in {path}")
        group = h5["electrons"]
        electrons = {name: group[name][:] for name in PHASE_SPACE_DTYPES}
        electrons.update(
            {name: group[name][:] for name in OPTIONAL_PHASE_SPACE_DTYPES if name in group}
        )
    validate_phase_space(electrons)
    return electrons


def read_phase_space_metadata(path: str | Path) -> dict[str, object]:
    with h5py.File(path) as h5:
        if h5.attrs.get("schema") != PHASE_SPACE_SCHEMA:
            raise ValueError(f"Unsupported phase-space schema in {path}")
        encoded = h5.attrs.get("metadata_json")
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        result = {} if encoded is None else json.loads(str(encoded))
        if "weight_semantics" in h5.attrs:
            result["weight_semantics"] = str(h5.attrs["weight_semantics"])
        return result


def generate_diagnostic_phase_space(
    electron_count: int,
    frame_count: int,
    beam_energy_eV: float,
    losses_eV: np.ndarray,
    fractions: np.ndarray,
    position_sigma_um: float,
    angular_sigma_mrad: float,
    integration_time_ns: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Generate a diagnostic mixture, not a specimen-scattering model."""
    if electron_count <= 0 or frame_count <= 0:
        raise ValueError("electron_count and frame_count must be positive")
    losses = np.asarray(losses_eV, dtype=np.float64)
    probabilities = np.asarray(fractions, dtype=np.float64)
    if losses.ndim != 1 or len(losses) == 0 or len(losses) != len(probabilities):
        raise ValueError("losses and fractions must be nonempty lists of equal length")
    if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError("fractions must be nonnegative with a positive sum")
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(random_seed)
    sampled_losses = rng.choice(losses, size=electron_count, p=probabilities)
    kinetic_energy = beam_energy_eV - sampled_losses
    if np.any(kinetic_energy <= 0.0):
        raise ValueError("A requested loss is greater than the beam energy")
    slopes = rng.normal(0.0, angular_sigma_mrad * 1.0e-3, size=(electron_count, 2))
    directions = np.column_stack((slopes[:, 0], slopes[:, 1], np.ones(electron_count)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    channels = np.where(np.isclose(sampled_losses, 0.0), 0, 4).astype(np.int32)
    return _typed_phase_space(
        {
            "frame_id": np.arange(electron_count, dtype=np.uint64) * frame_count // electron_count,
            "electron_id": np.arange(electron_count, dtype=np.uint64),
            "x_um": rng.normal(0.0, position_sigma_um, electron_count),
            "y_um": rng.normal(0.0, position_sigma_um, electron_count),
            "z_um": np.zeros(electron_count),
            "dir_x": directions[:, 0],
            "dir_y": directions[:, 1],
            "dir_z": directions[:, 2],
            "kinetic_energy_eV": kinetic_energy,
            "time_ns": rng.uniform(0.0, integration_time_ns, electron_count),
            "weight": np.ones(electron_count),
            "loss_channel": channels,
        }
    )
