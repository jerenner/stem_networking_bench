from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import SimulationConfig


def _column_schema(path: Path) -> tuple[list[str], list[np.dtype[Any]]]:
    names: list[str] = []
    dtypes: list[np.dtype[Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            if line.startswith("#column "):
                _, type_name, name = line.rstrip().split(maxsplit=2)
                names.append(name)
                dtypes.append(np.dtype(np.int32 if type_name == "int" else np.float64))
    if not names:
        raise ValueError(f"No Geant4 ntuple column declarations found in {path}")
    return names, dtypes


def read_geant4_csv(path: str | Path) -> dict[str, np.ndarray]:
    csv_path = Path(path)
    names, dtypes = _column_schema(csv_path)
    values = np.loadtxt(csv_path, delimiter=",", comments="#", ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError(
            f"{csv_path} contains {values.shape[1]} fields but declares {len(names)} columns"
        )
    return {name: values[:, index].astype(dtypes[index]) for index, name in enumerate(names)}


def read_transport_output(base_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    base = Path(base_path)
    return {
        name: read_geant4_csv(base.with_name(f"{base.name}_nt_{name}.csv"))
        for name in ("run_info", "primaries", "deposits", "events")
    }


def read_readout_calibration(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as h5:
        if h5.attrs.get("schema") not in {
            "eels-sim-readout-calibration-v1",
            "eels-sim-readout-calibration-v2",
        }:
            raise ValueError(f"Unsupported readout calibration schema in {path}")
        maps = h5["maps"]
        calibration = {
            "pedestal_adu": h5["maps/pedestal_adu"][:],
            "read_noise_adu": h5["maps/read_noise_adu"][:],
            "valid_pixel_mask": h5["maps/valid_pixel_mask"][:].astype(bool),
        }
        known_defect = (
            maps["known_defect_mask"][:].astype(bool)
            if "known_defect_mask" in maps
            else np.zeros(calibration["pedestal_adu"].shape, dtype=bool)
        )
        calibration["known_defect_mask"] = known_defect
        if "healthy_pedestal_adu" in maps:
            calibration["healthy_pedestal_adu"] = maps["healthy_pedestal_adu"][:]
            calibration["healthy_read_noise_adu"] = maps["healthy_read_noise_adu"][:]
        else:
            from .dark_calibration import repair_defect_map

            defect = known_defect | ~calibration["valid_pixel_mask"]
            calibration["healthy_pedestal_adu"] = repair_defect_map(
                calibration["pedestal_adu"], defect
            )
            calibration["healthy_read_noise_adu"] = repair_defect_map(
                calibration["read_noise_adu"], defect
            )
        calibration["signal_efficiency"] = (
            maps["signal_efficiency"][:]
            if "signal_efficiency" in maps
            else (~known_defect).astype(np.float32)
        )
        return calibration


def write_digitized_hdf5(
    path: str | Path,
    charge_electrons: np.ndarray,
    analog_adu: np.ndarray,
    raw: np.ndarray,
    config: SimulationConfig,
    input_base: str | Path,
    summary: dict[str, int | float],
    keep_intermediate: bool,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, min(raw.shape[1], 256), min(raw.shape[2], 512))
    with h5py.File(output, "w") as h5:
        h5.attrs["schema"] = "eels-sim-digitized-v1"
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["transport_input"] = str(Path(input_base))
        h5.attrs["config_toml"] = config.source_text

        frames = h5.create_group("frames")
        frames.attrs["axis_order"] = "frame,row,column"
        frames.create_dataset(
            "raw", data=raw, chunks=chunks, compression="gzip", shuffle=True
        ).attrs["units"] = "ADU"
        if keep_intermediate:
            frames.create_dataset(
                "analog_adu",
                data=analog_adu,
                chunks=chunks,
                compression="gzip",
                shuffle=True,
            ).attrs["units"] = "ADU"
            frames.create_dataset(
                "charge_electrons",
                data=charge_electrons,
                chunks=chunks,
                compression="gzip",
                shuffle=True,
            ).attrs["units"] = "electrons"

        metadata = h5.create_group("metadata")
        for section_name in (
            "experiment",
            "transport",
            "readout",
            "spectrometer",
            "framing",
        ):
            section = metadata.create_group(section_name)
            for key, value in asdict(getattr(config, section_name)).items():
                if value is not None:
                    section.attrs[key] = value
        for key, value in summary.items():
            metadata.attrs[key] = value
