from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .config import ScanConfig, load_config
from .lmto_scan import ELEMENTS, build_disordered_rocksalt
from .spectral_library import read_spectral_library

ABTEM_SPATIAL_SCHEMA = "eels-sim-abtem-spatial-v1"

EDGE_SHELLS: dict[str, tuple[int, int, int]] = {
    "Ti_M23": (22, 3, 1),
    "Mn_M23": (25, 3, 1),
    "Li_K": (3, 1, 0),
    "Ti_L23": (22, 2, 1),
    "O_K": (8, 1, 0),
    "Mn_L23": (25, 2, 1),
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TOML table")
    return value


def _positive_float(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = float(parameters.get(name, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"scan.abtem.{name} must be finite and positive")
    return value


def _nonnegative_float(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = float(parameters.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"scan.abtem.{name} must be finite and nonnegative")
    return value


def _measurement_image(measurement: object, scan_pixels: int, name: str) -> np.ndarray:
    if hasattr(measurement, "compute"):
        computed = measurement.compute()
        if computed is not None:
            measurement = computed
    array = np.asarray(getattr(measurement, "array"), dtype=np.float64)
    if array.shape[-2:] != (scan_pixels, scan_pixels):
        raise ValueError(
            f"abTEM {name} returned shape {array.shape}, expected scan axes "
            f"{(scan_pixels, scan_pixels)}"
        )
    if array.ndim > 2:
        array = array.sum(axis=tuple(range(array.ndim - 2)))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"abTEM {name} contains non-finite values")
    if np.any(array < -1.0e-10):
        raise ValueError(f"abTEM {name} contains negative intensity")
    return np.maximum(array, 0.0)


def _scan_parameters(scan: ScanConfig) -> dict[str, object]:
    parameters = dict(_mapping(scan.abtem, "scan.abtem"))
    allowed = {
        "potential_sampling_A",
        "slice_thickness_A",
        "potential_parametrization",
        "potential_projection",
        "probe_semiangle_mrad",
        "haadf_inner_mrad",
        "haadf_outer_mrad",
        "eels_inner_mrad",
        "eels_outer_mrad",
        "device",
        "epsilon_eV",
        "transition_order",
        "double_channel_edges",
        "frozen_phonon_configs",
        "thermal_sigma_A",
        "defocus_sign",
        "focal_depths_A",
        "maximum_batch",
    }
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"Unknown scan.abtem parameter(s): {sorted(unknown)}")
    return parameters


def _transition_support(edge_names: tuple[str, ...]) -> dict[str, dict[str, object]]:
    support: dict[str, dict[str, object]] = {}
    for name in edge_names:
        shell = EDGE_SHELLS.get(name)
        if shell is None:
            support[name] = {
                "status": "unsupported",
                "reason": "no abTEM subshell mapping is defined",
            }
        else:
            atomic_number, n, angular_momentum = shell
            support[name] = {
                "status": "mapped",
                "atomic_number": atomic_number,
                "n": n,
                "l": angular_momentum,
            }
    return support


def generate_lmto_abtem_spatial_response(
    config_path: str | Path,
    output_hdf5: str | Path,
) -> dict[str, object]:
    """Run through-focus abTEM HAADF and core-loss scans for synthetic LMTO."""
    simulation_config = load_config(config_path)
    scan = simulation_config.scan
    parameters = _scan_parameters(scan)
    library_file = simulation_config.specimen.parameters.get("library_file")
    if library_file is None:
        raise ValueError("LMTO abTEM scan requires specimen.parameters.library_file")
    library = read_spectral_library(str(library_file))
    if library.metadata.get("formula") != "Li1.2Mn0.4Ti0.4O2":
        raise ValueError("LMTO abTEM scan requires the LMTO spectral library")
    edge_names = tuple(component.name for component in library.core_loss)
    support = _transition_support(edge_names)
    unsupported = [name for name, status in support.items() if status["status"] == "unsupported"]
    if unsupported:
        raise ValueError(f"No abTEM transition mapping for edge(s): {unsupported}")

    try:
        import abtem
        from abtem.inelastic.core_loss import SubshellTransitions
        from ase import Atoms
    except ImportError as error:
        raise ImportError(
            "LMTO spatial scans require abTEM, ASE, and the GPAW core-loss environment"
        ) from error

    rng = np.random.default_rng(scan.random_seed)
    atoms_table, _, extent_A = build_disordered_rocksalt(scan, rng)
    symbols = [ELEMENTS[index] for index in atoms_table["element_id"]]
    sample_thickness_A = 0.5 * scan.lattice_constant_A * scan.depth_atomic_planes
    positions = np.column_stack((atoms_table["x_A"], atoms_table["y_A"], atoms_table["z_A"]))
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=(extent_A, extent_A, sample_thickness_A),
        pbc=True,
    )

    frozen_phonon_configs = int(parameters.get("frozen_phonon_configs", 1))
    if frozen_phonon_configs <= 0:
        raise ValueError("scan.abtem.frozen_phonon_configs must be positive")
    thermal_sigma_A = _nonnegative_float(parameters, "thermal_sigma_A", 0.0)
    potential_atoms: object = atoms
    if frozen_phonon_configs > 1 or thermal_sigma_A > 0.0:
        if thermal_sigma_A == 0.0:
            raise ValueError("scan.abtem.thermal_sigma_A must be positive for frozen phonons")
        potential_atoms = abtem.FrozenPhonons(
            atoms,
            num_configs=frozen_phonon_configs,
            sigmas=thermal_sigma_A,
            ensemble_mean=True,
            seed=scan.random_seed + 1,
        )

    potential_sampling_A = _positive_float(parameters, "potential_sampling_A", 0.12)
    slice_thickness_A = _positive_float(
        parameters, "slice_thickness_A", 0.5 * scan.lattice_constant_A
    )
    probe_semiangle_mrad = _positive_float(parameters, "probe_semiangle_mrad", 20.0)
    haadf_inner_mrad = _nonnegative_float(parameters, "haadf_inner_mrad", 50.0)
    haadf_outer_mrad = _positive_float(parameters, "haadf_outer_mrad", 80.0)
    eels_inner_mrad = _nonnegative_float(parameters, "eels_inner_mrad", 0.0)
    eels_outer_mrad = _positive_float(parameters, "eels_outer_mrad", 50.0)
    if haadf_outer_mrad <= haadf_inner_mrad:
        raise ValueError("HAADF outer angle must exceed its inner angle")
    if eels_outer_mrad <= eels_inner_mrad:
        raise ValueError("EELS outer angle must exceed its inner angle")
    epsilon_eV = _positive_float(parameters, "epsilon_eV", 10.0)
    transition_order = int(parameters.get("transition_order", 1))
    if transition_order <= 0:
        raise ValueError("scan.abtem.transition_order must be positive")
    configured_double = parameters.get("double_channel_edges", [])
    if not isinstance(configured_double, list) or not all(
        isinstance(value, str) for value in configured_double
    ):
        raise TypeError("scan.abtem.double_channel_edges must be an array of names")
    unknown_double = set(configured_double) - set(edge_names)
    if unknown_double:
        raise ValueError(f"Unknown double-channel edge name(s): {sorted(unknown_double)}")
    double_channel_edges = set(configured_double)
    device = str(parameters.get("device", "cpu"))
    maximum_batch = parameters.get("maximum_batch", "auto")
    if not (maximum_batch == "auto" or isinstance(maximum_batch, int) and maximum_batch > 0):
        raise ValueError("scan.abtem.maximum_batch must be 'auto' or positive")

    potential = abtem.Potential(
        potential_atoms,
        sampling=potential_sampling_A,
        slice_thickness=slice_thickness_A,
        parametrization=str(parameters.get("potential_parametrization", "kirkland")),
        projection=str(parameters.get("potential_projection", "finite")),
        device=device,
    )
    scan_grid = abtem.GridScan(
        start=(0.0, 0.0),
        end=potential.extent,
        gpts=(scan.scan_pixels, scan.scan_pixels),
        endpoint=False,
    )
    haadf_detector = abtem.AnnularDetector(
        inner=haadf_inner_mrad, outer=haadf_outer_mrad, to_cpu=True
    )
    eels_detector = abtem.AnnularDetector(inner=eels_inner_mrad, outer=eels_outer_mrad, to_cpu=True)
    focal_values = parameters.get("focal_depths_A")
    if focal_values is None:
        focal_depth_A = np.linspace(0.0, sample_thickness_A, scan.depth_sections)
    else:
        focal_depth_A = np.asarray(focal_values, dtype=np.float64)
        if focal_depth_A.shape != (scan.depth_sections,):
            raise ValueError("scan.abtem.focal_depths_A must have scan.depth_sections entries")
        if not np.all(np.isfinite(focal_depth_A)):
            raise ValueError("scan.abtem.focal_depths_A must be finite")
    defocus_sign = float(parameters.get("defocus_sign", 1.0))
    if defocus_sign not in (-1.0, 1.0):
        raise ValueError("scan.abtem.defocus_sign must be -1 or 1")
    defocus_A = defocus_sign * focal_depth_A

    transition_potentials: dict[str, object] = {}
    for name in edge_names:
        atomic_number, n, angular_momentum = EDGE_SHELLS[name]
        transitions = SubshellTransitions(
            atomic_number,
            n,
            angular_momentum,
            order=transition_order,
            epsilon=epsilon_eV,
            xc="PBE",
        )
        double_channel = name in double_channel_edges
        try:
            transition_potentials[name] = transitions.get_transition_potentials(
                extent=potential.extent,
                gpts=potential.gpts,
                energy=simulation_config.experiment.beam_energy_keV * 1.0e3,
                double_channel=double_channel,
            )
        except Exception as error:
            support[name]["status"] = "failed"
            support[name]["reason"] = f"{type(error).__name__}: {error}"
            raise RuntimeError(
                f"Failed to construct abTEM transition potential for {name}"
            ) from error
        support[name].update(
            {
                "status": "constructed",
                "transition_count": len(transition_potentials[name]),
                "epsilon_eV": epsilon_eV,
                "double_channel": double_channel,
            }
        )

    haadf_fraction = np.zeros(
        (scan.depth_sections, scan.scan_pixels, scan.scan_pixels), dtype=np.float32
    )
    raw_edge_intensity = np.zeros(
        (
            scan.depth_sections,
            len(edge_names),
            scan.scan_pixels,
            scan.scan_pixels,
        ),
        dtype=np.float32,
    )
    beam_energy_eV = simulation_config.experiment.beam_energy_keV * 1.0e3
    for depth_index, defocus in enumerate(defocus_A):
        probe = abtem.Probe(
            energy=beam_energy_eV,
            semiangle_cutoff=probe_semiangle_mrad,
            defocus=float(defocus),
            device=device,
        )
        haadf_measurement = probe.scan(
            potential,
            scan=scan_grid,
            detectors=haadf_detector,
            max_batch=maximum_batch,
            lazy=False,
        )
        haadf_fraction[depth_index] = _measurement_image(
            haadf_measurement, scan.scan_pixels, "HAADF"
        )
        for edge_index, name in enumerate(edge_names):
            double_channel = name in double_channel_edges
            try:
                measurement = probe.transition_potential_scan(
                    potential=potential,
                    transition_potentials=transition_potentials[name],
                    scan=scan_grid,
                    detectors=eels_detector,
                    sites=atoms,
                    max_batch=maximum_batch,
                    lazy=False,
                    double_channel=double_channel,
                )
                raw_edge_intensity[depth_index, edge_index] = _measurement_image(
                    measurement, scan.scan_pixels, name
                )
                support[name]["status"] = "calculated"
            except Exception as error:
                support[name]["status"] = "failed"
                support[name]["reason"] = f"{type(error).__name__}: {error}"
                raise RuntimeError(f"abTEM transition scan failed for {name}") from error

    area_A2 = extent_A**2
    element_ids = atoms_table["element_id"]
    edge_response_per_A2 = np.zeros_like(raw_edge_intensity, dtype=np.float64)
    normalization: dict[str, object] = {}
    for edge_index, component in enumerate(library.core_loss):
        atom_count = int(np.count_nonzero(element_ids == ELEMENTS.index(component.element)))
        areal_density = atom_count / area_A2
        means = raw_edge_intensity[:, edge_index].mean(axis=(-2, -1))
        if np.any(means <= 0.0):
            raise RuntimeError(f"abTEM returned zero mean intensity for {component.name}")
        edge_response_per_A2[:, edge_index] = (
            raw_edge_intensity[:, edge_index] * (areal_density / means)[:, None, None]
        )
        normalization[component.name] = {
            "element": component.element,
            "atom_count": atom_count,
            "target_mean_areal_density_per_A2": areal_density,
            "raw_mean_by_depth": means.tolist(),
            "method": "abTEM relative contrast normalized to GOSH atomic cross section",
        }

    metadata = {
        "engine": "abTEM",
        "abtem_version": str(abtem.__version__),
        "model": "through-focus disordered-rocksalt LMTO",
        "beam_energy_eV": beam_energy_eV,
        "potential_sampling_A": list(float(value) for value in potential.sampling),
        "potential_gpts": list(int(value) for value in potential.gpts),
        "potential_slices": int(potential.num_slices),
        "probe_semiangle_mrad": probe_semiangle_mrad,
        "haadf_detector_mrad": [haadf_inner_mrad, haadf_outer_mrad],
        "eels_detector_mrad": [eels_inner_mrad, eels_outer_mrad],
        "frozen_phonon_configs": frozen_phonon_configs,
        "thermal_sigma_A": thermal_sigma_A,
        "defocus_A": defocus_A.tolist(),
        "defocus_interpretation": (
            "configured through-focus series; focal_depth_A is a nominal label"
        ),
        "transition_support": support,
        "normalization": normalization,
        "limitations": [
            "synthetic unrelaxed random-cation LMTO structure",
            "GOSH supplies absolute edge normalization and energy dependence",
            "abTEM transition potentials supply relative spatial/channeling contrast",
            "no material-specific ELNES or charge-state dependence",
            "through-focus sections are separate acquisitions, not depth inversion",
        ],
    }
    summary: dict[str, object] = {
        "scan_shape": [scan.depth_sections, scan.scan_pixels, scan.scan_pixels],
        "edge_names": list(edge_names),
        "field_of_view_A": extent_A,
        "sample_thickness_A": sample_thickness_A,
        "atom_count": len(atoms),
        "double_channel_edges": sorted(double_channel_edges),
        "transition_status": {name: status["status"] for name, status in support.items()},
    }
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = ABTEM_SPATIAL_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["config_file"] = str(Path(config_path).resolve())
        h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        h5.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5.attrs["edge_names_json"] = json.dumps(edge_names)
        h5.attrs["element_labels_json"] = json.dumps(ELEMENTS)
        axes = h5.create_group("axes")
        axes.create_dataset("focal_depth_A", data=focal_depth_A)
        axes.create_dataset("defocus_A", data=defocus_A)
        axes.create_dataset(
            "scan_x_A",
            data=np.linspace(0.0, extent_A, scan.scan_pixels, endpoint=False),
        )
        axes.create_dataset(
            "scan_y_A",
            data=np.linspace(0.0, extent_A, scan.scan_pixels, endpoint=False),
        )
        structure = h5.create_group("structure/atoms")
        for name, values in atoms_table.items():
            structure.create_dataset(name, data=values, compression="gzip", shuffle=True)
        spatial = h5.create_group("spatial")
        spatial.create_dataset(
            "haadf_fraction",
            data=haadf_fraction,
            compression="gzip",
            shuffle=True,
        )
        spatial.create_dataset(
            "raw_edge_intensity",
            data=raw_edge_intensity,
            compression="gzip",
            shuffle=True,
        )
        spatial.create_dataset(
            "edge_response_per_A2",
            data=edge_response_per_A2.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
    return summary


def read_lmto_abtem_spatial_response(path: str | Path) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != ABTEM_SPATIAL_SCHEMA:
            raise ValueError(f"Unsupported abTEM spatial schema in {path}")
        atoms = {name: dataset[:] for name, dataset in h5["structure/atoms"].items()}
        return {
            "edge_names": tuple(json.loads(h5.attrs["edge_names_json"])),
            "element_labels": tuple(json.loads(h5.attrs["element_labels_json"])),
            "focal_depth_A": h5["axes/focal_depth_A"][:],
            "scan_x_A": h5["axes/scan_x_A"][:],
            "scan_y_A": h5["axes/scan_y_A"][:],
            "haadf_fraction": h5["spatial/haadf_fraction"][:],
            "raw_edge_intensity": h5["spatial/raw_edge_intensity"][:],
            "edge_response_per_A2": h5["spatial/edge_response_per_A2"][:],
            "atoms": atoms,
            "metadata": json.loads(h5.attrs["metadata_json"]),
            "summary": json.loads(h5.attrs["summary_json"]),
        }


def tile_lmto_abtem_spatial_response(
    source_hdf5: str | Path,
    output_hdf5: str | Path,
    tile_factor: int,
) -> dict[str, object]:
    """Expand a square cached abTEM response as a periodic supercell.

    This is deliberately a lossless replication of the calculated response.  It
    gives a larger scan field at the original probe sampling, but does not claim
    to introduce new cation disorder or independent frozen-phonon realizations.
    """
    if tile_factor <= 0:
        raise ValueError("tile factor must be positive")
    source_path = Path(source_hdf5)
    output_path = Path(output_hdf5)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("tiled response output must differ from its source")

    response = read_lmto_abtem_spatial_response(source_path)
    source_x = np.asarray(response["scan_x_A"], dtype=np.float64)
    source_y = np.asarray(response["scan_y_A"], dtype=np.float64)
    if len(source_x) < 2 or len(source_y) < 2:
        raise ValueError("spatial response needs at least two scan points per axis")
    step_x_A = float(np.median(np.diff(source_x)))
    step_y_A = float(np.median(np.diff(source_y)))
    if step_x_A <= 0.0 or step_y_A <= 0.0:
        raise ValueError("spatial response scan axes must be increasing")
    source_extent_A = float(response["summary"]["field_of_view_A"])
    source_shape = np.asarray(response["haadf_fraction"]).shape
    if source_shape[-2:] != (len(source_y), len(source_x)):
        raise ValueError("spatial response axes do not match its map shape")
    if not np.isclose(source_extent_A, len(source_x) * step_x_A):
        raise ValueError("source x axis is inconsistent with its field of view")
    if not np.isclose(source_extent_A, len(source_y) * step_y_A):
        raise ValueError("source y axis is inconsistent with its field of view")

    tiled_haadf = np.tile(np.asarray(response["haadf_fraction"]), (1, tile_factor, tile_factor))
    tiled_raw_edges = np.tile(
        np.asarray(response["raw_edge_intensity"]),
        (1, 1, tile_factor, tile_factor),
    )
    tiled_edge_response = np.tile(
        np.asarray(response["edge_response_per_A2"]),
        (1, 1, tile_factor, tile_factor),
    )

    atoms = response["atoms"]
    tiled_atoms: dict[str, np.ndarray] = {}
    for name, values in atoms.items():
        copies: list[np.ndarray] = []
        for tile_y in range(tile_factor):
            for tile_x in range(tile_factor):
                copy = np.asarray(values).copy()
                if name == "x_A":
                    copy = copy + tile_x * source_extent_A
                elif name == "y_A":
                    copy = copy + tile_y * source_extent_A
                copies.append(copy)
        tiled_atoms[name] = np.concatenate(copies)

    metadata = dict(response["metadata"])
    limitations = list(metadata.get("limitations", []))
    limitation = (
        f"expanded field is a periodic {tile_factor}x{tile_factor} replication "
        "of the source abTEM supercell; it adds no independent cation disorder"
    )
    if limitation not in limitations:
        limitations.append(limitation)
    metadata["limitations"] = limitations
    metadata["periodic_supercell"] = {
        "tile_factor": tile_factor,
        "source_file": str(source_path.resolve()),
        "source_field_of_view_A": source_extent_A,
        "source_scan_shape": list(source_shape),
        "preserves_original_probe_sampling": True,
    }
    normalization = metadata.get("normalization")
    if isinstance(normalization, dict):
        for edge_metadata in normalization.values():
            if isinstance(edge_metadata, dict) and "atom_count" in edge_metadata:
                edge_metadata["atom_count"] = int(edge_metadata["atom_count"]) * (tile_factor**2)

    summary = dict(response["summary"])
    summary.update(
        {
            "scan_shape": list(tiled_haadf.shape),
            "field_of_view_A": source_extent_A * tile_factor,
            "atom_count": len(next(iter(tiled_atoms.values()))),
            "periodic_tile_factor": tile_factor,
            "tiled_from": str(source_path.resolve()),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as target:
        for name, value in source.attrs.items():
            target.attrs[name] = value
        target.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        target.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        target.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        target.attrs["tiled_from"] = str(source_path.resolve())
        target.attrs["periodic_tile_factor"] = tile_factor

        axes = target.create_group("axes")
        axes.create_dataset("focal_depth_A", data=response["focal_depth_A"])
        if "axes/defocus_A" in source:
            axes.create_dataset("defocus_A", data=source["axes/defocus_A"][:])
        axes.create_dataset("scan_x_A", data=np.arange(tiled_haadf.shape[-1]) * step_x_A)
        axes.create_dataset("scan_y_A", data=np.arange(tiled_haadf.shape[-2]) * step_y_A)
        structure = target.create_group("structure/atoms")
        for name, values in tiled_atoms.items():
            structure.create_dataset(name, data=values, compression="gzip", shuffle=True)
        spatial = target.create_group("spatial")
        for name, values in (
            ("haadf_fraction", tiled_haadf),
            ("raw_edge_intensity", tiled_raw_edges),
            ("edge_response_per_A2", tiled_edge_response),
        ):
            spatial.create_dataset(name, data=values, compression="gzip", shuffle=True)
    return summary


def compare_lmto_abtem_spatial_responses(
    reference_path: str | Path,
    confirmation_path: str | Path,
    output_json: str | Path,
) -> dict[str, object]:
    """Compare relative spatial contrast at the nearest shared focal depth."""
    reference = read_lmto_abtem_spatial_response(reference_path)
    confirmation = read_lmto_abtem_spatial_response(confirmation_path)
    if reference["edge_names"] != confirmation["edge_names"]:
        raise ValueError("abTEM spatial responses have different edge orders")
    reference_shape = np.asarray(reference["haadf_fraction"]).shape[-2:]
    confirmation_shape = np.asarray(confirmation["haadf_fraction"]).shape[-2:]
    if reference_shape != confirmation_shape:
        raise ValueError("abTEM spatial responses have different scan grids")
    reference_depth = np.asarray(reference["focal_depth_A"], dtype=np.float64)
    confirmation_depth = np.asarray(confirmation["focal_depth_A"], dtype=np.float64)
    distance = np.abs(reference_depth[:, None] - confirmation_depth[None, :])
    reference_index, confirmation_index = np.unravel_index(int(np.argmin(distance)), distance.shape)

    def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.mean() <= 0.0 or right.mean() <= 0.0:
            raise ValueError("Cannot compare a zero-mean spatial response")
        left_normalized = left / left.mean()
        right_normalized = right / right.mean()
        left_flat = left_normalized.ravel()
        right_flat = right_normalized.ravel()
        correlation = (
            1.0
            if left_flat.std() == 0.0 and right_flat.std() == 0.0
            else float(np.corrcoef(left_flat, right_flat)[0, 1])
        )
        return {
            "relative_rms_difference": float(
                np.sqrt(np.mean((right_normalized - left_normalized) ** 2))
            ),
            "pearson_correlation": correlation,
            "reference_contrast_cv": float(left.std() / left.mean()),
            "confirmation_contrast_cv": float(right.std() / right.mean()),
            "reference_mean_raw": float(left.mean()),
            "confirmation_mean_raw": float(right.mean()),
        }

    comparison: dict[str, object] = {
        "schema": "eels-sim-abtem-spatial-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file": str(Path(reference_path).resolve()),
        "confirmation_file": str(Path(confirmation_path).resolve()),
        "reference_depth_index": int(reference_index),
        "confirmation_depth_index": int(confirmation_index),
        "reference_focal_depth_A": float(reference_depth[reference_index]),
        "confirmation_focal_depth_A": float(confirmation_depth[confirmation_index]),
        "focal_depth_difference_A": float(distance[reference_index, confirmation_index]),
        "haadf": metrics(
            np.asarray(reference["haadf_fraction"])[reference_index],
            np.asarray(confirmation["haadf_fraction"])[confirmation_index],
        ),
        "edges": {},
    }
    reference_edges = np.asarray(reference["raw_edge_intensity"])
    confirmation_edges = np.asarray(confirmation["raw_edge_intensity"])
    for edge_index, name in enumerate(reference["edge_names"]):
        comparison["edges"][name] = metrics(
            reference_edges[reference_index, edge_index],
            confirmation_edges[confirmation_index, edge_index],
        )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return comparison


def refine_lmto_abtem_haadf(
    config_path: str | Path,
    source_hdf5: str | Path,
    output_hdf5: str | Path,
    potential_sampling_A: float,
) -> dict[str, object]:
    """Recalculate only HAADF on a finer potential grid and retain edge maps."""
    if Path(source_hdf5).resolve() == Path(output_hdf5).resolve():
        raise ValueError("HAADF refinement output must differ from its source cache")
    if not np.isfinite(potential_sampling_A) or potential_sampling_A <= 0.0:
        raise ValueError("HAADF refinement sampling must be finite and positive")
    simulation_config = load_config(config_path)
    scan = simulation_config.scan
    parameters = _scan_parameters(scan)
    response = read_lmto_abtem_spatial_response(source_hdf5)
    try:
        import abtem
        from ase import Atoms
    except ImportError as error:
        raise ImportError("HAADF refinement requires abTEM and ASE") from error

    atoms_table = response["atoms"]
    extent_A = float(response["summary"]["field_of_view_A"])
    sample_thickness_A = float(response["summary"]["sample_thickness_A"])
    symbols = [ELEMENTS[index] for index in atoms_table["element_id"]]
    atoms = Atoms(
        symbols=symbols,
        positions=np.column_stack((atoms_table["x_A"], atoms_table["y_A"], atoms_table["z_A"])),
        cell=(extent_A, extent_A, sample_thickness_A),
        pbc=True,
    )
    frozen_phonon_configs = int(parameters.get("frozen_phonon_configs", 1))
    thermal_sigma_A = _nonnegative_float(parameters, "thermal_sigma_A", 0.0)
    potential_atoms: object = atoms
    if frozen_phonon_configs > 1 or thermal_sigma_A > 0.0:
        if thermal_sigma_A == 0.0:
            raise ValueError("scan.abtem.thermal_sigma_A must be positive for frozen phonons")
        potential_atoms = abtem.FrozenPhonons(
            atoms,
            num_configs=frozen_phonon_configs,
            sigmas=thermal_sigma_A,
            ensemble_mean=True,
            seed=scan.random_seed + 1,
        )
    potential = abtem.Potential(
        potential_atoms,
        sampling=potential_sampling_A,
        slice_thickness=_positive_float(
            parameters, "slice_thickness_A", 0.5 * scan.lattice_constant_A
        ),
        parametrization=str(parameters.get("potential_parametrization", "kirkland")),
        projection=str(parameters.get("potential_projection", "finite")),
        device=str(parameters.get("device", "cpu")),
    )
    scan_pixels = len(np.asarray(response["scan_x_A"]))
    scan_grid = abtem.GridScan(
        start=(0.0, 0.0),
        end=potential.extent,
        gpts=(scan_pixels, scan_pixels),
        endpoint=False,
    )
    detector = abtem.AnnularDetector(
        inner=_nonnegative_float(parameters, "haadf_inner_mrad", 50.0),
        outer=_positive_float(parameters, "haadf_outer_mrad", 80.0),
        to_cpu=True,
    )
    focal_depth_A = np.asarray(response["focal_depth_A"], dtype=np.float64)
    defocus_sign = float(parameters.get("defocus_sign", 1.0))
    maximum_batch = parameters.get("maximum_batch", "auto")
    haadf_fraction = np.zeros((len(focal_depth_A), scan_pixels, scan_pixels), dtype=np.float32)
    for depth_index, focal_depth in enumerate(focal_depth_A):
        probe = abtem.Probe(
            energy=simulation_config.experiment.beam_energy_keV * 1.0e3,
            semiangle_cutoff=_positive_float(parameters, "probe_semiangle_mrad", 20.0),
            defocus=defocus_sign * float(focal_depth),
            device=str(parameters.get("device", "cpu")),
        )
        measurement = probe.scan(
            potential,
            scan=scan_grid,
            detectors=detector,
            max_batch=maximum_batch,
            lazy=False,
        )
        haadf_fraction[depth_index] = _measurement_image(measurement, scan_pixels, "refined HAADF")

    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_hdf5, "r") as source, h5py.File(output_path, "w") as target:
        for name, value in source.attrs.items():
            target.attrs[name] = value
        for name in source:
            source.copy(name, target)
        del target["spatial/haadf_fraction"]
        target["spatial"].create_dataset(
            "haadf_fraction",
            data=haadf_fraction,
            compression="gzip",
            shuffle=True,
        )
        metadata = json.loads(target.attrs["metadata_json"])
        metadata["haadf_refinement"] = {
            "source_file": str(Path(source_hdf5).resolve()),
            "abtem_version": str(abtem.__version__),
            "potential_sampling_requested_A": potential_sampling_A,
            "potential_sampling_actual_A": [float(value) for value in potential.sampling],
            "potential_gpts": [int(value) for value in potential.gpts],
            "reason": ("50-80 mrad HAADF was not converged at the 0.12 A edge-map grid"),
        }
        target.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        target.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        target.attrs["refined_from"] = str(Path(source_hdf5).resolve())
    return {
        "scan_shape": list(haadf_fraction.shape),
        "potential_sampling_A": [float(value) for value in potential.sampling],
        "potential_gpts": [int(value) for value in potential.gpts],
        "haadf_fraction_min": float(haadf_fraction.min()),
        "haadf_fraction_max": float(haadf_fraction.max()),
        "haadf_fraction_mean": float(haadf_fraction.mean()),
    }
