from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from .phase_space import PHASE_SPACE_DTYPES, scatter_directions, validate_phase_space
from .specimen import SpecimenContext, SpecimenResult


@dataclass(frozen=True)
class AngularDistribution:
    """A normalized discrete distribution of paraxial scattering angles."""

    theta_x_mrad: np.ndarray
    theta_y_mrad: np.ndarray
    probability: np.ndarray

    def __post_init__(self) -> None:
        theta_x = np.asarray(self.theta_x_mrad, dtype=np.float64).ravel()
        theta_y = np.asarray(self.theta_y_mrad, dtype=np.float64).ravel()
        probability = np.asarray(self.probability, dtype=np.float64).ravel()
        if len(theta_x) == 0 or len(theta_x) != len(theta_y):
            raise ValueError("Angular-distribution arrays must have equal nonzero size")
        if len(theta_x) != len(probability):
            raise ValueError("Angular probabilities have the wrong size")
        if not (
            np.all(np.isfinite(theta_x))
            and np.all(np.isfinite(theta_y))
            and np.all(np.isfinite(probability))
        ):
            raise ValueError("Angular-distribution values must be finite")
        if np.any(probability < 0.0) or probability.sum() <= 0.0:
            raise ValueError("Angular probabilities must be nonnegative with positive sum")
        keep = probability > 0.0
        probability = probability[keep]
        probability /= probability.sum()
        object.__setattr__(self, "theta_x_mrad", theta_x[keep])
        object.__setattr__(self, "theta_y_mrad", theta_y[keep])
        object.__setattr__(self, "probability", probability)

    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.choice(len(self.probability), size=count, p=self.probability)
        return self.theta_x_mrad[indices], self.theta_y_mrad[indices]


@dataclass(frozen=True)
class AbTEMScatteringModel:
    elastic: AngularDistribution
    specimen_thickness_A: float
    core: AngularDistribution | None = None
    core_probability: float = 0.0
    core_energy_loss_eV: float | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.specimen_thickness_A) or self.specimen_thickness_A < 0:
            raise ValueError("specimen_thickness_A must be finite and nonnegative")
        if not np.isfinite(self.core_probability) or not 0.0 <= self.core_probability <= 1.0:
            raise ValueError("core_probability must be between zero and one")
        if self.core_probability > 0.0:
            if self.core is None:
                raise ValueError("A positive core_probability requires a core distribution")
            if self.core_energy_loss_eV is None or self.core_energy_loss_eV <= 0.0:
                raise ValueError("A positive core_probability requires a positive energy loss")


class AbTEMBackend(Protocol):
    def calculate(
        self, beam_energy_eV: float, parameters: Mapping[str, object]
    ) -> AbTEMScatteringModel: ...


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TOML table")
    return value


def _pair(value: object, name: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain two finite numbers")
    return float(values[0]), float(values[1])


def _triplet_of_positive_ints(value: object, name: str) -> tuple[int, int, int]:
    values = np.asarray(value)
    if values.shape != (3,) or np.any(values != values.astype(int)):
        raise ValueError(f"{name} must contain three integers")
    result = tuple(int(item) for item in values)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} entries must be positive")
    return result


def _measurement_distribution(
    measurement: object, max_angle_mrad: float
) -> tuple[AngularDistribution, float, tuple[int, int], tuple[float, float]]:
    if hasattr(measurement, "compute"):
        computed = measurement.compute()
        if computed is not None:
            measurement = computed
    array = np.asarray(getattr(measurement, "array"), dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("An abTEM pixelated measurement must have at least two axes")
    if array.ndim > 2:
        array = array.sum(axis=tuple(range(array.ndim - 2)))
    if not np.all(np.isfinite(array)):
        raise ValueError("abTEM returned non-finite diffraction intensities")
    if np.any(array < -1.0e-12):
        raise ValueError("abTEM returned negative diffraction intensities")
    array = np.maximum(array, 0.0)

    angular_sampling = tuple(float(item) for item in measurement.angular_sampling)
    if len(angular_sampling) != 2 or any(item <= 0.0 for item in angular_sampling):
        raise ValueError("abTEM returned invalid angular sampling")
    theta_x = (
        np.arange(array.shape[-2], dtype=np.float64) - (array.shape[-2] - 1) / 2.0
    ) * angular_sampling[0]
    theta_y = (
        np.arange(array.shape[-1], dtype=np.float64) - (array.shape[-1] - 1) / 2.0
    ) * angular_sampling[1]
    grid_x, grid_y = np.meshgrid(theta_x, theta_y, indexing="ij")
    radial_mask = np.hypot(grid_x, grid_y) <= max_angle_mrad + 1.0e-12
    captured_intensity = float(array[radial_mask].sum())
    if captured_intensity <= 0.0:
        raise ValueError("abTEM returned zero intensity inside the angular cutoff")
    distribution = AngularDistribution(grid_x[radial_mask], grid_y[radial_mask], array[radial_mask])
    return distribution, captured_intensity, array.shape, angular_sampling


class LiveAbTEMBackend:
    """Current abTEM 1.0.x implementation, imported only when selected."""

    _PARAMETERS = {
        "structure",
        "structure_file",
        "lattice_constant_A",
        "repetitions",
        "orientation",
        "potential_sampling_A",
        "slice_thickness_A",
        "potential_parametrization",
        "potential_projection",
        "probe_semiangle_mrad",
        "probe_position_fractional",
        "max_scattering_angle_mrad",
        "device",
        "energy_tolerance_eV",
        "core_loss",
    }

    @staticmethod
    def _load_modules():
        try:
            import abtem
            from ase.build import bulk
            from ase.io import read
        except ImportError as error:
            raise ImportError(
                "The abtem specimen kernel needs the optional abTEM dependency; "
                "install the project with `pip install -e '.[abtem]'`"
            ) from error
        return abtem, bulk, read

    @staticmethod
    def _atoms(parameters, bulk, read):
        structure = str(parameters.get("structure", "nio_rocksalt"))
        structure_file = parameters.get("structure_file")
        if structure_file is not None:
            atoms = read(Path(str(structure_file)))
            source = {"kind": "file", "path": str(Path(str(structure_file)))}
        elif structure == "nio_rocksalt":
            lattice_constant_A = float(parameters.get("lattice_constant_A", 4.17))
            if lattice_constant_A <= 0.0:
                raise ValueError("lattice_constant_A must be positive")
            atoms = bulk("NiO", "rocksalt", a=lattice_constant_A, cubic=True)
            source = {
                "kind": "built_in",
                "name": "nio_rocksalt",
                "lattice_constant_A": lattice_constant_A,
            }
        else:
            raise ValueError("Unknown built-in structure; use 'nio_rocksalt' or structure_file")
        repetitions = _triplet_of_positive_ints(
            parameters.get("repetitions", (1, 1, 1)), "repetitions"
        )
        atoms = atoms.repeat(repetitions)
        return atoms, source, repetitions

    def calculate(
        self, beam_energy_eV: float, parameters: Mapping[str, object]
    ) -> AbTEMScatteringModel:
        unknown = set(parameters) - self._PARAMETERS
        if unknown:
            raise ValueError(f"Unknown abTEM parameter(s): {sorted(unknown)}")
        if beam_energy_eV <= 0.0:
            raise ValueError("beam_energy_eV must be positive")
        orientation = str(parameters.get("orientation", "[001]"))
        if orientation != "[001]":
            raise ValueError(
                "The first abTEM adapter supports only a cell already oriented along [001]"
            )

        abtem, bulk, read = self._load_modules()
        atoms, structure_source, repetitions = self._atoms(parameters, bulk, read)
        potential_sampling_A = float(parameters.get("potential_sampling_A", 0.1))
        slice_thickness_A = float(parameters.get("slice_thickness_A", 2.0))
        probe_semiangle_mrad = float(parameters.get("probe_semiangle_mrad", 20.0))
        max_angle_mrad = float(parameters.get("max_scattering_angle_mrad", 80.0))
        if (
            min(
                potential_sampling_A,
                slice_thickness_A,
                probe_semiangle_mrad,
                max_angle_mrad,
            )
            <= 0.0
        ):
            raise ValueError("abTEM sampling, slice, probe, and angular values must be positive")
        probe_fractional = _pair(
            parameters.get("probe_position_fractional", (0.5, 0.5)),
            "probe_position_fractional",
        )
        if any(not 0.0 <= item <= 1.0 for item in probe_fractional):
            raise ValueError("probe_position_fractional entries must be between zero and one")
        device = str(parameters.get("device", "cpu"))
        parametrization = str(parameters.get("potential_parametrization", "kirkland"))
        projection = str(parameters.get("potential_projection", "finite"))

        potential = abtem.Potential(
            atoms,
            sampling=potential_sampling_A,
            slice_thickness=slice_thickness_A,
            parametrization=parametrization,
            projection=projection,
            device=device,
        )
        probe = abtem.Probe(
            energy=beam_energy_eV,
            semiangle_cutoff=probe_semiangle_mrad,
            device=device,
        )
        scan_position_A = tuple(
            fraction * extent for fraction, extent in zip(probe_fractional, potential.extent)
        )
        exit_waves = probe.multislice(potential, scan=scan_position_A, lazy=False)
        elastic_measurement = exit_waves.diffraction_patterns(
            max_angle=max_angle_mrad,
            fftshift=True,
            parity="odd",
            renormalize=True,
        )
        elastic, elastic_intensity, elastic_shape, angular_sampling = _measurement_distribution(
            elastic_measurement, max_angle_mrad
        )

        core_parameters = _mapping(parameters.get("core_loss", {}), "core_loss")
        core_enabled = bool(core_parameters.get("enabled", False))
        core = None
        core_probability = 0.0
        core_energy_loss_eV = None
        core_metadata: dict[str, object] = {"enabled": core_enabled}
        if core_enabled:
            from abtem.inelastic.core_loss import SubshellTransitions

            if "energy_loss_eV" not in core_parameters:
                raise ValueError("Enabled core_loss requires an explicit energy_loss_eV")
            atomic_number = int(core_parameters.get("atomic_number", 8))
            n = int(core_parameters.get("n", 1))
            angular_momentum = int(core_parameters.get("l", 0))
            order = int(core_parameters.get("order", 1))
            epsilon_eV = float(core_parameters.get("epsilon_eV", 10.0))
            xc = str(core_parameters.get("xc", "PBE"))
            double_channel = bool(core_parameters.get("double_channel", False))
            core_energy_loss_eV = float(core_parameters["energy_loss_eV"])
            probability_scale = float(core_parameters.get("probability_scale", 1.0))
            if core_energy_loss_eV <= 0.0 or epsilon_eV <= 0.0:
                raise ValueError("Core energy_loss_eV and epsilon_eV must be positive")
            if probability_scale <= 0.0:
                raise ValueError("core_loss.probability_scale must be positive")
            transitions = SubshellTransitions(
                atomic_number,
                n,
                angular_momentum,
                order=order,
                epsilon=epsilon_eV,
                xc=xc,
            )
            try:
                transition_potentials = transitions.get_transition_potentials(
                    extent=potential.extent,
                    gpts=potential.gpts,
                    energy=beam_energy_eV,
                    double_channel=double_channel,
                )
            except ModuleNotFoundError as error:
                if error.name and error.name.startswith("gpaw"):
                    raise ImportError(
                        "abTEM core-loss transition-potential generation requires "
                        "a working GPAW installation; elastic-only mode does not"
                    ) from error
                raise
            detector = abtem.PixelatedDetector(
                max_angle=max_angle_mrad,
                reciprocal_space=True,
                to_cpu=True,
            )
            core_measurement = probe.transition_potential_scan(
                potential=potential,
                transition_potentials=transition_potentials,
                scan=scan_position_A,
                detectors=detector,
                sites=atoms,
                lazy=False,
                double_channel=double_channel,
            )
            core, unscaled_probability, core_shape, core_sampling = _measurement_distribution(
                core_measurement, max_angle_mrad
            )
            core_probability = unscaled_probability * probability_scale
            if core_probability > 1.0 + 1.0e-12:
                raise ValueError(
                    "Integrated core-loss probability exceeds one; use a thinner "
                    "sample or a model that includes plural scattering"
                )
            core_probability = min(core_probability, 1.0)
            core_metadata = {
                "enabled": True,
                "atomic_number": atomic_number,
                "n": n,
                "l": angular_momentum,
                "order": order,
                "epsilon_eV": epsilon_eV,
                "energy_loss_eV": core_energy_loss_eV,
                "xc": xc,
                "double_channel": double_channel,
                "unscaled_integrated_probability": unscaled_probability,
                "probability_scale": probability_scale,
                "integrated_probability": core_probability,
                "diffraction_shape": list(core_shape),
                "angular_sampling_mrad": list(core_sampling),
            }

        cell_A = np.asarray(atoms.cell.array, dtype=np.float64)
        metadata = {
            "engine": "abTEM",
            "abtem_version": str(abtem.__version__),
            "structure": structure_source,
            "chemical_formula": atoms.get_chemical_formula(),
            "atom_count": len(atoms),
            "repetitions": list(repetitions),
            "orientation": orientation,
            "cell_A": cell_A.tolist(),
            "specimen_thickness_A": float(cell_A[2, 2]),
            "beam_energy_eV": beam_energy_eV,
            "probe_semiangle_mrad": probe_semiangle_mrad,
            "probe_position_fractional": list(probe_fractional),
            "probe_position_A": list(scan_position_A),
            "potential_sampling_A": list(float(item) for item in potential.sampling),
            "potential_gpts": list(int(item) for item in potential.gpts),
            "slice_thickness_A": slice_thickness_A,
            "potential_slices": int(potential.num_slices),
            "potential_parametrization": parametrization,
            "potential_projection": projection,
            "device": device,
            "maximum_scattering_angle_mrad": max_angle_mrad,
            "elastic_captured_intensity": elastic_intensity,
            "elastic_diffraction_shape": list(elastic_shape),
            "elastic_angular_sampling_mrad": list(angular_sampling),
            "core_loss": core_metadata,
            "limitations": [
                "static lattice (no frozen-phonon ensemble)",
                "one fixed probe position for every incident electron",
                "no low-loss, phonon, or plural-inelastic scattering",
                "angular distributions are conditional on the configured cutoff",
            ],
        }
        return AbTEMScatteringModel(
            elastic=elastic,
            specimen_thickness_A=float(cell_A[2, 2]),
            core=core,
            core_probability=core_probability,
            core_energy_loss_eV=core_energy_loss_eV,
            metadata=metadata,
        )


class AbTEMSpecimenKernel:
    """Adapt abTEM diffraction/core-loss results to per-parent weighted branches."""

    name = "abtem"

    def __init__(self, backend: AbTEMBackend | None = None):
        self._backend = LiveAbTEMBackend() if backend is None else backend

    def simulate(
        self,
        incident_electrons: Mapping[str, np.ndarray],
        context: SpecimenContext,
    ) -> SpecimenResult:
        count = validate_phase_space(incident_electrons)
        if not np.allclose(incident_electrons["weight"], 1.0):
            raise ValueError("The abtem kernel requires unit-weight incident electrons")
        energies = np.asarray(incident_electrons["kinetic_energy_eV"], dtype=np.float64)
        energy_tolerance_eV = float(context.parameters.get("energy_tolerance_eV", 1.0))
        if energy_tolerance_eV < 0.0:
            raise ValueError("energy_tolerance_eV must be nonnegative")
        if float(np.ptp(energies)) > energy_tolerance_eV:
            raise ValueError(
                "The first abTEM adapter requires one incident beam energy; split "
                "the input by energy or increase energy_tolerance_eV deliberately"
            )
        beam_energy_eV = float(np.mean(energies))
        model = self._backend.calculate(beam_energy_eV, context.parameters)

        branches: list[tuple[int, float, float, int | None, AngularDistribution]] = []
        elastic_probability = 1.0 - model.core_probability
        if elastic_probability > 0.0:
            branches.append((0, elastic_probability, 0.0, None, model.elastic))
        if model.core_probability > 0.0:
            assert model.core is not None
            assert model.core_energy_loss_eV is not None
            if np.any(energies <= model.core_energy_loss_eV):
                raise ValueError("Core energy loss is not below every incident energy")
            branches.append(
                (
                    1,
                    model.core_probability,
                    model.core_energy_loss_eV,
                    3,
                    model.core,
                )
            )

        incident_directions = np.column_stack(
            (
                incident_electrons["dir_x"],
                incident_electrons["dir_y"],
                incident_electrons["dir_z"],
            )
        )
        blocks: dict[str, list[np.ndarray]] = {name: [] for name in PHASE_SPACE_DTYPES}
        parent_blocks: list[np.ndarray] = []
        branch_blocks: list[np.ndarray] = []
        for branch_id, probability, energy_loss_eV, channel, distribution in branches:
            theta_x, theta_y = distribution.sample(count, context.rng)
            directions = scatter_directions(incident_directions, theta_x, theta_y)
            for name, dtype in PHASE_SPACE_DTYPES.items():
                if name == "electron_id":
                    continue
                if name == "weight":
                    values = np.full(count, probability, dtype=dtype)
                elif name == "kinetic_energy_eV":
                    values = np.asarray(energies - energy_loss_eV, dtype=dtype)
                elif name == "loss_channel":
                    values = (
                        np.asarray(incident_electrons[name], dtype=dtype).copy()
                        if channel is None
                        else np.full(count, channel, dtype=dtype)
                    )
                elif name == "z_um":
                    values = (
                        np.asarray(incident_electrons[name], dtype=dtype)
                        + model.specimen_thickness_A * 1.0e-4
                    )
                elif name == "dir_x":
                    values = directions[:, 0].astype(dtype, copy=False)
                elif name == "dir_y":
                    values = directions[:, 1].astype(dtype, copy=False)
                elif name == "dir_z":
                    values = directions[:, 2].astype(dtype, copy=False)
                else:
                    values = np.asarray(incident_electrons[name], dtype=dtype).copy()
                blocks[name].append(values)
            parent_blocks.append(
                np.asarray(incident_electrons["electron_id"], dtype=np.uint64).copy()
            )
            branch_blocks.append(np.full(count, branch_id, dtype=np.uint32))

        output_count = count * len(branches)
        outgoing = {
            name: (
                np.arange(output_count, dtype=dtype)
                if name == "electron_id"
                else np.concatenate(blocks[name])
            )
            for name, dtype in PHASE_SPACE_DTYPES.items()
        }
        outgoing["parent_electron_id"] = np.concatenate(parent_blocks)
        outgoing["branch_id"] = np.concatenate(branch_blocks)
        return SpecimenResult(
            electrons=outgoing,
            weight_semantics="branch_probability",
            metadata={
                "model": "abTEM angular-distribution adapter",
                "reference_energy_eV": context.reference_energy_eV,
                "beam_energy_eV": beam_energy_eV,
                "angular_sampling_strategy": (
                    "one conditional angle per branch and incident electron; "
                    "categorical materialization selects the physical outcome"
                ),
                "backend": dict(model.metadata or {}),
            },
        )
