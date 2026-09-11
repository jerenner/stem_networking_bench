# Specimen kernels and weighted-electron materialization

The specimen layer is a replaceable Python plugin. It accepts incident
electron phase space and returns weighted outgoing phase space in the same
versioned field layout. The spectrometer and detector therefore do not depend
on abTEM, MULTEM, or any other particular specimen engine.

The built-in `identity` kernel is a no-sample diagnostic. It validates the
interface but contains no scattering physics.

## Kernel interface

A kernel object has a stable `name` and a `simulate` method:

```python
from eels_sim.specimen import SpecimenResult

class MyKernel:
    name = "my-kernel"

    def simulate(self, incident_electrons, context):
        # context.reference_energy_eV
        # context.parameters
        # context.rng: deterministically seeded NumPy generator
        outgoing = ...
        return SpecimenResult(
            electrons=outgoing,
            weight_semantics="branch_probability",
            metadata={"model_version": "1"},
        )
```

Required output fields are those in `eels-sim-phase-space-v1`. Kernels that
emit multiple mutually exclusive outcomes for one incident electron also emit:

| Optional field | Type | Meaning |
|---|---|---|
| `parent_electron_id` | uint64 | Incident-electron truth ID |
| `branch_id` | uint32 | Kernel-defined outcome within that parent |
| `spectral_component_id` | uint16 | Principal sampled spectral component |
| `plural_order` | uint16 | Number of sampled inelastic events |

Each outgoing record still needs a globally unique `electron_id`; it is a
weighted record ID until materialization. Kernel parameters and metadata must
be JSON-serializable.

Plugins can be selected directly as `package.module:object`, or installed with
this Python entry-point group:

```toml
[project.entry-points."eels_sim.specimen_kernels"]
my_kernel = "my_package.kernel:MyKernel"
```

`MyKernel` may be an instance, class with a no-argument constructor, or factory
returning an object with `name` and `simulate`.

## Weight semantics

The kernel must state exactly what `weight` means. The three supported meanings
are intentionally not interchangeable:

### `branch_probability`

Records sharing `parent_electron_id` are mutually exclusive outcomes of one
incident electron. Their weights must sum to at most one. Categorical sampling
selects at most one branch. A sum below one leaves a probability that the
electron is absorbed, rejected, or otherwise outside the represented output.

This is the appropriate form for a kernel that emits, for every incoming
electron, alternatives such as ZLP, plasmon loss, and a core-loss transition.

### `expected_electrons`

Each record is an independent phase-space bin or ray whose weight is its
expected electron count. The materializer draws an independent Poisson count
for every record. This is appropriate for weighted intensity output from an
expensive wave calculation.

### `individual_electrons`

The kernel has already sampled every stochastic outcome and must emit unit
weights. Automatic materialization selects `passthrough`, which retains the
records and all component/lineage fields while normalizing output IDs. The
built-in energy-resolved kernel uses this form.

The materialized result always has `weight = 1`. It receives new contiguous
`electron_id` values and adds `source_record_id`; parent and branch lineage are
preserved when present. `max_electrons` prevents an accidental enormous
allocation caused by incorrect weights or units.

## Configuration and commands

```toml
[specimen]
kernel = "identity"
random_seed = 97531

[specimen.parameters]
# Plugin-specific, for example structure, thickness, orientation, and edge.

[materialization]
mode = "auto"
random_seed = 86420
max_electrons = 10000000
```

`auto` reads the kernel's weight-semantics metadata and chooses categorical,
Poisson, or passthrough sampling. An incompatible explicit mode is rejected
rather than silently changing the physical interpretation.

```bash
PYTHONPATH=python python3 -m eels_sim list-specimen-kernels

PYTHONPATH=python python3 -m eels_sim run-specimen \
  output/incident_phase_space.h5 \
  --config config/eels_reference.toml \
  --output output/specimen_weighted.h5

PYTHONPATH=python python3 -m eels_sim materialize-phase-space \
  output/specimen_weighted.h5 \
  --config config/eels_reference.toml \
  --output output/specimen_individual.h5
```

The unit-weight output is accepted directly by `transfer-spectrometer` and its
Geant4 CSV export.

## Built-in abTEM kernel

The first physical adapter is now available as `kernel = "abtem"`. It runs a
real static-lattice elastic multislice calculation and has an optional abTEM
transition-potential path for one core edge. It compresses each conditional
angular grid to one sampled weighted alternative per parent and therefore uses
the existing categorical materializer without changing the downstream
contract.

See [abtem_adapter.md](abtem_adapter.md) for installation, the NiO preset,
configuration, commands, core-loss/GPAW requirements, and current physics
limits. Low-loss and plural-scattering models can be added later without
changing the plugin or phase-space interfaces.

## Built-in energy-resolved kernel

The `energy-resolved` kernel consumes a versioned spectral-library HDF5 and
directly samples unit-weight electrons with continuous energy losses. It
supports a Poisson number of low-loss events, one optional core event,
component truth IDs, plural order, and configurable angular scaling. See
[energy_resolved.md](energy_resolved.md) for the bulk-Si validation and LMTO
demonstrator.
