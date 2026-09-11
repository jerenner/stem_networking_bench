# First abTEM specimen adapter

The built-in `abtem` kernel is the first physical implementation of the
specimen plugin interface. It runs an abTEM multislice calculation for a
static specimen and converts the pixelated diffraction result into weighted
outgoing-electron branches. abTEM is an optional dependency; importing the
rest of `eels_sim` does not import it.

The supplied `config/abtem_nio.toml` preset constructs cubic rocksalt NiO,
repeats it to a 2x2x4 cell, sends a 300 keV 20 mrad probe along [001], and
calculates elastic scattering out to 60 mrad. Every numerical and structural
input, the abTEM version, grid shape, angular sampling, cell, and known model
limitations are recorded in the output HDF5 metadata.

## Install and run

```bash
cd simulation
./simulation/scripts/setup_gpaw_env.sh
scripts/setup_gpaw_env.sh
conda activate .conda-envs/eels-sim-gpaw

cd simulation
eels-sim generate-phase-space \
  --config config/abtem_nio.toml --output output/abtem_incident.h5 \
  --electrons 100 --frames 1

MPLCONFIGDIR=/tmp/eels-sim-mpl eels-sim run-specimen \
  output/abtem_incident.h5 \
  --config config/abtem_nio.toml \
  --output output/abtem_weighted.h5

eels-sim materialize-phase-space \
  output/abtem_weighted.h5 \
  --config config/abtem_nio.toml \
  --output output/abtem_individual.h5

eels-sim transfer-spectrometer \
  output/abtem_individual.h5 \
  --config config/abtem_nio.toml \
  --output output/abtem_detector_entries.h5 \
  --geant4-csv output/abtem_geant4_source.csv
```

`scripts/setup_gpaw_env.sh` locates the common Miniconda/Miniforge
initialization scripts, then creates or updates the native project prefix in
`.conda-envs/eels-sim-gpaw`, using
`environment-gpaw.yml`, then compiles GPAW 25.7.0 with the settings in
`config/gpaw_siteconfig.py`. The build is serial and links the Conda LibXC,
OpenBLAS, and FFTW libraries. This is necessary because conda-forge currently
has no native `osx-arm64` GPAW binary, while its available macOS GPAW package
is Intel-only and this machine has no Rosetta runtime.

The generated input is a diagnostic beam source, not a measured EELS spectrum.
The specimen output is physical only to the extent of the configured abTEM
model and structure.

## Convergence study

The reproducible one-at-a-time study is defined in
`config/abtem_nio_convergence.toml`. Run it in the same GPAW environment:

```bash
cd simulation
MPLCONFIGDIR=/tmp/eels-sim-mpl eels-sim run-abtem-convergence \
  --config config/abtem_nio_convergence.toml \
  --output-json output/abtem_nio_convergence.json \
  --output-csv output/abtem_nio_convergence.csv \
  --output-plot output/abtem_nio_convergence.png \
  --output-combined-plot output/abtem_nio_combined_confirmation.png
```

The current matrix contains 14 unique abTEM calculations: the original 12
one-at-a-time calculations plus a combined reference and a larger/finer
confirmation. Each numerical or model axis changes only one baseline setting:

- lateral supercell: 1x1, 2x2, and 3x3, referenced to 3x3;
- real-space potential sampling: 0.18, 0.15, 0.12, and 0.09 A, referenced to
  0.09 A;
- angular cutoff: 40, 60, and 80 mrad, referenced to 80 mrad; and
- single versus double channeling, referenced to double channeling.

The 1, 2, 4, and 8 unit-cell thickness series is different: thickness changes
the physical specimen, so it is reported as an absolute trend and as
core-loss probability per A rather than treated as a numerical error. The JSON
keeps all input parameters, backend metadata, angular summary statistics, and
radial histograms. Shape comparisons include both total-variation distance and
radial Wasserstein-1 distance in mrad; the latter is less brittle when a narrow
diffraction feature moves between adjacent histogram bins. The CSV has one row
per axis point with differences from that axis's reference; the PNG gives the
quick diagnostic view. A failed case is recorded without discarding completed
calculations, and the command exits nonzero if any case failed.

The configured `comparison_collection_angle_mrad = 50` matches the
spectrometer preset. Each distribution is also truncated and renormalized at
that acceptance, and the report records the absolute core/elastic probability
that can reach the detector. This prevents convergence changes in deliberately
simulated 50-100 mrad tails from being confused with changes in the measured
signal.

### First completed matrix (2026-08-25)

All 12 cases completed. The baseline integrated O-K probability was
`6.0221e-7` for the 16.68 A specimen. Relative to each one-axis reference, the
baseline result changed as follows:

| changed axis | configured reference | baseline core probability difference | elastic radial W1 | core radial W1 |
|---|---:|---:|---:|---:|
| lateral extent | 3x3 | +3.28% | 0.525 mrad | 0.580 mrad |
| potential sampling | 0.09 A | +1.01% | 1.664 mrad | 0.745 mrad |
| angular cutoff | 80 mrad | -3.04% | approximately 0 mrad | 1.211 mrad |
| double channeling | enabled | +5.34% | 0 mrad | 1.246 mrad |

The thickness series was close to linear in integrated core probability. The
probability per A at 4 and 8 repeated cells differed by only about 0.01%, while
the 1- and 2-cell values were 2.47% and 1.86% below the 4-cell value. This is a
physical channeling/thickness result, not a numerical convergence claim.

The original baseline is therefore a reasonable few-percent exploratory model
for the integrated edge yield, but it is not a 1% reference and the angular
distributions are less converged than the scalar yield. These are
one-at-a-time comparisons and cannot establish combined convergence by
themselves. The physical thickness must be chosen from the specimen rather
than from this numerical study.

### Combined reference and confirmation (2026-08-25)

The combined reference uses 3x3x4 repetitions, 0.09 A sampling, an 80 mrad
simulation cutoff, and double channeling. The confirmation holds the 16.68 A
physical thickness and all beam/probe settings fixed while changing to 4x4x4,
0.075 A, and 100 mrad. Both completed successfully:

| quantity | combined reference | confirmation | confirmation change |
|---|---:|---:|---:|
| atoms / potential grid | 288 / 139x139 | 512 / 223x223 | -- |
| full-range O-K probability | 5.9172e-7 | 6.0147e-7 | +1.648% |
| O-K probability within 50 mrad | 5.4305e-7 | 5.3983e-7 | -0.592% |
| elastic probability within 50 mrad | 0.98138 | 0.98219 | +0.083% |
| O-K radial RMS within 50 mrad | 28.186 mrad | 28.418 mrad | +0.823% |
| elastic radial RMS within 50 mrad | 21.388 mrad | 21.860 mrad | +2.207% |

Within the detector's acceptance, the core yield and core radial RMS agree to
better than 1%; their radial Wasserstein-1 distance is 0.278 mrad. The
full-range probability differs more because the confirmation adds the 80-100
mrad O-K tail, which the 50 mrad spectrometer rejects. The elastic accepted
probability is stable, while the accepted elastic angular shape is converged
only at about the 2% RMS level (0.532 mrad radial Wasserstein-1 distance).

Thus the 3x3/0.09 A/80 mrad/double-channel reference is adequate for roughly
1% detector-accepted O-K yield observables, but not for a blanket 1% claim on
all angular observables. The 4x4/0.075 A/100 mrad confirmation is the safer
production choice when its modest additional runtime is acceptable. A stricter
elastic/ZLP angular target would require another check or one-at-a-time
isolation around the confirmation settings.

## How the weighted adapter works

One abTEM calculation produces an elastic angular intensity grid at a fixed
probe position. If the optional core edge is enabled, a transition-potential
calculation produces a second angular grid and its integrated per-incident
probability.

For each incident electron the adapter samples one angle from each conditional
grid and emits:

- an elastic/ZLP branch with weight `1 - p_core`; and
- a core-loss branch with weight `p_core` and the configured energy loss.

The records share `parent_electron_id`. Categorical materialization selects
exactly one branch for every parent because the branch weights sum to one.
This is an unbiased Monte Carlo compression of the two angular grids and does
not introduce a second Poisson fluctuation in the already-individual incident
beam. The outgoing direction is formed in a local basis around the incoming
direction, and the specimen cell thickness is added to `z_um`.

## Core-loss option

The first edge configuration is O K (`Z=8`, `n=1`, `l=0`) with single
channeling by default:

```toml
[specimen.parameters.core_loss]
enabled = true
atomic_number = 8
n = 1
l = 0
order = 1
epsilon_eV = 10.0
energy_loss_eV = 532.0
xc = "PBE"
double_channel = false
probability_scale = 1.0
```

abTEM uses GPAW's all-electron atomic calculation to generate these transition
potentials. The supplied Conda setup installs that dependency, and the preset
enables the O K branch. The adapter still reports a targeted error if GPAW is
missing. `double_channel = true` propagates the inelastically scattered wave
through the remaining specimen and is substantially more expensive.

For the current 2x2x4 static NiO cell, single-channel smoke calculation, and
60 mrad cutoff, the integrated probability was approximately `6.02e-7` per
incident electron. Treat this as a software-validation result, not a validated
physical cross section: it still needs convergence studies and comparison to
a known thickness and collection-angle measurement.

`energy_loss_eV` is currently a single representative loss assigned to every
selected core event. It is separate from abTEM's `epsilon_eV`, which controls
the excited continuum radial state. The value in the preset is provisional;
an energy-resolved ELNES kernel is still needed to model the full edge shape.

## Deliberate first-version limits

- The static-lattice elastic calculation does not yet include thermal diffuse
  scattering or a frozen-phonon ensemble.
- All incident records use one configured probe position. Input `x_um/y_um`
  are preserved as exit coordinates but do not select different abTEM scan
  positions yet.
- One run accepts only a monoenergetic incident beam within
  `energy_tolerance_eV`.
- There is at most one core-loss branch and no low-loss/plasmon, phonon, or
  plural-inelastic model.
- Angular distributions are normalized conditional on
  `max_scattering_angle_mrad`; convergence must be checked by increasing the
  real-space grid and angular cutoff.
- The built-in crystal is [001] rocksalt NiO. `structure_file` can load another
  ASE-readable cell, but the first adapter assumes it is already oriented with
  the beam along Cartesian `+z`.

The next specimen step should add a frozen-phonon ensemble and scan-position
grouping, then replace the monoenergetic core branch with an energy-resolved
edge distribution and validate its absolute probability against a known
thickness/collection-angle measurement.
