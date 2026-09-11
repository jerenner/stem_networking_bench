# Energy-resolved specimen kernel

The built-in `energy-resolved` kernel turns a material spectral library into
unit-weight individual electrons. It is the bridge between electronic-
structure calculations and the existing spectrometer/detector chain; it does
not infer a specimen spectrum from DOEELS frames.

## Event model

For every incident electron the kernel:

1. draws at most one core component from the library's mutually exclusive
   integrated probabilities;
2. draws an independent Poisson number of low-loss excitations;
3. samples continuous energies from the selected component distributions and
   adds them, allowing a core edge to sit on a plural-scattering background;
4. samples an angular kick with elastic, low-loss, and core widths combined in
   quadrature; and
5. emits exactly one outgoing electron with `weight = 1`.

`spectral_component_id` records the principal component (`0` ZLP, `1` low
loss, `2+` library core components), while `plural_order` is the total number
of sampled inelastic events. `parent_electron_id` provides one-to-one truth
lineage. These fields survive passthrough materialization and spectrometer
acceptance.

This first implementation factorizes energy and angle and treats the core
components as mutually exclusive. It does not yet model energy-angle
correlations, spatially varying composition/channeling, interference between
edges, surface-loss thickness dependence, or a full compound ELNES
calculation.

## Spectral-library contract

`eels-sim-spectral-library-v1` is an HDF5 file containing:

- beam energy, specimen thickness, material, and elastic angular width;
- an optional normalized low-loss energy density, mean event count, and
  angular width;
- zero or more normalized core energy densities, onset energies, integrated
  event probabilities, angular widths, source strings, and metadata; and
- provenance metadata declaring what is calculated, fitted, or provisional.

The runtime sampler depends only on NumPy/HDF5. A library may therefore be
produced by GPAW, FEFF, eXSpy, abTEM, experimental reference data, or another
electronic-structure code without changing the kernel.

## Bulk-Si validation

`scripts/generate_si_gpaw_library.py` performs a small periodic diamond-Si LDA
ground-state calculation and GPAW-RPA dielectric response. The macroscopic
loss function with local-field effects supplies the 0.5–60 eV low-loss
density. The checked result has a dominant bulk-plasmon feature near 17–18 eV.

The Si L2,3 component beginning at 99.2 eV is deliberately an analytic
continuum/near-edge proxy. Its onset validates core-event sampling and
spectrometer placement, but its detailed ELNES and absolute intensity are not
a first-principles prediction. The low-loss mean event count and core
probability are provisional values for a nominal 50 nm sample.

```bash
cd simulation

.conda-envs/eels-sim-gpaw/bin/python \
  scripts/generate_si_gpaw_library.py

MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim simulate-1d-spectrum \
  --config config/energy_si.toml \
  --electrons 250000 --frames 117 \
  --output-hdf5 output/si_bulk_validation_spectrum.h5 \
  --output-plot output/si_bulk_validation_spectrum.png \
  --energy-min-eV -2 --energy-max-eV 250 --energy-step-eV 0.2
```

The GPAW `.gpw`, response CSV, and logs are cached in `output/gpaw_si`, so the
generator reuses completed stages.

## LMTO atomic-GOS demonstrator

Here LMTO means the disordered-rocksalt battery material
Li1.2Mn0.4Ti0.4O2. The demonstrator includes Ti M2,3 (~35 eV), Mn M2,3
(~49 eV), Li K (~54.7 eV), Ti L2,3 (~456 eV), O K (~532 eV), and Mn L2,3
(~640 eV), plus a three-feature low-loss mixture. This intentionally places
several shallow edges close enough to exercise simultaneous acquisition while
retaining widely separated high-energy edges.

The six core components now come from eXSpy's DFT-GOSH 1.5 database. Their
aperture-integrated atomic energy cross sections are converted into absolute
event probabilities using the configured LMTO stoichiometry, density, and
thickness. The low-loss mixture remains analytic. GOSH is an independent-atom
baseline and therefore does not predict LMTO-specific ELNES or oxidation-state
features. See [exspy_gosh.md](exspy_gosh.md) for the exact calculation and
provenance.

```bash
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/python scripts/download_gosh.py

MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/python scripts/generate_lmto_library.py

MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim simulate-1d-spectrum \
  --config config/energy_lmto.toml \
  --electrons 1000000 --frames 465 \
  --output-hdf5 output/lmto_gosh_1d_demonstrator.h5 \
  --output-plot output/lmto_gosh_1d_demonstrator.png \
  --energy-min-eV -2 --energy-max-eV 790 --energy-step-eV 0.25
```

Each spectrum HDF5 contains total truth/accepted histograms, accepted
component histograms, and event-level energy, component, plural order, and raw
detector row/column. `--output-phase-space` and `--output-detector-entries`
can additionally retain the standard intermediate files for Geant4 input.

## What makes the LMTO spectrum quantitative

The next physics upgrade is to replace the independent-atom GOSH shapes with
material-specific FEFF ELNES, especially for Ti L2,3, O K, and Mn L2,3. An
energy-resolved abTEM calculation remains the later route to atomic-position
and orientation dependence. Thickness-dependent low/plural-loss rates and
detector energy response should then be calibrated independently.
