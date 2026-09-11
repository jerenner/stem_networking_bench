# EELS simulation

This directory is a self-contained, first-principles simulation chain for the
microscope, silicon sensor, and readout. The detector response and first-order
spectrometer transfer are executable end to end:

```text
specimen-exit electron phase space
    -> energy/position/angle spectrometer transfer and acceptance
    -> four-lane ZLP plus CoreLoss detector layout
    -> Geant4 transport and energy deposits in Si
    -> e-h pair creation with Fano fluctuations
    -> field-driven drift and analytic diffusion
    -> stochastic charge sharing onto pixels
    -> gain, pedestal, common-mode/read noise, ADC quantization
    -> HDF5 frames [frame, row, column]
```

The specimen and spectrometer are explicit upstream modules rather than an
empirical spectrum sampled from the existing data. This is important: Geant4
is appropriate for transport and detector energy deposition, but detailed
crystal channeling and EELS edge physics require a quantum scattering model.

## What is implemented

- `cpp/`: a C++17 Geant4 executable, `eels_detector`, with configurable sensor
  rows, columns, pitch, thickness, step size, primary energy, position spread,
  energy spread, and angular spread.
- `python/eels_sim/`: phase-space and spectrometer interfaces, charge
  transport, pixelization, electronics, command-line tools, and HDF5 writers.
- `spectral_libraries/`: versioned energy distributions and event rates used
  by the energy-resolved individual-electron specimen kernel.
- `macros/`: a fast 64x128 smoke run and a physical 960x3840 EELS preset.
- `config/`: separately sourced microscope and provisional detector/readout
  parameters.
- `docs/`: architecture, interface contracts, assumptions, and the route to a
  quantum specimen model.
- `tests/`: deterministic checks of occupancy, drift/diffusion, charge
  conservation, and ADC behavior.
- `calibration/`: compact calibration summaries and diagnostic plots. The
  generated map HDF5 is reproducible and excluded from version control.

Geant4 treats the silicon as one slab. Pixel boundaries are applied during
charge collection, so changing from 960x3840 to another array does not create
millions of Geant4 volumes. The live DAQIRI receiver's 1024x3840 buffer can be
produced later by padding the physical 960 rows at the output boundary.

## Parameter provenance

The added `backpropcount_arxiv2511.03933.pdf` gives two distinct parameter
sets, intentionally kept separate:

| Context | Energy | Sensor/frame | Rate/current |
|---|---:|---|---|
| Paper's experimental microscope | 300 keV | 4D Camera, 576x576 | 87 kHz; about 30 pA |
| Paper's synthetic detector hits | 100 keV | 5 um Si; 10 um pixels | not a frame-timing model |
| Original EELS geometry preset | 300 keV | 960x3840; provisional 5/10 um thickness/pitch | 87 kHz; 30 pA reference |
| Current LMTO application demo | 200 keV | 960x3840; provisional 5/10 um thickness/pitch | 87 kHz; 30 pA reference |

At 30 pA and 87 kHz, the Poisson mean is about 2152.24 incident electrons per
frame and the nominal integration time is 11.4943 us. Sensor thickness, pitch,
bias, and ADC gain must be replaced when the EELS detector documentation is
available. The existing DOEELS data should be used to estimate dark pedestal,
row/column structure, common-mode noise, read noise, gain nonuniformity, and
saturation—not the specimen scattering law.

## Build and run

Set `EELS_GEANT4_PREFIX` to a Geant4 installation built with the required data
files. If that installation needs nonstandard runtime libraries, provide their
colon-separated directories through `EELS_RUNTIME_LIBRARY_PATH`.

```bash
cd simulation
export EELS_GEANT4_PREFIX=/path/to/geant4/install
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$EELS_GEANT4_PREFIX"
cmake --build build -j8
./scripts/run_geant4_local.sh macros/smoke.mac
PYTHONPATH=python python3 -m eels_sim inspect output/smoke_deposits
PYTHONPATH=python python3 -m eels_sim digitize \
  output/smoke_deposits \
  --config config/smoke.toml \
  --output output/smoke_frames.h5 \
  --keep-intermediate
PYTHONPATH=python python3 -m unittest discover -s tests -v
```

Geant4 emits CSV ntuples to avoid an HDF5 ABI dependency; the Python stage
writes the canonical HDF5 product.

Useful commands:

```bash
PYTHONPATH=python python3 -m eels_sim occupancy -c config/eels_reference.toml
PYTHONPATH=python python3 -m eels_sim --help
```

## Reproduce the 200 keV movie

Generated simulation products, measured microscope data, GOSH tables, render
caches, and movies are intentionally excluded from Git. The repository contains
the complete code, deterministic seeds, configurations, Geant4 macro, scene
definitions, and an orchestration script.

Two measured NiO inputs must be supplied separately:

- a dark-frame HDF5 stack with dataset `frames`;
- the 15 pA spectrum-frame HDF5 stack used for sparse-tail calibration.

Create the physics and Manim environments:

```bash
scripts/setup_gpaw_env.sh
scripts/setup_demo_env.sh
```

Then run the complete 200 keV chain:

```bash
export EELS_GEANT4_PREFIX=/path/to/geant4/install
export EELS_DARK_HDF5=/path/to/nio_15pa_dark_frames_float32_uncompressed.h5
export EELS_SPECTRUM_HDF5=/path/to/nio_15pa_spectrum_frames_float32_uncompressed.h5
scripts/reproduce_200kev_demo.sh
```

The script downloads the checksum-verified DFT-GOSH table, generates the 200
keV GOSH library, runs abTEM and the HAADF refinement, constructs the spectrum
image, generates and compresses the Geant4 silicon response, recalibrates the
NiO detector model, simulates the raw readout and exposure sweep, generates the
presentation assets, and renders the final 1080p movie. The result is written
to `demo/renders/lmto_doeels_workflow_200keV_FullDemo.mp4`.

The simulation is deterministic for the recorded software versions and random
seeds, but bitwise identity is not guaranteed across operating systems,
compilers, FFT libraries, or GPU/CPU backends. See [demo/README.md](demo/README.md)
for the exact presentation inputs and scientific claim boundaries.

## Spectrometer transfer

The implemented first-order transfer maps specimen-exit energy, position, and
angle onto a continuous stitched spectrum and then into the physical DOEELS
layout. Individual low-loss electrons are distributed among four 192-column
ZLP read lanes; CoreLoss begins at raw column 768. Acceptance, dispersion,
quadratic dispersion, magnification, angle-position coupling, output angle,
energy blur, and detector PSF are configurable.

The current 0.257732 eV/column dispersion and column-52 ZLP position are
provisional values extracted from the 15 pA study, not an instrument energy
calibration. See [spectrometer.md](docs/spectrometer.md) for the equations,
end-to-end commands, assumptions, and measurements still needed.

## Specimen plugins and materialization

The specimen stage now has a replaceable Python plugin interface. Kernels can
be selected through the `eels_sim.specimen_kernels` package entry-point group
or directly as `package.module:object`. The built-in `identity` kernel is only
a no-sample pipeline test; it does not model scattering.

Outputs use one of three explicit semantics:

- `branch_probability`: mutually exclusive branches of the same incident
  electron are sampled categorically, producing at most one electron.
- `expected_electrons`: each weighted ray/bin is sampled independently with a
  Poisson count.
- `individual_electrons`: the kernel has already performed the stochastic
  draw; automatic materialization is an identity/passthrough operation.

The materializer emits unit-weight individual electrons, preserves
parent/branch/source lineage, and places a configurable upper bound on output
size. Its output feeds directly into the implemented spectrometer transfer.
See [specimen_plugins.md](docs/specimen_plugins.md) for the API, plugin
registration, configuration, commands, and sampling rules.

The built-in `abtem` kernel now supplies the first physical specimen adapter.
`config/abtem_nio.toml` runs a verified static-lattice elastic multislice
calculation for [001] rocksalt NiO and emits per-parent weighted angular
branches. Its O K transition-potential branch is enabled in the preset and
uses the native project-local GPAW environment created by
`scripts/setup_gpaw_env.sh`. See [abtem_adapter.md](docs/abtem_adapter.md) for
the exact model, install/run commands, metadata, deliberate first-version
limits, and the executable 14-case NiO convergence study defined in
`config/abtem_nio_convergence.toml`. The study now includes 14 calculations:
the one-at-a-time matrix plus a combined reference and larger/finer
confirmation, with separate metrics inside the 50 mrad spectrometer
acceptance.

The built-in `energy-resolved` kernel now supplies continuous low/core-loss
energies, plural low-loss scattering, angular kicks, and one individually
sampled output electron per incident electron. A GPAW-RPA bulk-Si low-loss
validation library and a DFT-GOSH atomic-cross-section
Li1.2Mn0.4Ti0.4O2 (LMTO) 1D demonstrator are included, along with generated
HDF5 sample spectra and plots. The GOSH baseline has absolute atomic core
cross sections but not material-specific ELNES. See
[energy_resolved.md](docs/energy_resolved.md) for the event model, exact
commands, provenance, and quantitative limitations, and
[exspy_gosh.md](docs/exspy_gosh.md) for the GOS provider.

The LMTO spatial demonstrator now turns those six edges into a complete
spectrum image, HAADF-like reconstruction, Li/Mn/Ti/O maps, and a
standalone element/focal-depth slider. It uses a synthetic disordered-rocksalt
volume and ideal model-component unmixing, while preserving integer
individual-electron statistics at every probe position. See
[lmto_scan.md](docs/lmto_scan.md) for the command, HDF5 contract, and the
important distinction between the useful focal-section demonstration and
experimentally justified depth recovery. The fast analytic preset is 64x64;
the more expensive abTEM reference used by the raw-frame chain is 16x16.

The physical spatial provider is also implemented. It caches an abTEM
multislice raster with annular HAADF and GPAW transition-potential maps for all
six LMTO edges, normalizes their relative channeling contrast with the GOSH
cross sections, and then reuses the ordinary spectrum sampler. The supplied
reference uses double channeling for Li K, Ti L2,3, O K, and Mn L2,3, plus a
separate 0.09 A HAADF refinement and sampling confirmation. The result shows
resolved Ti/Mn L-edge columns and the expected strong delocalization of Li K.

The spectrum image can now be rendered into one physical `960x3840` raw
DOEELS frame at every selected `(x,y)` probe coordinate. Physical-rate frames
sample a complete `7x7` charge-sharing template from one of 1,000
energy-matched Geant4 events for each electron (200 keV in the current demo;
300 keV in the earlier baseline), followed by the healthy or defective dark
calibration, correlated/read noise, and ADC. The output includes pre-readout
controls and raw-frame reconstructions so a lost element map can be attributed
to electron statistics, spectrometer mapping, or electronics. See
[raw_doeels.md](docs/raw_doeels.md) for commands, the HDF5 contract, and the
current calibration limits.

For realistic multi-readout dwell without retaining terabytes of frames, use
the nested streaming sweep:

```bash
PYTHONPATH=python python -m eels_sim simulate-raw-doeels-exposure-sweep \
  --config config/energy_lmto_abtem_200keV.toml \
  --spectrum-image output/lmto_abtem_200keV_spectrum_image.h5 \
  --output-hdf5 output/lmto_raw_doeels_200keV_exposure_sweep_16x16.h5 \
  --exposure 1 --exposure 10 --exposure 30 --exposure 100 \
  --exposure 300 --exposure 1000 --exposure 3000 --exposure 10000 \
  --exposure 30000 --exposure 100000
```

It combines the analog channel below the counting boundary with NiO-calibrated
sparse counting above 36.08 eV. The current 200 keV 16x16 sweep recovers
Mn/O/Ti with correlations of approximately 0.91/0.81/0.90 at 34.5 ms per
position while retaining no raw detector frames.

The 16x16 scan is the primary detector-reconstruction demonstration because it
shows the complete independently calculated abTEM field without spatial
replication. Build its 34.48 ms/position slider from exposure checkpoint 6:

```bash
PYTHONPATH=python python3 -m eels_sim build-raw-doeels-slider-demo \
  --input-hdf5 output/lmto_raw_doeels_200keV_exposure_sweep_16x16.h5 \
  --output-html output/lmto_raw_doeels_200keV_16x16_35ms_slider.html \
  --output-png output/lmto_raw_doeels_200keV_16x16_35ms_montage.png \
  --element Mn --element O --element Ti --exposure-index 6
```

A 64x64 tiled diagnostic is also reproducible end to end:

```bash
PYTHONPATH=python python3 -m eels_sim tile-lmto-abtem-spatial \
  --input-hdf5 output/lmto_abtem_spatial_refined.h5 \
  --output-hdf5 output/lmto_abtem_spatial_refined_64x64_periodic.h5 \
  --tile-factor 4
PYTHONPATH=python python3 -m eels_sim simulate-lmto-scan \
  --config config/energy_lmto_abtem_64x64.toml \
  --output-hdf5 output/lmto_abtem_spectrum_image_64x64.h5
PYTHONPATH=python python3 -m eels_sim simulate-raw-doeels-exposure-sweep \
  --config config/energy_lmto_abtem_64x64.toml \
  --spectrum-image output/lmto_abtem_spectrum_image_64x64.h5 \
  --output-hdf5 output/lmto_raw_doeels_64x64_35ms.h5 \
  --exposure 3000
PYTHONPATH=python python3 -m eels_sim build-raw-doeels-slider-demo \
  --input-hdf5 output/lmto_raw_doeels_64x64_35ms.h5 \
  --output-html output/lmto_raw_doeels_64x64_35ms_slider.html \
  --output-png output/lmto_raw_doeels_64x64_35ms_montage.png \
  --element Mn --element O --element Ti
```

At 87 kHz this tiled diagnostic is 141.24 s (2.354 min) of ideal scan time. Its
underlying 3.32 nm field is a periodic 4x4 expansion of the validated 16x16
abTEM cell; it is useful for load/scaling tests, but does not add structural
information. Detector noise is sampled independently while the atomic
arrangement repeats. See [lmto_64x64_demo.md](docs/lmto_64x64_demo.md) for
output interpretation.

## LMTO application-workflow movie

The revised 200 keV Manim demo under [`demo/`](demo/README.md) is designed for
a general technical audience. It starts with chip and battery end workflows,
then shows the explicit LMTO lattice, one realistic raw frame, construction of
an EELS spectrum, edge-to-map reconstruction, and DOEELS-based selection among
the HAADF/Mn/O/Ti views. Its central systems message is the combination of a fast advanced
camera with DAQIRI network-to-GPU acquisition and streaming GPU reduction.
Detailed abTEM/GOSH and detector derivations remain available as generated
assets rather than dominating the main story. The movie claims earlier
feedback and less data movement, but no numerical speedup until a matched
end-to-end benchmark is available.

```bash
demo/render_demo.sh assets
demo/render_demo.sh preview
demo/render_demo.sh full
```

## Detector calibration

The detector calibration stage is now executable:

```bash
# Generate statistically useful bare-silicon response libraries.
./scripts/run_geant4_local.sh macros/calibration_100keV.mac
./scripts/run_geant4_local.sh macros/calibration_300keV.mac

# Compare total charge and cluster width to backpropcount/sim.
PYTHONPATH=python python3 -m eels_sim compare-response \
  --legacy-100 /path/to/backpropcount/sim/EM_5um_front_100evts_100keV.pkl \
  --legacy-300 /path/to/backpropcount/sim/EM_5um_front_100evts_300keV.pkl \
  --new-100 output/calibration_100keV \
  --new-300 output/calibration_300keV \
  --config config/smoke.toml \
  --output calibration/single_electron_response.json

# Fit maps on frames 0:256 and validate on the held-out frames 256:512.
PYTHONPATH=python python3 -m eels_sim calibrate-dark \
  /path/to/doeels/nio_15pa_dark_frames_float32_uncompressed.h5 \
  --output calibration/eels_15pa_dark_calibration.h5 \
  --calibration-frames 256 \
  --validation-frames 256 \
  --known-defect 0:480,2272:2288

# Fit effective signal conversion and spreading from isolated NiO tail events.
PYTHONPATH=python python3 -m eels_sim calibrate-nio-sparse-detector \
  --spectrum-hdf5 /path/to/doeels/nio_15pa_spectrum_frames_float32_uncompressed.h5 \
  --dark-hdf5 /path/to/doeels/nio_15pa_dark_frames_float32_uncompressed.h5 \
  --readout-calibration calibration/eels_15pa_dark_calibration.h5 \
  --response-kernel calibration/monte_carlo_300keV_response_kernel.h5 \
  --output-hdf5 calibration/nio_15pa_sparse_detector_calibration.h5 \
  --output-png calibration/nio_15pa_sparse_detector_calibration.png
```

`config/eels_reference.toml` loads the resulting pedestal and temporal-noise
maps. It also uses the measured global, row, and column correlated-noise RMS
values. The map's total variance is reduced by those correlated components
before independent pixel noise is sampled, avoiding double counting.

The readout effects can be selected independently:

| Option | `true` | `false` |
|---|---|---|
| `use_calibrated_pedestal` | measured spatial pedestal map | scalar `pedestal_adu` |
| `use_calibrated_noise` | measured per-pixel temporal RMS | scalar `read_noise_adu` |
| `use_calibrated_correlated_noise` | global/row/column components | no correlated components |
| `simulate_known_defects` | original maps plus dead-channel signal loss | repaired healthy maps and full signal efficiency |

The generic `eels_reference.toml` preset leaves known defects off. The
`eels_dataset_reproduction.toml` preset turns them on to reproduce the DOEELS
acquisition. The dead ADC region is calibration metadata, not detector
geometry, so another calibration can define different regions or none.

The NiO tail fit gives an effective 0.275 ADC code per collected e-h pair and
0.50-pixel residual spreading, conditional on the current 5 um silicon model.
See [calibration.md](docs/calibration.md) and
[nio_sparse_detector_calibration.md](docs/nio_sparse_detector_calibration.md)
for results, limitations, and the measurements needed to separate sensor
thickness from electronics gain.

## Where the specimen model plugs in

The upstream interface is now an electron phase-space table containing one
outgoing electron per row: frame/electron ID, position, direction, kinetic
energy, time, and statistical weight. A specimen engine such as abTEM or
MULTEM can write that contract directly. The spectrometer maps unit-weight
individual electrons into the Geant4 source; weighted quantum outputs need an
explicit materialization/resampling step before detector transport.

See [architecture.md](docs/architecture.md) and
[data_contracts.md](docs/data_contracts.md) for the staged implementation plan.
