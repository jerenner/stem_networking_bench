# Spectrometer transfer

The spectrometer MVP maps an individual-electron specimen-exit phase-space
table onto the physical detector plane. It is a first-order paraxial transfer,
not a detailed charged-particle field solve. The coefficients are explicit so
that measured ray data, an optical design, or a higher-order transfer map can
replace the provisional values without changing the phase-space or detector
interfaces.

## Transfer model

For a reference beam energy `E0`, each electron has energy loss
`dE = E0 - Ekin` and paraxial input angles
`theta_x,y = 1000 * dir_x,y / dir_z` in mrad. The continuous stitched spectral
column is

```text
c = c0 + dE / dispersion + q*dE^2
    + (Mx*x + Ax*theta_x) / pixel_pitch + energy/PSF blur
```

and the non-dispersive detector row is

```text
r = r0 + (My*y + Ay*theta_y) / pixel_pitch + PSF blur.
```

The output angles have independent position, input-angle, and energy terms.
This permits a general first-order position/angle coupling in both transverse
planes. The implementation then applies, in order:

1. forward-going and collection-semiangle acceptance;
2. configured energy-loss acceptance;
3. energy dispersion, optional quadratic dispersion, and optional Gaussian
   energy/point-spread blur;
4. the detector's ZLP/CoreLoss raw-column layout;
5. detector row/column geometric acceptance.

Every rejected electron receives a diagnostic reason. Its physical kinetic
energy is never changed by energy-resolution blur; blur only perturbs where it
focuses on the detector.

## Four ZLP reads

DOEELS raw columns `[0,768)` contain four 192-column reads of the same ZLP
range. CoreLoss starts at raw column 768. The transfer first computes a
continuous stitched coordinate. For a low-loss electron it randomly chooses
one read lane using a reproducible seed:

```text
raw_column = stitched_column + lane * 192, lane in 0..3
```

For CoreLoss it inserts the three additional ZLP lane widths:

```text
raw_column = stitched_column + 3 * 192.
```

Thus a zero-loss peak at stitched column 52 appears at raw columns 52, 244,
436, and 628; folding and summing the lanes recovers one continuous spectrum.
An electron is assigned to one lane, never duplicated. This is a provisional
model of the repeated-read acquisition behavior and should be checked against
the detector/spectrometer documentation.

## Current DOEELS reference values

The reference preset uses 300 keV, a ZLP center at stitched column 52, and a
provisional dispersion of 0.257732 eV/column. That dispersion came from
assigning the first broad post-ZLP feature in the current data to 25 eV; the
DM4 files do not contain a valid energy calibration. The 100 um/mrad
non-dispersive angular mapping, 50 mrad collection semiangle, magnifications,
output-angle mapping, and zero blur are likewise starting parameters, not
instrument measurements.

## Commands

The diagnostic generator is only an interface/transfer test. It does not
claim to simulate a specimen:

```bash
PYTHONPATH=python python3 -m eels_sim generate-phase-space \
  --config config/eels_reference.toml \
  --output output/spectrometer_smoke_phase_space.h5 \
  --electrons 100 --frames 2 \
  --losses-eV 0,25,100 --fractions 0.5,0.3,0.2 \
  --position-sigma-um 0.2 --angular-sigma-mrad 5

PYTHONPATH=python python3 -m eels_sim transfer-spectrometer \
  output/spectrometer_smoke_phase_space.h5 \
  --config config/eels_reference.toml \
  --output output/spectrometer_smoke_detector_entries.h5 \
  --geant4-csv output/spectrometer_smoke_source.csv

./scripts/run_geant4_local.sh macros/spectrometer_smoke.mac

PYTHONPATH=python python3 -m eels_sim digitize \
  output/spectrometer_smoke_deposits \
  --config config/eels_reference.toml \
  --output output/spectrometer_smoke_frames.h5 \
  --keep-intermediate
```

The source CSV length must match `/run/beamOn` in the Geant4 macro. The C++
source rejects nonunit weights and propagates frame/electron truth IDs into its
primary ntuple.

## Calibration still required

- energy dispersion and ZLP reference versus spectrometer settings and drift;
- energy resolution/PSF, including its energy dependence;
- horizontal and vertical position-angle transfer coefficients;
- collection aperture shape, alignment, and vignetting;
- higher-order aberrations and detector-plane distortions;
- the exact physical meaning and timing of the four ZLP reads.

A calibration acquisition with the zero-loss peak at several known energy
offsets and controlled beam tilts/positions would constrain these terms far
more directly than the present specimen data.
