# NiO sparse-tail detector calibration

The existing 15 pA NiO acquisition provides both pieces needed for a first
data-anchored detector model:

- the dark stack constrains pedestal, temporal noise, correlated noise, and
  acquisition-specific bad pixels; and
- the low-occupancy high-loss tail of the spectrum stack contains isolated
  electron clusters from which an effective charge-to-ADC conversion and
  residual spatial spreading can be inferred.

This deliberately does **not** use the NiO spectrum as a specimen model. The
tail supplies detector-event shapes and amplitudes only; LMTO edge energies and
rates still come from the energy-resolved specimen calculation.

## Event selection

The calibration works directly from
`doeels/nio_15pa_spectrum_frames_float32_uncompressed.h5` and the corresponding
dark HDF5 stack. Stored values are divided by the measured storage LSB of 64.
For every frame, the pedestal map is removed and a separate per-column median
is subtracted in each detector half. Candidate events are strict eight-neighbor
local maxima in rows `[32,928)` and raw columns `[1986,3840)`, using the larger
of the pixel's `8 sigma` threshold and a global `8 sigma` floor. Events above
823 ADC codes are rejected as X-ray-like outliers. Only isolated 7x7 patches
are retained.

The first 256 spectrum frames give 22,401 isolated candidates. The held-out
256 dark frames give 1,414 candidates under the same direct processing, an
estimated 6.31% background fraction before histogram subtraction. The more
mature columnwise-BLR counting study, which also applies its dynamic blinker
mask, finds a much lower 0.42% dark fraction in the far tail and only 1.11 times
Poisson excess. That is the better evidence that the tail is countable; the
new direct path keeps the calibration reproducible and removes its residual
dark histogram statistically.

The global threshold is 37.80 ADC codes. Dark-subtracted single-event peak
quantiles at probabilities `[0.10,0.25,0.50,0.75,0.90]` are
`[44.18,52.82,70.60,103.69,173.34]` ADC codes. Corresponding 3x3 patch-sum
quantiles are `[69.98,89.36,122.01,186.35,328.36]` ADC codes.

## Fit to the Monte Carlo response

The response fit samples the 1,000-event, uniformly subpixel-phased 300 keV
Geant4 library. It jointly compares thresholded peak-amplitude and 3x3
patch-sum quantiles while adding measured read noise. Two parameters are fitted:

| Parameter | Fit |
|---|---:|
| Effective gain | 0.275 ADC code / collected e-h pair |
| Additional Gaussian spreading | 0.50 detector pixel |
| Modeled response above the 8-sigma threshold | 85.7% |

A diagnostic objective tolerance gives a gain range of 0.245--0.3025 ADC code
per pair; it is not a statistical confidence interval. At the best fit, modeled
peak quantiles are `[45.75,55.41,71.79,104.77,166.45]`, and modeled 3x3 sums
are `[75.37,94.26,122.96,176.26,288.27]` ADC codes.

The result is an **effective calibration conditional on the current bare 5 um
silicon Geant4 model**. Sensor active thickness/dead layers determine how many
pairs an electron produces, while ADC conversion determines how those pairs
become codes; this acquisition alone cannot separate those effects. The fitted
0.50-pixel spreading similarly absorbs physical diffusion, interpixel coupling,
and geometry/model mismatch.

## Reproduce

```bash
PYTHONPATH=python python -m eels_sim calibrate-nio-sparse-detector \
  --spectrum-hdf5 /path/to/doeels/nio_15pa_spectrum_frames_float32_uncompressed.h5 \
  --dark-hdf5 /path/to/doeels/nio_15pa_dark_frames_float32_uncompressed.h5 \
  --readout-calibration calibration/eels_15pa_dark_calibration.h5 \
  --response-kernel calibration/monte_carlo_300keV_response_kernel.h5 \
  --output-hdf5 calibration/nio_15pa_sparse_detector_calibration.h5 \
  --output-png calibration/nio_15pa_sparse_detector_calibration.png
```

The HDF5 output retains all selected spectrum/dark event patches, empirical
histograms, the dark-subtracted mean cluster, fit metadata, and recommended
simulation parameters. The LMTO raw-frame preset now applies the fitted gain
and 0.50-pixel response blur.

## Valid use and remaining measurement

Sparse counting is supported in the high-loss tail, and the previous study
placed its conservative usable boundary at approximately 36 eV loss. It must
not be extrapolated into the ZLP/low-loss region, where multiple electrons
overlap; those pixels remain an integrating measurement in the simulation and
analysis.

A dedicated no-specimen, low-occupancy acquisition at documented detector
settings is still valuable. It would remove specimen/scattering ambiguity,
provide cleaner clusters across the sensor, and—together with known active
geometry or an independent gain measurement—break the thickness/gain
degeneracy. It is an improvement to this calibration, not a prerequisite for
using the current NiO-anchored effective model.
