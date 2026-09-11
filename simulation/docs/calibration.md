# Detector calibration

## Single-electron response

The archived `backpropcount/sim` libraries contain pixelized charge from 100
nominal events at 100 and 300 keV. The new comparison uses 1,000 Geant4 events
at each energy followed by the new drift/diffusion model.

| Observable | Legacy 100 keV | New 100 keV | Legacy 300 keV | New 300 keV |
|---|---:|---:|---:|---:|
| Median e-h pairs | 1276.0 | 862.1 | 322.5 | 403.6 |
| New/legacy median | | 0.676 | | 1.251 |
| Median cluster RMS [pixels] | 1.073 | 0.214 | 0.897 | 0.177 |

The different signs of the energy-dependent charge discrepancy mean that a
single thickness or gain scale is not an adequate fit. The cluster-width
difference is expected: the archived code used an unbiased 1 um random walk,
while the new model uses a 20 V field, mobility saturation, and analytic
diffusion. The legacy geometry also contains SiO2, Si3N4, and aluminium layers,
whereas the current model is bare 5 um silicon. These models must be made
geometrically identical before treating the ratios as detector measurements.

Outputs:

- `calibration/single_electron_response.json`
- `calibration/single_electron_response.png`

The raw DOEELS renderer uses a separate 1,000-event response run from
`macros/detector_response_300keV.mac`. Its impact coordinates uniformly cover
one full pixel phase, avoiding the center/boundary bias of a narrow Gaussian
calibration beam. Each aligned 7x7 charge template is retained in
`calibration/monte_carlo_300keV_response_kernel.h5`; the current library has a
mean of 581.390 and median of 410 collected e-h pairs, with 13.44% of charge
outside the impact pixel on average. These values still describe the
provisional bare-5-um-silicon model, not a measured DOEELS sensor response.

## DOEELS dark/readout calibration

Input: `doeels/nio_15pa_dark_frames_float32_uncompressed.h5`, with the first
256 frames used for fitting and the final 256 used only for validation.

The DM4-derived values are exact multiples of 64. Calibration divides by 64,
recovering a 12-bit effective ADC code stored in a wider numeric type.

| Metric | Result [ADC code] |
|---|---:|
| Median pedestal | 2021.910 |
| Median calibration temporal RMS | 4.566 |
| Median held-out temporal RMS | 4.567 |
| Held-out pedestal-bias RMS | 0.883 |
| Global common-mode RMS | 0.155 |
| Row-correlated RMS | 0.372 |
| Column-correlated RMS | 0.818 |
| Valid pixels | 99.217% |

Pixels with calibration temporal RMS above 500 raw units, or 7.8125 ADC codes,
are marked invalid. There are 28,877 such pixels in the calibration half. This
is close to, but intentionally not forced to equal, the 30,944 blinkers found
when the previous analysis used all 512 frames.

The HDF5 product contains:

```text
/maps/pedestal_adu
/maps/read_noise_adu
/maps/healthy_pedestal_adu
/maps/healthy_read_noise_adu
/maps/signal_efficiency
/maps/validation_bias_adu
/maps/validation_noise_adu
/maps/valid_pixel_mask
/maps/known_defect_mask
/temporal/global_common_mode_adu
/temporal/row_common_mode_adu
/temporal/column_common_mode_adu
```

The known DOEELS dead ADC block is the half-open region rows `[0, 480)` and
columns `[2272, 2288)`, or 7,680 pixels. It is stored in the calibration file,
not hard-coded into sensor geometry. The healthy maps repair that block and
pixels rejected by the noise cut using row-wise interpolation between healthy
neighbors. With defect simulation enabled, the original pedestal/noise maps
are used and the known block has zero signal efficiency. With it disabled, the
repaired maps and unit signal efficiency are used.

Pedestal structure, per-pixel noise, correlated noise, and known defects are
separate switches in the `[readout]` configuration. Disabling a calibrated map
uses its scalar fallback; disabling correlated noise removes its configured
global/row/column components. `config/eels_reference.toml` is the healthy
default, while `config/eels_dataset_reproduction.toml` deliberately restores
the acquisition-specific defects.

The calibrated digitizer reproduces the marginal distribution of a held-out
dark frame: both have median 2025 codes; the simulated and measured spatial
standard deviations are 574.80 and 574.73 codes. This is a pipeline validation,
not an independent goodness-of-fit test, because the spatial pedestal and noise
maps came from the same dark acquisition.

## NiO sparse-tail response calibration

The lowest-current NiO spectrum stack also contains a usable low-occupancy
region away from the ZLP. Strict local maxima above an 8-sigma threshold are
extracted from that high-loss tail and statistically corrected with held-out
dark frames. A joint comparison of their peak amplitudes and 3x3 charge sums
against the Geant4 event library gives an effective conversion of **0.275 ADC
code per collected e-h pair** and **0.50 pixel** additional response spreading.

This is conditional on the current bare-5-um-silicon Geant4 model; physical
sensor thickness and conversion gain are not independently identified. It is
nevertheless a much stronger raw-frame preset than the former placeholder
gain, and it uses NiO only for detector response—not for LMTO specimen spectra.
See [nio_sparse_detector_calibration.md](nio_sparse_detector_calibration.md) for
the selection, fit diagnostics, command, and limitations.

## What remains uncalibrated or degenerate

- Physical pixel pitch, active thickness, dead-layer stack, collection-side
  convention, bias/electric-field profile, and temperature during acquisition;
  active thickness remains degenerate with the effective fitted gain.
- A clean, specimen-independent single-electron cluster distribution for the
  actual EELS sensor. The fitted extra blur currently absorbs response mismatch.
- Transfer of this dark calibration to another acquisition time and operating
  condition.
- Non-Gaussian blinker behavior, saturation/recovery, inter-pixel capacitance,
  and temporal correlations beyond global/row/column Gaussian components.

A low-occupancy, no-specimen beam run at both 100 and 300 keV, accompanied by
dark frames at the same detector settings, remains the cleanest next detector
measurement. It would refine charge-cloud width and operating-point dependence;
known sensor geometry or an independent electronics gain is still needed to
fully separate pair production from ADC conversion.
