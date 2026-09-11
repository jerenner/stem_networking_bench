# 64x64 LMTO detector-reconstruction demonstration

This optional tiled/scaling demonstration produces a 64x64 STEM scan with four presentation
channels: HAADF, Mn, O, and Ti. The elemental images are reconstructed from
simulated spectra after spectrometer transfer, the Monte Carlo-derived silicon
charge-cloud response, calibrated electronic noise, and the hybrid
analog/sparse-counting estimator. They are not direct renderings of the input
atomic labels.

The primary structural demonstration remains the independently calculated
16x16 field. This 64x64 variant is retained to test acquisition duration,
streaming accumulation, and data volume; it should not be presented as a
larger independently simulated specimen region.

## Acquisition represented

- Field of view: 33.2 A (3.32 nm) square.
- Probe raster: 64x64, or 0.51875 A per probe step.
- Selected focal section: source depth index 1.
- Readout rate: 87 kHz.
- Integrations per probe position: 3,000.
- Dwell per probe position: 34.4828 ms.
- Incident electrons per integration: 2,152.
- Incident electrons per probe position: 6,456,000.
- Ideal scan time: 141.241 s (2.354 min), excluding scan overhead.

The compact streaming file does not retain the 12,288,000 individual
960x3840 detector frames. It draws the same nested incident-electron,
spectrometer, Monte Carlo event-response, counting, and electronics statistics
and retains their sufficient statistics, accumulated spectra, component fits,
HAADF counts, and element maps. This is the intended practical route for a
multi-minute acquisition.

## Current result

At 34.4828 ms/position, the counted reconstruction correlations with the
noise-free expected maps are:

| Channel | Pearson correlation |
| --- | ---: |
| Mn L2,3 | 0.882 |
| O K | 0.820 |
| Ti L2,3 | 0.876 |

The lower analog-only correlations are expected in this current setup because
readout noise accumulates over 3,000 frames. Above the 36.08 eV calibration
boundary, the displayed elemental channels use the NiO-calibrated sparse
counting efficiency and dark-event correction.

## Spatial-response caveat

The expensive abTEM input remains the validated 16x16 calculation: a 2x2
lateral LMTO supercell, 0.09 A refined HAADF potential sampling, a 50--80 mrad
HAADF detector, and double-channeling transition potentials for the higher
energy edges. For this demonstration it is repeated periodically 4x4 to obtain
64x64 positions over 3.32 nm while preserving the validated 0.51875 A probe
sampling.

Consequently, the atomic disorder and ideal abTEM contrast repeat every 16
pixels. Shot noise, detector response, dark false events, and readout noise are
sampled independently at every position. A future native large-cell abTEM run
would remove the repeated structure but is not required to validate the
detector and reconstruction chain.

## Outputs

- `output/lmto_abtem_spatial_refined_64x64_periodic.h5`: tiled abTEM response
  and replicated atom table, with provenance and periodicity metadata.
- `output/lmto_abtem_spectrum_image_64x64.h5`: complete energy-resolved source
  spectrum image at all three focal sections.
- `output/lmto_raw_doeels_64x64_35ms.h5`: compact detector simulation and
  reconstructed spectra/maps at the middle focal section.
- `output/lmto_raw_doeels_64x64_35ms_slider.html`: interactive discrete
  HAADF/Mn/O/Ti channel slider.
- `output/lmto_raw_doeels_64x64_35ms_montage.png`: static four-channel view.

The HTML and PNG independently scale each channel's display contrast and apply
0.6-pixel periodic Gaussian smoothing. Quantitative values and unsmoothed maps
remain in the HDF5 file.
