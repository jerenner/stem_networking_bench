# Raw DOEELS spectrum-image simulation

The spectrum-image demonstrator can now be rendered as one physical
`960 x 3840` ADC frame for every selected `(focal depth, scan y, scan x,
integration)` coordinate. This is a detector simulation, not an image-space
approximation: integer electrons are drawn from the local EELS spectrum,
mapped into the DOEELS column layout, transported through a reusable Geant4
single-electron response library, spread over detector pixels, and digitized
with the NiO-calibrated effective gain/spreading and pedestal/noise maps.

## Physics and sampling sequence

For each probe coordinate the renderer:

1. draws exactly `electrons_per_integration` outgoing electrons from the local
   energy-loss spectrum;
2. maps each energy through the configured dispersion, energy blur and
   dispersive point-spread, including the four ZLP lanes and subpixel
   interpolation;
3. draws the non-dispersive detector row from the configured beam profile;
4. samples an aligned `7 x 7` collected-charge template from one of 1,000
   300 keV Geant4 silicon events, whose impacts uniformly cover a full pixel,
   for every accepted electron, then applies the residual 0.50-pixel spreading
   fitted from isolated NiO clusters;
5. applies gain, healthy/defective signal efficiency, calibrated pedestal,
   independent and correlated noise, ADC rounding, and saturation; and
6. writes the raw frame before moving to the next scan position, so RAM use is
   approximately one detector frame rather than the full scan.

The response library retains the broad Monte Carlo distribution of deposited
charge (mean 581.390 and median 410 e-h pairs per incident 300 keV electron)
and event-to-event charge-sharing patterns. Runs above
`raw_doeels.exact_response_max_electrons` per frame use sampled total charge
with the ensemble-mean spatial template. This fallback is intended for
artificial high-dose diagnostics; the 2,152-electron 87 kHz preset uses exact
event templates.

## Commands

```bash
# Generate center/edge/corner responses with uniform subpixel impacts.
./scripts/run_geant4_local.sh macros/detector_response_300keV.mac

# Compress those events into the reusable response library.
PYTHONPATH=python python -m eels_sim build-detector-response-kernel \
  --config config/energy_lmto_abtem.toml \
  --geant4-base output/detector_response_300keV \
  --output-hdf5 calibration/monte_carlo_300keV_response_kernel.h5 \
  --radius-pixels 3

# Anchor effective signal gain/spreading to the lowest-current NiO data.
PYTHONPATH=python python -m eels_sim calibrate-nio-sparse-detector \
  --spectrum-hdf5 /path/to/doeels/nio_15pa_spectrum_frames_float32_uncompressed.h5 \
  --dark-hdf5 /path/to/doeels/nio_15pa_dark_frames_float32_uncompressed.h5 \
  --readout-calibration calibration/eels_15pa_dark_calibration.h5 \
  --response-kernel calibration/monte_carlo_300keV_response_kernel.h5 \
  --output-hdf5 calibration/nio_15pa_sparse_detector_calibration.h5 \
  --output-png calibration/nio_15pa_sparse_detector_calibration.png

# One 87 kHz integration at every x,y position in the middle focal section.
PYTHONPATH=python python -m eels_sim simulate-raw-doeels \
  --config config/energy_lmto_abtem.toml \
  --spectrum-image output/lmto_abtem_spectrum_image.h5 \
  --output-hdf5 output/lmto_raw_doeels_87khz_nio_calibrated.h5

PYTHONPATH=python python -m eels_sim plot-raw-doeels \
  --input-hdf5 output/lmto_raw_doeels_87khz_nio_calibrated.h5 \
  --output-png output/lmto_raw_doeels_87khz_nio_calibrated.png
```

`--depth-index`, `--scan-stride`, `--integrations-per-position`, and
`--electrons-per-integration` override the TOML values. Repeated integrations
are separate ADC reads and their reconstructed maps are summed at each scan
coordinate.

## Streaming exposure sweep

Retaining every high-rate frame is unnecessary when the goal is a spectrum
image. The compact sweep simulates nested exposure increments, immediately
projects each increment through the same detector-row estimator used by the
raw reconstruction, and retains only accumulated spectra, HAADF, component
maps, and diagnostics:

```bash
PYTHONPATH=python python -m eels_sim simulate-raw-doeels-exposure-sweep \
  --config config/energy_lmto_abtem.toml \
  --spectrum-image output/lmto_abtem_spectrum_image.h5 \
  --output-hdf5 output/lmto_raw_doeels_exposure_sweep_16x16.h5 \
  --exposure 1 --exposure 10 --exposure 30 --exposure 100 \
  --exposure 300 --exposure 1000 --exposure 3000 \
  --exposure 10000 --exposure 30000 --exposure 100000

PYTHONPATH=python python -m eels_sim plot-raw-doeels-exposure-sweep \
  --input-hdf5 output/lmto_raw_doeels_exposure_sweep_16x16.h5 \
  --output-png output/lmto_raw_doeels_exposure_sweep_16x16.png
```

The checkpoints are nested: the 3,000-frame result contains the same first
1,000 integrations as the 1,000-frame result plus 2,000 new integrations.
No `960x3840` frames are written. The ten-checkpoint 16x16 product is 12.7 MB,
versus hundreds of terabytes for the equivalent maximum-exposure raw stream.

Low-occupancy increments sample individual projected Geant4 event templates.
Larger increments use their compound mean/covariance, preserving response
fluctuations without materializing hundreds of millions of electrons. Readout
noise and ADC-rounding variance scale with the number of independent frames.
Per-frame saturation is neglected; the one-frame calibrated scan found only
one saturated pixel in approximately 944 million pixel reads.

The reconstruction is hybrid. Data below the measured counting-validity
boundary remain in the analog/integrating channel. At and above 36.08 eV, the
counter applies the NiO-fitted 85.7% detection efficiency and adds the measured
0.7227 dark events/readout in the far tail before subtracting their mean. The
dark spectrum between 36 and 350 eV is not independently measured, so the Li K
result is optimistic with respect to false counts there. Spatial pile-up is
neglected because the current model has only about 1.8 countable events spread
over roughly 1.66 million valid imaging pixels per readout.

For the current 1.66 nm LMTO model, counted-map correlations at selected dwell
times are:

| Dwell | Li | Mn | O | Ti |
|---:|---:|---:|---:|---:|
| 3.45 ms | -0.01 | 0.47 | 0.45 | 0.47 |
| 11.49 ms | 0.02 | 0.71 | 0.66 | 0.68 |
| 34.48 ms | 0.23 | 0.87 | 0.84 | 0.87 |
| 114.94 ms | 0.31 | 0.96 | 0.94 | 0.96 |
| 1.149 s | 0.67 | 1.00 | 0.99 | 1.00 |

Mn, O, and Ti are already suitable for a demonstration near 35 ms/position.
Li remains difficult because its expected spatial modulation in this structure
is only about 0.7%, despite a relatively large total Li K count.

## HDF5 contract

Schema attribute: `eels-sim-raw-doeels-scan-v1`.

| Dataset | Shape | Meaning |
|---|---|---|
| `/frames/raw` | `[frame,960,3840]` | simulated ADC codes |
| `/scan/frame_*_index` | `[frame]` | focal depth, x, y, and integration lookup |
| `/truth/incident_spectrum_counts` | `[frame,energy]` | sampled specimen-output electrons |
| `/truth/raw_column_hit_counts` | `[frame,3840]` | accepted electrons before silicon, optional |
| `/truth/generated_charge_pairs` | `[frame]` | collected charge before readout |
| `/truth/expected_component_counts` | `[frame,component]` | expected source mixture at the requested dose |
| `/reconstruction/pre_readout_energy_counts` | `[frame,energy]` | spectrum after ideal column folding |
| `/reconstruction/energy_counts` | `[frame,energy]` | spectrum estimated from the raw ADC frame |
| `/reconstruction/*component_counts` | `[frame,component]` | signed model-component fits |
| `/reconstruction/scan_maps/elements/*` | `[depth,y,x]` | expected, pre-readout, and raw-ADC element maps |

Signed coefficients are deliberate. Clipping negative noisy fits to zero
creates a large positive bias for weak edges and can make an unusable map look
plausible.

## What the NiO-calibrated result says

The stage-by-stage products separate a successful physics transfer from a
detectability problem. The fitted 0.275 ADC-code/e-h-pair gain makes an
individual high-loss electron comfortably measurable relative to the 4.6-code
median read noise, consistent with the actual NiO counting study. The complete
16x16 physical scan has no material saturation (one saturated pixel among
roughly 944 million pixel reads).

It does **not**, however, create core-loss electrons. At 2,152 incident
electrons per 11.49 us scan position, the whole 256-position simulation expects
only 10.5 Ti L2,3, 17.0 O K, and 7.2 Mn L2,3 events. Their per-position maps are
therefore shot-noise dominated even before electronics, and the raw-ADC map
correlations are consistent with zero. The Li K channel has about 259 expected
events, but its predicted spatial contrast is only a few percent and is also
not recovered in one integration per position. A demonstration elemental map
must accumulate many independent short integrations or use a longer valid
dwell; putting millions of electrons into one synthetic ADC read is not an
equivalent substitute because it changes pile-up and saturation.

The effective response is now anchored to data, but this is not yet a firm
instrument sensitivity forecast. Detector active thickness/dead layers remain
degenerate with gain, and the row profile, dispersion, and point-spread
function are still provisional. A realistic sample thickness and acquisition
dwell/repetition plan are the next inputs needed for an honest LMTO map-SNR
prediction. See
[nio_sparse_detector_calibration.md](nio_sparse_detector_calibration.md) for
the detector-fit limitations.
