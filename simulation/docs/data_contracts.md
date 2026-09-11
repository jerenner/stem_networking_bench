# Data contracts

All coordinates are right-handed. The silicon sensor is centered at the origin,
the electron beam travels in `+z`, and charge is currently collected at the
`+z` face. Pixel row corresponds to `y`; column corresponds to `x`.

## Specimen-to-spectrometer phase space (implemented HDF5)

Schema attribute: `eels-sim-phase-space-v1`

Group: `/electrons`

| Field | Type | Units | Meaning |
|---|---|---|---|
| `frame_id` | uint64 | 1 | Integration window |
| `electron_id` | uint64 | 1 | Stable truth identifier |
| `x_um`, `y_um`, `z_um` | float64 | um | Exit position at specimen plane |
| `dir_x`, `dir_y`, `dir_z` | float64 | 1 | Unit direction vector |
| `kinetic_energy_eV` | float64 | eV | Energy after the specimen |
| `time_ns` | float64 | ns | Time relative to frame start |
| `weight` | float64 | electrons | Statistical weight; 1 for analog events |
| `loss_channel` | int32 | enum | ZLP, phonon, plasmon, core edge, other |

Specimen kernels may also add `parent_electron_id` (uint64), `branch_id`
(uint32), `spectral_component_id` (uint16), and `plural_order` (uint16).
Materialization adds `source_record_id` (uint64) and assigns new, contiguous
individual-electron IDs.

Weighted records allow an expensive quantum calculation to produce a
probability distribution. The implemented materializer uses categorical
sampling for mutually exclusive per-parent branch probabilities, or Poisson
sampling for independent expected-electron weights.
The current spectrometer-to-Geant4 path intentionally requires `weight == 1`;
it fails rather than silently treating a statistical weight as one electron.

## Spatial spectrum image (implemented HDF5)

Schema attribute: `eels-sim-spectrum-image-v1`

The LMTO scan product stores `/spectrum/counts` as
`[focal_depth,scan_y,scan_x,energy]` uint32 electron counts. Coordinate and
energy axes are under `/axes`; `/structure/atoms` retains generated atomic
truth; `/truth` retains noiseless spatial responses; and `/reconstruction`
contains HAADF and Li/Mn/Ti/O maps. `/spectrum/component_total_counts` and
`/spectrum/component_profiles` keep the latent component draw and spectral
bases used to form the full energy cube. `/spectrum/expected_component_counts`
stores the pre-sampling component expectation so downstream dose sweeps do not
inherit frozen Monte Carlo noise from this source file.

Each spectrum sums exactly to the configured `electrons_per_probe`. Focal
depth indexes separate synthetic acquisitions, not a depth coordinate inferred
from one conventional 2D scan. Full details and limitations are in
[lmto_scan.md](lmto_scan.md).

The optional reusable abTEM input uses schema
`eels-sim-abtem-spatial-v1`. It stores annular-detector fractions as
`/spatial/haadf_fraction`, raw transition-potential output as
`/spatial/raw_edge_intensity`, and GOSH-normalized local atom densities as
`/spatial/edge_response_per_A2`, all indexed by focal depth and scan position.
The root metadata records edge order, subshell support, channeling choices,
potential grids, detector apertures, and refinement provenance.

## Spectrometer-to-detector entries (implemented HDF5)

Schema attribute: `eels-sim-detector-entry-v1`

Accepted records are stored in `/electrons`. The original identifiers and
phase-space index are retained alongside the transferred values:

| Field | Type | Units | Meaning |
|---|---|---|---|
| `event_id` | uint64 | 1 | Contiguous Geant4 event index |
| `frame_id`, `electron_id` | uint64 | 1 | Upstream integration/truth identifiers |
| `source_index` | uint64 | 1 | Row in the specimen-exit phase-space table |
| `x_um`, `y_um`, `z_um` | float64 | um | Detector-entry position |
| `dir_x`, `dir_y`, `dir_z` | float64 | 1 | Unit detector-entry direction |
| `kinetic_energy_eV` | float64 | eV | Physical post-specimen energy |
| `energy_loss_eV` | float64 | eV | Reference energy minus kinetic energy |
| `mapped_energy_loss_eV` | float64 | eV | Loss after configured energy blur |
| `time_ns`, `weight`, `loss_channel` | mixed | ns/1/enum | Propagated truth |
| `stitched_column` | float64 | pixel | Continuous spectrum before raw layout |
| `raw_column`, `raw_row` | float64 | pixel | Continuous physical detector coordinate |
| `zlp_lane` | int16 | enum | ZLP read lane, or -1 for CoreLoss |

`parent_electron_id`, `branch_id`, `source_record_id`,
`spectral_component_id`, and `plural_order` are propagated when present so the
detector-entry HDF5 remains traceable to the weighted kernel record, sampled
spectral process, and incident electron.

`/diagnostics` contains every input `frame_id`, `electron_id`, and a rejection
code. Root attributes contain the transfer configuration and acceptance
summary as JSON. Integer pixel coordinates denote pixel centers.

The optional Geant4 CSV export carries the accepted fields needed by the C++
particle source. Its `event_id` must be contiguous from zero and all weights
must equal one.

## Geant4 transport truth (implemented CSV ntuples)

The output base `output/name` produces:

- `name_nt_run_info.csv`: rows, columns, pitch, thickness, and maximum step.
- `name_nt_primaries.csv`: transport event ID, upstream frame/electron IDs,
  position, direction, kinetic energy, time, weight, and loss channel.
- `name_nt_deposits.csv`: per-step event/track IDs, process type/subtype,
  position, time, deposited energy, and pre-step kinetic energy.
- `name_nt_events.csv`: total deposited energy and step count per primary.

Geant4 event ID is a contiguous transport index. The digitizer uses propagated
frame IDs when present and falls back to fixed-size grouping for legacy or
macro-generated sources.

## Digitized HDF5 (implemented)

Schema attribute: `eels-sim-digitized-v1`

| Dataset | Type | Shape | Units |
|---|---|---|---|
| `/frames/raw` | uint16 | `[frame,row,column]` | ADU |
| `/frames/analog_adu` | float32 | same | ADU, optional |
| `/frames/charge_electrons` | uint32 | same | collected electrons, optional |

The physical DOEELS frame preset is `[N,960,3840]`. The receiver currently
uses `[N,1024,3840]`; the additional 64 rows should be an explicit adapter or
padding operation rather than part of the physical sensor model.

Configuration values and charge-conservation totals are stored in `/metadata`
and root attributes. Later versions should add `/truth/primaries` and a sparse
truth-to-frame association without changing `/frames/raw`.

## NiO sparse detector calibration HDF5 (implemented)

Schema attribute: `eels-sim-nio-detector-calibration-v1`

The `/measured` group retains selected 7x7 spectrum and held-out-dark event
patches, the dark-subtracted empirical mean patch, peak/3x3 histograms, and
event counts per frame. The root `summary_json` stores selection thresholds,
dark-subtracted response quantiles, the fitted effective gain and residual
spreading, sensitivity ranges, recommended simulation values, and caveats.
The fit is conditional on the referenced Monte Carlo response kernel rather
than an independent measurement of physical sensor thickness.

## Raw DOEELS scan HDF5 (implemented)

Schema attribute: `eels-sim-raw-doeels-scan-v1`

This product connects a spatial spectrum image directly to the physical
DOEELS layout. `/frames/raw` is `[frame,960,3840]` uint16 ADC data and
`/scan/frame_depth_index`, `frame_y_index`, `frame_x_index`, and
`frame_integration_index` map each frame back to its probe coordinate.
`/truth` stores sampled energy counts, optional pre-silicon column hits,
generated charge, saturation, and expected spectral-component counts.

`/reconstruction` contains both a pre-readout control and a reconstruction
from the noisy ADC data. Its `/scan_maps/elements/<element>` groups store
`expected_counts`, `pre_readout_counts`, and `detector_counts` on
`[depth,y,x]`. Component fits retain negative values so readout-limited weak
edges are not biased upward by clipping. See [raw_doeels.md](raw_doeels.md)
for the complete chain and commands.

## Raw DOEELS exposure sweep HDF5 (implemented)

Schema attribute: `eels-sim-raw-doeels-exposure-sweep-v1`

The leading axis is a nested exposure checkpoint. `/axes` stores integrations
per position, dwell time, energy, focal-depth, and scan coordinates. `/truth`
stores cumulative incident spectra, accepted/rejected electrons, and expected
component counts. `/reconstruction` stores pre-readout, analog raw-ADC, and
hybrid counted spectra/components on
`[exposure,depth,y,x,energy-or-component]`, plus accumulated HAADF counts.
Element groups contain expected, pre-readout, integrating, and counted maps
with their correlations. There is deliberately no `/frames/raw` dataset.
