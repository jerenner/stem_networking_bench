# Python Script Layout

The reusable processing math lives in `stem_analysis/`. Scripts here are entry
points built on top of that package or focused diagnostics from the detector
studies.

## Main Offline Pipeline

Use the top-level compatibility wrapper from the repository root:

```bash
python run_offline_pipeline.py raw_frames.h5 processed_frames.h5 --input-dataset /frames
```

The implementation is `scripts/offline/run_offline_pipeline.py`. It is the
canonical CPU/NumPy reproduction of the `PyTorchProcessorOp` correction chain.

## Directory Roles

- `conversion/`: raw data conversion tools, especially DM4 to uncompressed HDF5.
- `dark/`: dark-frame creation and dark-frame quality-control plotting.
- `offline/`: HDF5 replay-style processing and processed-frame plotting.
- `studies/`: manifest-driven studies that combine multiple files/currents and
  write publication-style summary products.
  `run_nio_threshold_study.py` sweeps ImageJ-style positive thresholds before
  column-spectrum accumulation.
  `run_nio_counting_study.py` applies the full correction chain, calibrates a
  STEMPy standard electron-counting threshold against processed dark frames,
  identifies the sparse counting-valid spectral region from measured event
  occupancy, and compares tail fluctuations with the true `1/sqrt(N)` count
  statistic. It requires the optional `stempy` package and is exposed through
  the top-level `run_nio_counting_study.py` compatibility wrapper.
  `apply_dead_adc_spectrum_correction.py` reconstructs the known dead top-half
  ADC block at columns 2272..2287 after row aggregation, using a calibrated
  bottom-half spectral contribution and preserving reconstruction uncertainty.
  `analyze_tail_noise.py` compares corrected and thresholded tail fluctuations
  with counting-statistics expectations across the beam-current study.
- `diagnostics/`: narrower scripts used to investigate BLR artifacts,
  dark-frame recovery trends, single-frame samples, or local sanity checks.
  `plot_raw_frame_qc.py` writes per-frame TIFF figures that combine a raw 2D
  frame view with ZLP/CoreLoss BLR-row baseline diagnostics.
  `plot_nio_counted_frame_qc.py` verifies representative STEMPy-counted EELS
  frames against the saved frame/channel matrix and writes corrected-frame,
  event-density, per-frame-spectrum, zoomed event-overlay, and lossless binary
  event-mask views.
  `plot_nio_counted_poisson_band.py` overlays the counted tail with a local
  smooth expectation and its `mu +/- sqrt(mu)` (equivalently relative
  `1 +/- 1/sqrt(mu)`) band. This is a visual complement to, not a replacement
  for, the independent batch-fluctuation test.
  `plot_nio_counted_full_analog.py` compares the complete final analog spectrum,
  including its folded ZLP, with the counting-valid CoreLoss-only STEMPy result.
  `analyze_coreloss_features.py` localizes persistent CoreLoss structures such
  as the dead ADC block and separates detector artifacts from repeatable spectral
  features across files and beam currents.

## Invocation

Most scripts also have top-level wrappers for backwards-compatible commands such
as `python make_dark_frame.py ...` or `python compare_nio_stitch_study.py ...`.
When importing from Python, prefer the package/module paths:

```python
from stem_analysis import ProcessorConfig, process_tensor_block
```

Generated HDF5 files, CSVs, figures, Nsight profiles, and local scratch outputs
are intentionally ignored by the repository `.gitignore`.
