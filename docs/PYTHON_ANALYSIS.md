# Python Offline Analysis

This directory contains the CPU/NumPy analysis code used to reproduce and study
the STEM Holoscan processing chain without live network hardware. The intent is
to keep the core processing math auditable and reusable, while keeping specific
case studies separate.

## Main Package

`stem_analysis/` is the reusable package. It is the best place to look for the
canonical analysis implementation.

- `stem_analysis.config.ProcessorConfig` mirrors the processor knobs used by
  `PyTorchProcessorOp`.
- `stem_analysis.processing.process_tensor_block()` mirrors the runtime
  operation order:
  `float32 conversion -> dark subtraction -> grouped BLR -> valid-pixel mask -> dynamic half-column mask -> output`.
- `stem_analysis.dm4` contains DM4 loading and frame-stack normalization helpers.
- `stem_analysis.hdf5` contains small HDF5 path/dataset helpers.
- `stem_analysis.spectra` contains ZLP folding and collapsed ZLP/CoreLoss
  spectrum construction.
- `stem_analysis.stitch` contains the current stitch calibration strategy:
  robust no-BLR log-quadratic ZLP/CoreLoss gain fitting, followed by cubic
  Hermite repair of transition columns `191..193`.
- `stem_analysis.plotting` contains shared plotting/cache and detector-region
  utilities.

## Canonical Offline Processor Runner

Use `run_offline_pipeline.py` when you want to run the same processing sequence
as the Holoscan processor on an HDF5 frame stack. The implementation lives at
`scripts/offline/run_offline_pipeline.py`; the top-level file is a compatibility
wrapper that keeps imports simple.

Example:

```bash
python run_offline_pipeline.py raw_frames.h5 processed_frames.h5 \
  --input-dataset /frames \
  --output-dataset /processed \
  --frames-per-tensor 128 \
  --noop true \
  --subtract-dark-frame true \
  --dark-frame-path dark_frame.h5 \
  --dark-frame-dataset /processed \
  --apply-valid-pixel-mask true \
  --valid-pixel-mask-dataset /valid_pixel_mask \
  --apply-blr-correction true \
  --apply-dynamic-half-column-mask true \
  --dynamic-mask-median-window-pixels 31 \
  --dynamic-mask-threshold-ratio 1.0 \
  --dynamic-mask-threshold-offset 500.0 \
  --dynamic-mask-excluded-edge-rows 32 \
  --dynamic-mask-two-sided true
```

The module form also works from the repository root:

```bash
python -m scripts.offline.run_offline_pipeline raw_frames.h5 processed_frames.h5 ...
```

## Dark Frame Creation

Dark-frame tools live under `scripts/dark/`.

- `make_dark_frame.py` builds a blinker-aware averaged dark frame from HDF5
  frames.
- `make_dark_frame_from_dm4.py` builds the same dark-frame layout directly from
  DM4 files.
- `plot_dark_frame.py` creates dark-frame QA plots.

The dark-frame builder writes:

- `/processed`: repaired dark frame, stored as `[1, rows, cols]`.
- `/raw_dark_mean`: raw mean before blinker repair.
- `/dark_stddev`: per-pixel temporal standard deviation.
- `/blinker_mask`: pixels whose temporal dark stddev exceeded threshold.
- `/valid_pixel_mask`: pixels treated as valid by runtime masking.

For detailed dark-frame and runtime-mask semantics, see
[`DARK_FRAME_WORKFLOW.md`](DARK_FRAME_WORKFLOW.md).

## Manifest-Driven Studies

Study scripts live under `scripts/studies/`. They consume raw data manifests and
write result folders outside the repository tree.

- `run_nio_current_study.py` runs the full current-sweep spectrum analysis.
- `compare_nio_current_study.py` compares current-sweep outputs.
- `run_nio_stitch_study.py` builds stitch-study summaries for each current.
- `compare_nio_stitch_study.py` fits stitch gains and writes final stitched
  spectrum plots/tables.
- `analyze_spectrum_dm4.py` streams one current's spectrum DM4 files into final
  spectrum summaries.
- `analyze_stitch_dm4.py` streams one current's DM4 files into no-BLR/grouped-BLR
  stitch summaries.

Current stitch policy:

1. Build no-BLR calibration spectra after dark subtraction and static masking.
2. Collapse the four repeated 192-column ZLP reads into one 192-column ZLP.
3. Fit a robust log-quadratic curve over ZLP columns `160..190` and CoreLoss
   columns `194..223`, excluding transition columns `191..193`.
4. Apply the fitted no-BLR gain to final grouped-BLR spectra.
5. Fill columns `191..193` using a cubic Hermite segment anchored at columns
   `190` and `194`, with slopes estimated from `184..190` and `194..200`.

## Diagnostic Scripts

Scripts under `scripts/diagnostics/` are intentionally narrower. They document
case studies we used to understand detector behavior, BLR artifacts, dark-frame
recovery trends, and single-frame BLR samples. They may still be useful, but they
should not be treated as the main processing pipeline.

For a concise index of script directories and intended entry points, see
[`../scripts/README.md`](../scripts/README.md).

## Generated Files

Generated HDF5 files, figures, CSV summaries, Nsight profiles, and local study
outputs are ignored by `.gitignore`. Keep data/result directories outside the
repo or under ignored paths when running studies locally.
