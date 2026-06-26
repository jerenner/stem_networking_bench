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
- `diagnostics/`: narrower scripts used to investigate BLR artifacts,
  dark-frame recovery trends, single-frame samples, or local sanity checks.

## Invocation

Most scripts also have top-level wrappers for backwards-compatible commands such
as `python make_dark_frame.py ...` or `python compare_nio_stitch_study.py ...`.
When importing from Python, prefer the package/module paths:

```python
from stem_analysis import ProcessorConfig, process_tensor_block
```

Generated HDF5 files, CSVs, figures, Nsight profiles, and local scratch outputs
are intentionally ignored by the repository `.gitignore`.
