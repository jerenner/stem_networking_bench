# Dark Frame Workflow

This note documents the dark-frame calibration path used by the STEM networking
bench application. It covers two pieces:

1. Building a blinker-aware dark frame from a saved HDF5 frame stack.
2. Applying the dark frame and optional valid-pixel mask in the Holoscan PyTorch
   operator.

## Motivation

The detector can produce unstable pixels that are extremely bright at random
times. In the ImageJ analysis these are referred to informally as "blinkers".
They should not contribute to the dark frame because a simple average would
encode transient bright events into the calibration image. They also should not
be treated as valid detector pixels in downstream analysis unless a later
per-frame rejection method explicitly repairs them.

The current implementation handles blinkers during dark-frame construction by
using the temporal standard deviation of each pixel across the dark stack. It
also exports a mask that can be applied at runtime after dark subtraction.

## Source ImageJ Logic

The closest ImageJ reference is `doeels/Dark_processing_v3.ijm`.

That macro:

1. Creates an average-intensity projection of the dark stack.
2. Creates a standard-deviation projection of the same dark stack.
3. Flags pixels with temporal standard deviation above `THRESHOLD = 500`.
4. For every flagged pixel, searches vertically in the same column for nearby
   pixels whose temporal standard deviation is below the threshold.
5. Uses `NEIGHB = 10` accepted same-column neighbors to replace the flagged
   pixel value in the average dark frame.
6. Keeps the top and bottom detector halves separate during this search.
7. Avoids the outer `EDGE = 32` rows, which are treated as non-image / BLR
   metadata-pattern rows.

The same repository also contains `doeels/Create_Mask_Average_Outliers.ijm`.
That macro is different: it creates a mask from the stack average after dark and
BLR subtraction by looking for spatial outliers in each column and detector
half. Because it is based on an average image, it is not a true per-frame random
blinker rejection method. It is better understood as a persistent spatial
outlier mask.

## Building The Dark Frame

The implemented dark-frame builder is:

```bash
python make_dark_frame.py \
  sparse_frames_out_net.h5 \
  walking_dot_dark_frame.h5 \
  --input-dataset /processed \
  --output-dataset /processed \
  --frames 256 \
  --chunk-size 16
```

For a real dark calibration, the input file should contain dark frames acquired
with no signal. For integration tests, a walking-dot file can be used to exercise
the mechanics, but the resulting dark frame and mask should not be interpreted
as a physically meaningful dark calibration.

### Inputs

`make_dark_frame.py` expects an HDF5 dataset shaped either:

```text
[frames, rows, cols]
```

or:

```text
[rows, cols]
```

The normal pipeline output dataset is `/processed`, with shape
`[frames, 1024, 3840]` when saving raw receiver frames.

### Statistics

For each pixel, the script computes:

```text
mean = sum(pixel_values) / N
stddev = sqrt(mean(pixel_values^2) - mean^2)
```

The accumulation is chunked over frames so the full input stack does not need to
be loaded at once. The accumulator uses `float64`; stored output datasets use
`float32` for image-valued products and `uint8` for masks.

### Blinker Detection

Pixels are flagged as blinkers when:

```text
dark_stddev[row, col] > blinker_std_threshold
```

The default threshold is `500.0`, matching the ImageJ macro. This threshold is
appropriate as a first pass because it preserves the prior analysis behavior,
but it should be revisited with real dark acquisitions.

For walking-dot integration tests, the dot path will create high temporal
variance and may flag many pixels as blinkers. That is expected and useful for
testing the machinery, but it does not mean those pixels are true dark-current
blinkers.

### Blinker Repair

For each flagged pixel, the script repairs the dark-frame value using the same
same-column neighbor idea as ImageJ:

1. Decide whether the pixel is in the top half or bottom half of the image.
2. Search upward and downward from the flagged row in the same column.
3. Do not cross the middle row separating detector halves.
4. Do not search into the outer `edge_rows` rows.
5. Accept only neighbors whose temporal stddev is below the blinker threshold.
6. Continue until `repair_neighbors` accepted pixels are found or both search
   directions hit their limits.
7. If enough neighbors are found, replace the dark-frame pixel with their
   average.
8. If not enough neighbors are found, leave the raw mean in `/processed`, but
   still mark the pixel invalid in `/valid_pixel_mask`.

The defaults are:

```text
blinker_std_threshold = 500.0
repair_neighbors = 10
edge_rows = 32
```

These match the relevant ImageJ constants:

```text
THRESHOLD = 500
NEIGHB = 10
EDGE = 32
```

### Output Datasets

The dark-frame HDF5 file contains:

```text
/processed
```

Blinker-repaired average dark frame, stored as `[1, rows, cols]` float32. This
is the dataset loaded by the PyTorch operator for dark subtraction.

```text
/raw_dark_mean
```

Raw average before blinker repair, stored as `[1, rows, cols]` float32.

```text
/dark_stddev
```

Per-pixel temporal standard deviation of the dark stack, stored as
`[1, rows, cols]` float32.

```text
/blinker_mask
```

Mask where `1` means the pixel exceeded the temporal stddev threshold and was
flagged as a blinker. Stored as `[1, rows, cols]` uint8.

```text
/valid_pixel_mask
```

Mask where `1` means the pixel is considered valid at runtime and `0` means it
was flagged as a blinker. Stored as `[1, rows, cols]` uint8.

The `/processed` dataset also stores attributes documenting the build, including
the number of frames averaged, source file, source dataset, blinker threshold,
repair-neighbor count, edge rows, and the number of flagged/repaired pixels.

## Plotting Dark Frames

Use `plot_dark_frame.py` to make a quick overview figure:

```bash
python plot_dark_frame.py \
  walking_dot_dark_frame.h5 \
  --output walking_dot_dark_frame_overview.png
```

The plot includes:

1. The repaired dark frame from `/processed`.
2. The temporal standard-deviation image from `/dark_stddev`.
3. The blinker mask from `/blinker_mask`.
4. Metadata and summary statistics from the HDF5 file.

The image panels use `imshow(..., interpolation="none")` so individual detector
pixels are not smoothed.

## Runtime Dark Subtraction And Masking In PyTorch

The runtime implementation is in `cpp/pytorch_processor_op.cpp`.

Relevant configuration keys:

```yaml
processor:
  noop: true
  subtract_dark_frame: true
  dark_frame_path: "/path/to/dark_frame.h5"
  dark_frame_dataset: "/processed"
  apply_valid_pixel_mask: true
  valid_pixel_mask_dataset: "/valid_pixel_mask"
  apply_blr_correction: true
  blr_rows: 30
  blr_zlp_width: 768
  blr_zlp_group_columns: 4
  blr_core_group_columns: 16
  apply_dynamic_half_column_mask: true
  dynamic_mask_median_window_pixels: 31
  dynamic_mask_threshold_ratio: 1.0
  dynamic_mask_threshold_offset: 500.0
  dynamic_mask_excluded_edge_rows: 32
  dynamic_mask_two_sided: true
```

### Loading

At operator initialization:

1. If `subtract_dark_frame` is true, the operator opens `dark_frame_path`.
2. It reads `dark_frame_dataset`.
3. The dataset must be shaped `[rows, cols]` or `[1, rows, cols]`.
4. The dark frame is read as float32 and copied once to a CUDA tensor.
5. If `apply_valid_pixel_mask` is true, the same file is opened for
   `valid_pixel_mask_dataset`.
6. The mask is read as float32 and copied once to a CUDA tensor.

The mask is expected to contain `1` for valid pixels and `0` for invalid pixels.

### Per-Batch Compute

For every incoming frame tensor:

1. The receiver tensor is wrapped as a PyTorch CUDA tensor without copying.
2. If any correction is enabled, the processor uses the fused correction path
   for both network `uint16` tensors and HDF5 replay `float32` tensors. This is
   important: HDF5 replay and live receiver data execute the same correction
   sequence once they enter the processor.
3. The fused path writes a corrected float32 working tensor. Conceptually, the
   first step is conversion plus optional dark subtraction:

   ```text
   corrected = frame_batch.float32
   if subtract_dark_frame:
       corrected = corrected - dark_frame
   ```

4. If BLR correction is enabled, grouped edge-row baselines are computed per
   frame, detector half, and ZLP/CoreLoss column group from the dark-subtracted
   values. These baselines are then subtracted from all rows in the matching
   detector half and column group.
5. If dynamic half-column masking is enabled, the fused path also computes the
   batch mean image from the corrected values. The later mask kernel compares
   each pixel against a local same-column median within the top or bottom
   detector half. Pixels whose mean differs from that median by more than the
   configured threshold are zeroed in every frame of the batch.
6. If valid-pixel masking is enabled, invalid static-mask pixels are also zeroed
   in every frame of the batch.
7. If `noop` is true, the full corrected float32 batch is emitted. If `noop` is
   false, the corrected batch is summed along the frame dimension and one
   `[rows, cols]` float32 image is emitted.

The conceptual operation order is therefore:

```text
float32 conversion
-> dark subtraction
-> grouped BLR baseline estimation/subtraction
-> static valid-pixel mask
-> dynamic half-column mask
-> noop/pass-through or sum-output mode
```

The implementation fuses some of these steps for performance, but the offline
Python analysis in `stem_analysis.processing.process_tensor_block()` follows the
same math and ordering.

### Performance Nuance

The current corrected-frame path materializes a full corrected float32 batch.
For a `[128, 1024, 3840]` batch, that is a large tensor. The fused CUDA kernels
reduce the correction sequence to one full-size output write plus the supporting
small baseline/mean products:

```text
uint16 or float32 batch
-> fused convert/subtract/BLR/batch-mean
-> optional static/dynamic mask
-> corrected float32 batch
-> optional sum
```

This is the right representation if later analysis truly needs individual
dark-subtracted frames. It may still be too expensive for analyses that only need
a reduced output, because writing the full corrected batch is itself expensive.
A baseline sum-only path can keep up because it writes less intermediate data:

```text
uint16 batch -> float32 batch -> sum
```

For analyses whose final output is a sum, the algebraically equivalent optimized
path is:

```text
sum(frames - dark) = sum(frames) - N * dark
sum((frames - dark) * mask) = (sum(frames) - N * dark) * mask
```

That avoids materializing the full corrected batch and only applies dark
subtraction/masking to the reduced `[rows, cols]` image. This optimization has
not been implemented yet because the final analysis may require per-frame
operations before reduction.

For future nonlinear per-frame analysis, the preferred direction is likely a
fused GPU operation that performs:

```text
load uint16 -> subtract dark -> apply mask -> compute feature/reduction
```

without writing a full intermediate corrected batch.

## Current Limitations

The dark-frame builder handles temporally unstable pixels in the dark stack.

The runtime now has two mask mechanisms:

1. A static valid-pixel mask loaded from the dark-frame HDF5 file.
2. A dynamic half-column mask based on the batch mean image.

The dynamic mask catches pixels that are anomalous in the mean of the current
batch. It is still not a fully per-frame repair algorithm: a one-frame transient
can be diluted by the batch mean, while a persistent or repeated outlier is much
more likely to be caught. If future analysis needs single-frame blinker repair,
that should be added as a separate correction step.

The same dark frame and mask are currently applied to all inputs handled by the
processor operator. If receiver streams require separate dark calibrations, the
pipeline will need either separate processor instances or metadata-aware routing.
