# STEM DAQIRI NiO Demo Storyboard

Target: 58-65 seconds for the silent draft, 16:9, 1080p, 30 frames/s. A later
narrated cut can extend holds toward 70-80 seconds. The first draft uses
on-screen captions.

## Scientific and performance claims

- The NiO images are recorded detector data replayed through the validated
  NumPy mirror of the runtime correction chain. They are not represented as a
  live network capture.
- Packet flow and tiling are schematic. Rates shown in the acquisition scene
  come from the configured measured run and are labeled as measured examples.
- The DualEELS/tiled-readout scene compares acquisition architecture only. It
  does not assert a numerical speedup before matched timing measurements exist.
- Intermediate images use the operation order implemented by the runtime:
  float32 conversion, dark subtraction, grouped BLR, static valid-pixel mask,
  and dynamic two-sided half-column mask.
- The event display is deliberately thinned and latest-product-oriented. It
  does not imply that every detector frame is sent to DigitalMicrograph.

## 00:00-00:12 - Acquisition and assembly

**Visual:** Eight 100 GbE lanes feed packet glyphs into an IGX/GPU block. A
detector frame fills by tiles, then joins a stack labeled `1 / 128` through
`128 / 128`.

**Caption:** "UDP detector tiles arrive directly in GPU memory."

**Counters:** Configured line rate, measured aggregate input, packet rate, and
assembled frame rate. Counters must say "example measured run".

**Narration draft:** "The DAQ receives independent detector streams over up
to eight 100-gigabit Ethernet interfaces. Packet headers place each tile into
GPU-resident frames, and 128 consecutive frames form the natural processing
bucket."

## 00:12-00:20 - Dynamic-range acquisition strategy

**Visual:** Side-by-side paths compare a conventional DualEELS low-loss
exposure, energy-range switch, CoreLoss exposure, and alignment/splicing with
the continuous tiled readout. The tiled detector fills four repeated ZLP
regions and the CoreLoss region before GPU assembly.

**Caption:** "Different readout rates; full detector rows remain available
before reduction."

**Narration draft:** "Conventional DualEELS acquires optimized low-loss and
CoreLoss exposures in rapid succession. The tiled design instead streams four
repeated zero-loss regions and the CoreLoss region through one region-aware
readout, retaining the detector rows for correction before reduction. This is
an architectural comparison; a numerical speedup has not yet been measured."

## 00:20-00:34 - GPU correction chain

**Visual:** The representative NiO frame transforms through four recorded-data
assets: raw, dark-subtracted, BLR-corrected, and fully corrected. A mask overlay
briefly highlights static bad pixels and dynamic blinkers before they disappear.

**Caption rail:** `RAW -> DARK -> BLR -> MASKED`.

**Narration draft:** "One fused GPU path converts samples to float, subtracts
the dark reference, estimates the baseline from detector edge rows, and removes
static and dynamically detected outliers. The displayed examples are generated
by the same ordered mathematics used by the online processor."

## 00:34-00:45 - Scientific reduction

**Visual:** The 128 corrected frames collapse into a bucket sum. Rows integrate
into a one-dimensional detector profile; the four repeated ZLP reads fold into
one physical ZLP region, followed by CoreLoss. The view changes from linear to
logarithmic scale.

**Caption:** "128 corrected frames -> one NiO EELS spectrum".

**Narration draft:** "The corrected bucket can be retained frame by frame or
reduced. Summing detector rows produces the EELS spectrum. Repeated zero-loss
  measurements are folded and shown beside the CoreLoss region for inspection
  over the full dynamic range."

## 00:45-00:53 - Outputs and control loop

**Visual:** The GPU pipeline splits. One branch writes selected full buckets to
HDF5. The other sends one representative frame and the bucket sum over ZeroMQ
to DigitalMicrograph. A return arrow carries start, stop, visualization, and
burst commands back to the persistent DAQ supervisor.

**Caption:** "Full bursts when requested. Thin live products continuously."

**Narration draft:** "Burst mode writes selected processing stages to HDF5.
In parallel, a low-latency ZeroMQ stream publishes only a representative frame
and bucket sum. DigitalMicrograph displays those products and returns runtime
controls over a separate request-reply channel."

## 00:53-00:59 - Summary

**Visual:** The complete path is restated from detector packets through GPU
assembly, corrections, EELS products, and DigitalMicrograph.

## Review checklist

- Confirm whether the displayed 70.38 Gb/s counter is appropriate for the
  audience and test configuration.
- Stitching is disabled by default. Enable it only with an independently
  established no-BLR calibration; do not estimate the gain directly from the
  grouped-BLR demo bucket.
- Replace draft captions with approved terminology for microscope and detector
  hardware.
- Add narration only after the silent animation is scientifically approved.
