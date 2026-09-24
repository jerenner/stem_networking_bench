# LMTO + DAQIRI combined demonstration storyboard

Target: approximately 75 seconds, silent, 16:9, 1080p, 30 frames/s. On-screen
captions carry the complete story; narration can be added later without
changing the scientific claims.

## 00:00-00:05 - Promise

**Visual:** Structure, Mn, O, and Ti outputs beside the title.

**Narration:** "DOEELS combines structure and chemistry in one scan. This
demonstration follows the electrons all the way from an LMTO specimen through
a fast detector and GPU-native acquisition to elemental maps."

## 00:05-00:15 - Microscope concept

**Visual:** Electron pulses pass through scan coils and converge onto a colored
LMTO lattice. The sample, annular HAADF detector, EELS entrance aperture,
magnetic prism, and silicon camera form one vertical optical path, as in the
collaborator's setup sketch. HAADF scattering branches to a structural signal
at the side. Transmitted electrons pass through the aperture and prism. The
highest-energy ZLP path continues straight onto the left of the silicon
sensor; low-loss electrons bend slightly, while higher-loss CoreLoss electrons
bend farther right. The camera, DAQIRI, and live-product boxes form one row
below the prism.

**Narration:** "A focused electron probe crosses the lattice at each raster
position. High-angle scattering produces the structural HAADF signal. The
DOEELS path uses a magnetic spectrometer to spread transmitted electrons by
energy onto a fast silicon camera."

## 00:15-00:22 - Raster and dose

**Visual:** The simulated 16 by 16 probe grid and acquisition facts. The 3,000
readouts are explicitly labeled as single detector frames.

**Narration:** "The probe visits 256 positions. At each point, three thousand
single-frame detector readouts represent about 35 milliseconds and more than
six million incident electrons. Both measurements remain registered to that
position."

## 00:22-00:33 - Detector packets become a frame

**Visual:** Eight source lanes carry animated tile packets into DAQIRI. RX0-3
are grouped and colored as FPGA 0; RX4-7 are grouped and colored as FPGA 1. A
real simulated DOEELS frame fills as drawn in the collaborator's tiling sketch:
phase 1 completes the leftmost 192-column ZLP read while contiguous CoreLoss
bands grow four tile rows inward from both the top and bottom. Phase 2
completes the second ZLP read and adds the next two inward bands, and so on.
Each phase therefore adds one complete ZLP read and one quarter of CoreLoss.
The ZLP uses tall, narrow tiles while CoreLoss uses short, wide tiles.

**Narration:** "The detector is read through eight sources. Each source carries
120 equal-payload tiles. FPGA 0 supplies the top half and FPGA 1 the bottom.
Together they place 192 tall, narrow ZLP tiles and 768 short, wide CoreLoss
tiles into one GPU-resident frame."

## 00:33-00:43 - Acquisition and analysis overlap

**Visual:** A continuous path connects single camera frames, DAQIRI assembly,
a frame-stack/bucket stage, GPU correction and counting, spectral accumulation,
and map updates. A side branch retains selected HDF5 validation bursts.

**Narration:** "The high-rate path stays on the GPU. Pedestal correction,
electron-event counting, spectral accumulation, and map updates can overlap
acquisition. Selected raw bursts remain available for validation rather than
requiring every readout to move through an offline workflow."

## 00:43-00:51 - Spectrum at one point

**Visual:** The reconstructed low-loss and counted core-loss spectrum, with Ti,
O, and Mn edge positions. The noisy counted CoreLoss trace is rebinned into
approximately 4 eV energy bins for display; counts are summed, not discarded.

**Narration:** "Many corrected readouts form one spectrum at each probe point.
Fitting the titanium, oxygen, and manganese edge amplitudes supplies one value
per element for that position."

## 00:51-01:02 - Elemental maps update with the raster

**Visual:** HAADF, Mn, O, and Ti maps reveal one probe position at a time in the
simulation order: `x` scans left-to-right, flies back, and `y` advances to the
next row.

**Narration:** "As raster rows complete, the structural and elemental views can
update together. This is the key workflow capability: acquisition and
interpretation form one continuous path while the sample is still present."

## 01:02-01:12 - Select chemistry from the same scan

**Visual:** A larger registered output cycles through structure, Mn, O, and Ti,
with the corresponding measurement or edge fit identified.

**Narration:** "The operator can inspect the overall structure and then select
an element. These are fitted EELS-edge amplitudes, not colors painted onto atom
labels, and all four views come from the same scan."

## 01:12-01:17 - Close

**Visual:** The full chain is restated from specimen to live map.

**Narration:** "Advanced cameras provide the signal. DAQIRI and GPU-native
processing turn it into earlier, actionable chemical feedback."

## Claim boundaries

- The LMTO lattice, raw detector pixels, spectrum, and final maps come from the
  existing 200 keV physics simulation.
- The 3,000 readouts at each probe position are 3,000 individual detector
  frames, accumulated into frame stacks/buckets for processing.
- The packet animation reflects the native target geometry: eight sources with
  120 tiles each; 192 `128 x 32`-pixel ZLP tiles plus 768 `32 x 128`-pixel
  CoreLoss tiles; 960 equal-payload packets per detector frame. It is a
  schematic, not a recorded packet trace.
- The target sequencing shown groups RX0-3 under FPGA 0 and RX4-7 under FPGA 1.
  The two halves fill from their outer edges inward in contiguous bands; each
  phase completes one ZLP read and one quarter of CoreLoss. This follows the
  supplied sketch, not a measured packet-arrival trace.
- The temporary legacy-transmitter compatibility path still receives
  3,840-sample row-shaped payloads, discards offsets 120-127, and duplicates a
  256-sample prefix before scattering into this tile geometry. Its current
  compact `source_ordinal * 120 + row_offset` mapping is linear and does not
  encode the target FPGA interleave shown in the movie; production tile identity
  must be reconciled before live use.
- The CoreLoss display uses count-preserving energy rebinning only. It does not
  change the simulated electron statistics or the edge-fit inputs.
- The map animation follows the simulation writer's nested `y`, then `x`,
  loops: unidirectional left-to-right fast scans with flyback between rows.
  Hardware may instead be configured for bidirectional/serpentine scanning.
- The map reveal illustrates streaming accumulation. Its animation duration is
  not a benchmark or latency measurement.
- No numerical speedup is claimed. A matched end-to-end workflow benchmark is
  still required.
- HAADF and DOEELS are synchronized but separate detector branches.
- Element channels are reconstructed EELS-edge amplitudes, not atom labels.
- The GOSH baseline does not include LMTO-specific oxidation state, bonding,
  coordination, or ELNES.
