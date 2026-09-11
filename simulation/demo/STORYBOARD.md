# DOEELS application-workflow storyboard

Current preview duration: **1:51.96**, 16:9, 30 frames/s. The movie is silent
and uses complete on-screen captions; the narration below is optional. Times
are rounded because transitions overlap section boundaries.

## Audience and narrative rule

The audience does not need to know how to operate an EELS microscope. Every
technical detail should answer one of three questions:

1. What is scanned?
2. How does one detector stream become a spectrum and then a chemical map?
3. What new workflow becomes practical with a fast camera and GPU-native DAQ?

abTEM, GOSH, Geant4, and the NiO calibration establish provenance. Their
specialist derivations remain in the repository rather than interrupting the
main story.

## 00:00-00:05 — Promise

**Visual:** Structure, Mn, O, and Ti views beside the title.

**Narration:** “DOEELS lets us see structure and chemistry in the same scan.
This demonstration follows a 200-kiloelectronvolt beam from a model battery
material through a fast detector and GPU-native analysis to elemental maps.”

## 00:05-00:12 — End application

**Visual:** Clearly labeled conceptual chip cross-section and battery particle.

**Narration:** “The practical question is simple: select a transistor,
particle, or interface, scan it, and ask where each element is. The two drawings
are workflow concepts; the physics demonstration that follows uses an LMTO
battery-cathode cell.”

## 00:12-00:18 — One scan, several views

**Visual:** Structure, Mn, O, and Ti reconstructions side by side.

**Narration:** “The structural image shows the larger pattern. The other views
select manganese, oxygen, or titanium from the energy lost by the electrons.
All four come from the same raster acquisition.”

## 00:18-00:28 — Explicit LMTO lattice

**Visual:** The 128-atom 3D cell, composition and dimensions, followed by all
eight atomic planes.

**Narration:** “Our demonstrator is a synthetic periodic
Li1.2Mn0.4Ti0.4O2 random-cation rocksalt cell. It contains 128 explicit atoms
in eight planes across a volume 0.83 nanometres wide and 1.66 nanometres thick.
It is not a relaxed or experimentally measured grain.”

## 00:28-00:34 — Probe raster and dose

**Visual:** The independent 16 x 16 probe grid over the simulated structure,
plus beam and dwell values.

**Narration:** “A focused 200-keV, 30-picoamp probe visits 16 by 16 positions.
Three thousand short readouts correspond to 34.5 milliseconds and 6.456
million incident electrons at each position.”

## 00:34-00:41 — Two synchronized measurements

**Visual:** Beam, specimen, HAADF branch, DOEELS branch, and fast silicon
camera.

**Narration:** “High-angle scattered electrons form the HAADF structural
signal. The DOEELS branch sends the energy-loss distribution through a magnetic
spectrometer to a fast pixelated silicon camera. They are separate detectors,
synchronized to the same raster point.”

## 00:41-00:50 — Raw detector data to spectrum

**Visual:** One simulated 960 x 3840 raw readout before and after pedestal
subtraction, then the five-step reduction rail.

**Narration:** “A camera readout is a noisy pixel image, not an element map.
The GPU corrects pedestal and noise, reduces detector rows, folds repeated
zero-loss lanes, and accumulates many short frames into a calibrated histogram
of counts versus energy loss.”

## 00:50-00:57 — Readable EELS spectrum

**Visual:** Integrated zero-/low-loss signal and a separately scaled counted
core-loss region with Ti, O, Mn, and combined fitted components.

**Narration:** “The strong left-hand feature is the zero-loss and low-loss
signal. The enlarged high-loss region contains much rarer individual electron
events. Known edge energies identify the elements. At this manganese-rich
probe, the fit assigns about 515 counts to the manganese edge, so the signal is
present even though it is noisy.”

**Interpretation note:** Core-loss edges are onsets and extended profiles, not
necessarily narrow line peaks. The earlier movie plotted them against the huge
low-loss scale, which made valid counts appear absent. This panel uses the
counted-data scale and overlays the fitted components.

## 00:57-01:03 — Spectrum to elemental maps

**Visual:** Raster -> fit 456/532/640 eV edges -> Mn/O/Ti images.

**Narration:** “At every probe point we fit the titanium, oxygen, and manganese
edge amplitudes. Repeating that measurement over the raster turns each fitted
amplitude into one image pixel.”

## 01:03-01:10 — Fast camera plus GPU-native DAQ

**Visual:** Camera -> DAQIRI network-to-GPU -> GPU correction/counting -> live
spectra and maps, with a branch for selective HDF5 bursts.

**Narration:** “The advanced camera creates many short readouts. DAQIRI keeps
the high-rate path on the GPU, where correction, event counting, spectral
accumulation, and map updates can happen during acquisition. Selected raw bursts
can still be saved for validation.”

## 01:10-01:16 — Workflow value

**Visual:** Offline acquire/write/move/analyze/map sequence versus the
overlapped streaming sequence.

**Narration:** “The benefit is time-to-map, not a data-rate number by itself.
Streaming removes unnecessary movement and gives earlier feedback while the
sample is still available. A matched end-to-end speedup remains to be measured,
so the movie makes no numerical speedup claim.”

## 01:16-01:23 — Dose feedback

**Visual:** Nested exposure sweep and the selected 35-ms checkpoint.

**Narration:** “Because the map updates as counts accumulate, the experiment
can show when the result becomes usable. In this simulation, manganese,
oxygen, and titanium reach correlations of 0.91, 0.81, and 0.90 near 35
milliseconds per position.”

## 01:23-01:35 — Element selection

**Visual:** The display cycles through Structure, Mn, O, and Ti while the image
and fitted-edge annotation change.

**Narration:** “The intended interface is deliberately simple: inspect the
overall structure, then select an element. The channels are not painted atom
labels; each is reconstructed from the corresponding energy-loss signature in
the same acquisition.”

## 01:35-01:41 — Return to the workflow

**Visual:** Chip and battery concepts return.

**Narration:** “The same output model can be applied to a transistor layer or a
battery interface: switch the selected element while staying registered to the
structural image.”

## 01:41-01:49 — Provenance and limits

**Visual:** What is modeled versus what still needs measured instrument or
material data.

**Narration:** “This is a physics-based demonstration with measured NiO
pedestal and noise, but not yet a calibrated prediction for the final
instrument. Material-specific chemical fine structure, measured spectrometer
calibration, confirmed sensor geometry, and a matched performance benchmark
remain future inputs.”

## 01:49-01:52 — Close

**Visual:** Sample -> fast camera -> DAQIRI and GPU -> spectrum -> element map.

**Narration:** “The result is one continuous path from fast detector data to
useful chemical maps.”

## Scientific claim boundaries

- The 200 keV GOSH, abTEM, Geant4, calibration, raw-frame, and exposure-sweep
  products were recalculated; this is not a label-only change from 300 keV.
- The 16 x 16 maps cover one independently calculated 0.83 nm abTEM field. No
  4 x 4 spatial tiling is used in the principal demonstration.
- GOSH supplies independent-atom edge shapes and cross sections, not
  LMTO-specific oxidation state, bonding, coordination, or ELNES.
- The spectrometer is a provisional first-order mapping, not a magnetic-field
  solve or measured optical calibration.
- HAADF and DOEELS are simultaneous branches but different detectors. HAADF is
  not passed through the DOEELS silicon response.
- Geant4 uses a provisional bare 5 micrometre silicon sensor. Active thickness
  and gain remain partly degenerate in the NiO fit.
- The compact 35-ms sweep statistically represents 3,000 readouts per point and
  retains spectra and sufficient statistics, not all 768,000 raw frames.
- No depth recovery is claimed. Separate focal sections are separate simulated
  acquisitions.

## External-review checklist

- Confirm the actual convergence aperture, EELS collection aperture, HAADF
  annulus, dispersion, and point-spread function for the 200 keV microscope.
- Confirm the active silicon thickness, dead layers, pixel pitch, and absolute
  ADC conversion with detector documentation or a dedicated calibration.
- Measure an offline and streaming workflow end to end before inserting a
  quantitative speedup.
- Decide whether the final public version should name DAQIRI alone or also show
  the specific camera and transport hardware.
- Replace the conceptual chip/battery drawings with registered measured or
  simulated larger-scale datasets when those become available.
- Add FEFF or another material-specific ELNES source before discussing valence,
  oxidation state, bonding, or coordination.
