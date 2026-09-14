# DOEELS application-workflow demo

This directory builds a reproducible Manim movie showing how a fast pixelated
camera and GPU-native DAQ turn one STEM raster into a structural image and
element-selective maps. The main story is intended for a general technical
audience: chip and battery workflows first, enough microscope physics to make
the result credible, and implementation detail only where it explains the new
capability.

The revised demo uses **200 keV throughout**. The GOSH library, abTEM spatial
response, Geant4 silicon response, NiO-constrained detector fit, raw frame, and
16 x 16 exposure sweep were recalculated rather than relabeled from the earlier
300 keV version. The older files and movie remain available for comparison.

## The message

The instrument produces many short detector readouts at each probe position.
The proposed DAQIRI path receives those data on the GPU, corrects and counts
electron events, accumulates a spectrum, and updates elemental maps during the
scan. Selective raw HDF5 bursts can still be retained when needed.

This enables earlier feedback and avoids making a large raw-file round trip the
only workflow. The movie deliberately does **not** claim a numerical speedup:
a matched end-to-end measurement against the offline workflow has not yet been
made.

## Scientific provenance

| Quantity shown | Source in this demonstration |
| --- | --- |
| LMTO lattice | Synthetic, periodic, unrelaxed 128-atom Li1.2Mn0.4Ti0.4O2 random-cation rocksalt cell |
| Beam and scan | 200 keV, 30 pA, 87 kHz, 16 x 16 positions, 0.51875 Angstrom step |
| Probe propagation and spatial edge contrast | abTEM multislice and transition potentials |
| Core-loss edge profiles and cross sections | eXSpy with DFT-GOSH |
| HAADF | abTEM annular signal scaled to dose and Poisson sampled as a separate detector branch |
| Spectrometer | Provisional first-order transfer into the 960 x 3840 detector geometry |
| Silicon response | 1,000 Geant4 200 keV events followed by charge collection and diffusion |
| Pedestal and electronics noise | Measured NiO dark frames |
| Effective gain, residual blur, and counting efficiency | NiO sparse-tail events fitted against the 200 keV Geant4 response |
| Final Mn/O/Ti maps | Detector-simulated 16 x 16 exposure sweep at 34.48 ms per position |

NiO is used only for detector and electronics calibration. Its specimen physics
and EELS spectrum are not transferred into the LMTO model.

At the selected 3,000-readout dwell, the model uses 6.456 million incident
electrons per probe position and gives map correlations of 0.91 for Mn, 0.81
for O, and 0.90 for Ti against the expected maps. These are simulated
correlations, not measured instrument performance.

## What the audience sees

1. A chip cross-section and a battery particle as representative end workflows.
2. Structure, Mn, O, and Ti views from a single scan.
3. The LMTO composition, all 128 atoms, and all eight explicit atomic planes.
4. The 200 keV probe raster, current, dwell, and electrons per position.
5. Separate HAADF structural contrast and DOEELS energy-loss measurements.
6. One realistic noisy raw camera frame and the short reduction chain that
   converts it into a spectrum.
7. A readable low-loss and counted core-loss spectrum. The Mn-rich example has
   about 515 fitted Mn-edge counts, so the Mn contribution is visible rather
   than implied by an annotation on an empty axis.
8. Edge fitting at approximately 456 eV (Ti), 532 eV (O), and 640 eV (Mn),
   repeated over the raster to form elemental maps.
9. The fast-camera -> DAQIRI -> GPU -> live-output path and optional selective
   raw storage.
10. Exposure convergence and DOEELS-based selection among the Structure, Mn,
    O, and Ti views.
11. A concise provenance and limitations screen.

Detailed abTEM/GOSH derivations, phase-space transfer coefficients, and the
detector calibration plots remain generated assets and source HDF5 products,
but are intentionally not part of the main general-audience narrative.

## Quick start: edit or render the committed result

The compact `assets/generated_200keV/` presentation pack is committed. It
contains derived images and scientific metadata, not the bulk HDF5 simulation
products. A fresh clone only needs the Manim environment to edit or render the
existing movie:

```bash
scripts/setup_demo_env.sh
demo/render_demo.sh preview
demo/render_demo.sh full
```

Run these commands from `simulation/`. The main output is:

```text
demo/renders/lmto_doeels_workflow_200keV_FullDemo.mp4
```

To regenerate the assets or change the physics, create both local environments
and follow the full workflow in [`../README.md`](../README.md):

```bash
scripts/setup_gpaw_env.sh
scripts/setup_demo_env.sh
scripts/reproduce_200kev_demo.sh
```

Render an individual low-resolution section while editing:

```bash
demo/render_demo.sh scene ApplicationScene
demo/render_demo.sh scene SpecimenScene
demo/render_demo.sh scene SpectrumScene
demo/render_demo.sh scene DaqiriScene
demo/render_demo.sh scene ResultsScene
```

The default font is Avenir Next. It can be overridden if needed:

```bash
LMTO_DEMO_FONT="DejaVu Sans" demo/render_demo.sh preview
```

All presentation labels use one explicit font and avoid mixed Unicode fallback
for arrows and subscripts; this removes the irregular character spacing seen in
the earlier cut.

## Inputs and versioned presentation assets

[`demo_config.json`](demo_config.json) is the single presentation input. It
points at the external/generated 200 keV HDF5 products and writes deterministic
graphics to `assets/generated_200keV/`. The versioned
`assets/generated_200keV/demo_metadata.json` records the exact input paths,
scan parameters, fitted detector values, selected probe positions, element
counts, correlations, and scientific limitations used by the scenes.

The presentation-relevant 200 keV physics products are:

- `../spectral_libraries/lmto_gosh_200keV.h5`;
- `../output/lmto_abtem_spatial_200keV_refined.h5`;
- `../output/lmto_abtem_200keV_spectrum_image.h5`;
- `../calibration/monte_carlo_200keV_response_kernel.h5`;
- `../calibration/nio_15pa_sparse_detector_calibration_200keV.h5`;
- `../output/lmto_raw_doeels_200keV_87khz_nio_calibrated_stride4.h5`;
- `../output/lmto_raw_doeels_200keV_exposure_sweep_16x16.h5`.

## Claim boundaries

- The chip and full battery-particle images are workflow concepts, not simulated
  datasets. The physics calculation shown is the small LMTO cell.
- The GOSH baseline is independent-atom physics. It does not yet predict
  LMTO-specific oxidation state, bonding, coordination, or ELNES.
- The spectrometer dispersion, point-spread function, row mapping, and repeated
  ZLP-lane interpretation remain provisional until measured calibration exists.
- Silicon thickness and effective gain remain partly degenerate in the NiO fit.
- The 35 ms product retains spectra and sufficient statistics, not all 768,000
  physical raw readouts.
- The through-focus products are separate simulated focal sections, not depth
  recovered from one conventional scan. No depth-resolution claim is made in
  this movie.
- Element maps are fitted edge amplitudes. They are not colored atom labels.

## Validation

```bash
.conda-envs/eels-sim-gpaw/bin/python -m unittest demo/test_prepare_demo_assets.py -v
.venv-demo/bin/python -m py_compile demo/scenes.py demo/prepare_demo_assets.py
```

[`STORYBOARD.md`](STORYBOARD.md) contains the narration, claim-safe wording,
and external-review checklist.
