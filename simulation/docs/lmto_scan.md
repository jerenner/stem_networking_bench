# LMTO spatial spectrum-image demonstrator

The spatial demonstrator produces the same classes of views as the supplied
HAADF/element example: a HAADF-like image, selectable Li/Mn/Ti/O maps, a full
energy-loss spectrum at every scan coordinate, and a focal-depth series. The
default model is Li1.2Mn0.4Ti0.4O2 (LMTO), not the Ca/Nd/Ti/O material in the
reference image.

Two spatial providers are available. `analytic` is the fast Gaussian
atomic-density demonstrator. `abtem_cache` is the physical upgrade: a cached
abTEM multislice probe scan supplies annular HAADF and six edge-specific
transition-potential maps, while GOSH supplies absolute cross sections and
energy dependence.

Fine structure is not required merely to demonstrate where Li, Mn, Ti, and O
are located. Their represented edges have distinct onsets, and the existing
DFT-GOSH library supplies their energy-dependent atomic cross sections. Fine
structure becomes important when the goal changes from element identification
to oxidation state, coordination, bonding, or a realistic fit to an
experimental LMTO spectrum.

## Run the demonstrator

### Fast analytic version

```bash
cd simulation
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim simulate-lmto-scan \
  --config config/energy_lmto.toml \
  --output-hdf5 output/lmto_spectrum_image.h5 \
  --output-montage output/lmto_element_scan.png \
  --output-html output/lmto_element_scan.html
```

Open `output/lmto_element_scan.html` in a browser. The reconstruction selector
switches among HAADF, Li, Mn, Ti, and O. The second control changes the
simulated focal depth. The HTML is standalone; its display maps are embedded
and it does not need a server or the HDF5 file.

The default acquisition contains 8 independent 64x64 focal sections, 792
one-eV energy bins from -2 to 790 eV, and 200,000 incident electrons per probe
position. It represents 6.55 billion statistically sampled electron outcomes,
but stores binned integer counts rather than allocating one row per electron.
That binning is statistically equivalent to individual categorical sampling
for the model used here and keeps the output compact.

### abTEM spatial version

The abTEM calculation is separated from stochastic spectrum generation so it
can be cached and reused when changing dose, energy binning, or reconstruction:

```bash
# About five minutes on the current workstation: three depths, all six edges.
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim \
  generate-lmto-abtem-spatial \
  --config config/energy_lmto_abtem.toml \
  --output-hdf5 output/lmto_abtem_spatial.h5

# The 50-80 mrad HAADF annulus needs the finer 0.09 A potential grid. This
# inexpensive command retains the validated 0.12 A edge maps and replaces only
# the HAADF channel in a provenance-linked cache.
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim \
  refine-lmto-abtem-haadf \
  --config config/energy_lmto_abtem.toml \
  --input-hdf5 output/lmto_abtem_spatial.h5 \
  --output-hdf5 output/lmto_abtem_spatial_refined.h5 \
  --potential-sampling-A 0.09

# Fast cached stochastic sampling: 2,000,000 electrons per probe position.
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim simulate-lmto-scan \
  --config config/energy_lmto_abtem.toml \
  --output-hdf5 output/lmto_abtem_spectrum_image.h5 \
  --output-montage output/lmto_abtem_element_scan.png \
  --output-html output/lmto_abtem_element_scan.html
```

The reference is a 16x16 raster over 0.83 nm, a 1.66 nm-thick periodic LMTO
cell, and three nominal focal depths. It uses a 20 mrad probe, 50-80 mrad
HAADF detector, and 0-50 mrad EELS detector. Li K, Ti L2,3, O K, and Mn L2,3
use double channeling; the shallow Ti/Mn M2,3 maps use single channeling.
abTEM 1.0.10 and the project GPAW environment successfully calculate all six
requested shells.

The displayed Ti and Mn maps deliberately select their localized L2,3 edges,
not their much stronger but spatially diffuse M2,3 edges. O uses O K and Li
uses Li K. Li K has only about 0.7% spatial coefficient of variation in this
calculation and therefore does **not** resolve individual Li columns. That is a
physical result of the shallow, delocalized edge rather than a plotting defect.
The Ti L2,3 and Mn L2,3 maps have roughly 88-90% raw spatial contrast, while O
K has about 44%.

## Spatial sampling confirmation

`config/energy_lmto_abtem_confirmation.toml` repeats the middle focal section
at 0.09 A potential sampling. Compare it with:

```bash
.conda-envs/eels-sim-gpaw/bin/eels-sim compare-lmto-abtem-spatial \
  --reference output/lmto_abtem_spatial_refined.h5 \
  --confirmation output/lmto_abtem_spatial_confirmation_0p09A.h5 \
  --output-json output/lmto_abtem_spatial_refined_vs_0p09.json
```

After mean normalization, the 0.12/0.09 A edge-map correlations are
0.988-0.99997 and relative RMS differences are 0.04-1.14%. The original
0.12 A HAADF differed strongly because 80 mrad was near its valid angular
limit; the final combined cache therefore uses the confirmed 0.09 A HAADF.
This confirms the chosen real-space sampling for the present small/static
model, but it is not yet a convergence study of lateral extent, thickness,
thermal diffuse scattering, structure relaxation, or scan sampling.

## Analytic spatial/event model

1. A periodic, unrelaxed disordered-rocksalt volume is generated with a 4.15 A
   lattice constant. Oxygen occupies the anion sublattice and cation sites are
   randomly assigned at Li:Mn:Ti = 0.6:0.2:0.2, equivalent to
   Li1.2Mn0.4Ti0.4O2 after normalization to O2.
2. Each focal section weights atomic planes with the configured axial Gaussian
   and convolves the projected atom density with the configured lateral probe.
3. HAADF is an incoherent `Z^1.7` column-response proxy followed by Poisson
   counting noise. It is not yet an abTEM annular-detector calculation.
4. At every probe coordinate, atomic DFT-GOSH cross sections and local areal
   densities set the optical depth of all six represented edges: Ti M2,3,
   Mn M2,3, Li K, Ti L2,3, O K, and Mn L2,3. A multinomial draw assigns every
   incident electron to ZLP, low loss, or one core edge. A second draw places
   it in an energy bin. Counts are conserved exactly.
5. Element maps sum the known simulated edge components for each element and
   apply the configured reconstruction PSF. This is an ideal model-component
   unmixing result. Real data processing must estimate backgrounds and separate
   the overlapping Ti M, Mn M, and Li K signals rather than reading their
   latent simulation labels.

The event-rate calculation is quantitative to the extent of the configured
independent-atom GOSH cross sections. The atomic arrangement, HAADF response,
probe response, low-loss model, and element reconstruction are demonstrator
models, not an instrument-validated prediction.

For the abTEM provider, items 1-3 above are replaced by multislice propagation,
an annular detector, and GPAW-backed transition potentials. The final spectrum
still uses GOSH energy profiles and normalization. Element reconstruction is
still ideal latent-component unmixing, and the output still stops before the
silicon detector/readout simulation.

## HDF5 contract

The root schema is `eels-sim-spectrum-image-v1`.

| Path | Shape | Meaning |
|---|---|---|
| `/axes/focal_depth_A` | `[depth]` | simulated focal positions |
| `/axes/scan_x_A`, `/axes/scan_y_A` | `[x]`, `[y]` | probe coordinates |
| `/axes/energy_centers_eV` | `[energy]` | spectrum-bin centers |
| `/structure/atoms/*` | `[atom]` | generated element, position, and plane truth |
| `/truth/element_response_per_A2` | `[depth,element,y,x]` | noiseless local response |
| `/reconstruction/haadf_counts` | `[depth,y,x]` | HAADF-like counted image |
| `/reconstruction/elements/{Li,Mn,Ti,O}` | `[depth,y,x]` | element maps |
| `/spectrum/component_total_counts` | `[depth,y,x,component]` | sampled latent components |
| `/spectrum/component_profiles` | `[component,energy]` | normalized spectral bases |
| `/spectrum/counts` | `[depth,y,x,energy]` | full integer spectrum image |

The reusable `eels-sim-abtem-spatial-v1` cache separately stores
`/spatial/haadf_fraction`, `/spatial/raw_edge_intensity`, and
`/spatial/edge_response_per_A2`. Its metadata records transition support,
single/double channeling, detector angles, actual potential grids, GOSH
normalization, and any HAADF refinement source.

## Meaning of the depth slider

The depth control is explicitly a **synthetic focal-section series**. Each
slider position is a separate simulated acquisition with an axial response
centered at another depth. A conventional 2D EELS spectrum image does not by
itself reveal the depth of every atom. A credible depth-section prediction
needs a converged 3D multislice calculation, the real convergence aperture and
aberrations, channeling, and an experimental acquisition protocol such as a
through-focus or tilt series.

The next upgrades are a relaxed and larger LMTO structural model, frozen-
phonon averaging, and a broader convergence study. FEFF-derived ELNES can be
added to the same spectral bases independently when chemical-state contrast is
needed. The cached spectrum image can now be passed through the implemented
spectrometer, event-sampled Geant4 silicon response, charge sharing, calibrated
readout, and ADC model. See [raw_doeels.md](raw_doeels.md).
