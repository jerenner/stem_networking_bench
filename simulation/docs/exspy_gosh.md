# eXSpy/DFT-GOSH core-loss provider

The LMTO spectrum now uses atomic generalized oscillator strengths instead of
analytic core-edge continua. `eels_sim.exspy_gosh` evaluates eXSpy's DFT-GOSH
cross sections on the configured energy grid and converts them into the
`eels-sim-spectral-library-v1` contract.

## Reproducible input

The provider is pinned to the public Segger–Guzzinati–Kohl DFT-GOSH 1.5
database:

- DOI: `10.5281/zenodo.7645765`
- file: `Segger_Guzzinati_Kohl_1.5.0.gosh`
- MD5: `7fee8891c147a4f769668403b54c529b`

The 42 MB source database is reproducibly downloaded but excluded from source
control. The generated compact spectral library contains the normalized energy
densities, integrated cross sections, and full provenance required at runtime.

```bash
cd simulation
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/python scripts/download_gosh.py
```

## Six LMTO components

| Component | eXSpy/GOSH subshells | Atomic onset(s) |
|---|---|---:|
| Ti M2,3 | `Ti_M3 + Ti_M2` | 35 eV |
| Mn M2,3 | `Mn_M3 + Mn_M2` | 51 eV |
| Li K | `Li_K` | 55 eV |
| Ti L2,3 | `Ti_L3 + Ti_L2` | 456, 462 eV |
| O K | `O_K` | 532 eV |
| Mn L2,3 | `Mn_L3 + Mn_L2` | 640, 651 eV |

eXSpy stores spin-orbit partners as individual subshells. The provider evaluates
each partner with its proper onset and occupancy factor and sums their
energy-differential cross sections. The default microscope calculation uses a
300 keV beam, 20 mrad convergence semiangle, and 50 mrad collection
semiangle. Cross sections are in barn/eV/atom after aperture integration.

## From cross section to event probability

The nominal LMTO baseline uses Li1.2Mn0.4Ti0.4O2, a configurable provisional
density of 4.0 g/cm3, and 50 nm thickness. For component `i`,

```text
tau_i = integrated_cross_section_i
        * stoichiometric_coefficient_i
        * formula_unit_areal_density
        * 1e-24 cm2/barn
```

The present kernel permits at most one core event. It therefore assigns

```text
p_i = tau_i / sum(tau) * (1 - exp(-sum(tau))).
```

At the defaults, the total represented core optical depth is 0.03050 and the
core-event probability is 0.03004. Every component stores its integrated cross
section, optical depth, and resulting probability in HDF5 metadata.

Generate the compact library and one-million-electron sample:

```bash
MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/python scripts/generate_lmto_library.py

MPLCONFIGDIR=/tmp/eels-sim-mpl \
  .conda-envs/eels-sim-gpaw/bin/eels-sim simulate-1d-spectrum \
  --config config/energy_lmto.toml \
  --electrons 1000000 --frames 465 \
  --output-hdf5 output/lmto_gosh_1d_demonstrator.h5 \
  --output-plot output/lmto_gosh_1d_demonstrator.png \
  --energy-min-eV -2 --energy-max-eV 790 --energy-step-eV 0.25
```

## Physics boundary

This upgrade makes the independent-atom core cross sections and their aperture
dependence quantitative for the stated density and thickness. It does not add
LMTO-specific oxidation, bonding, coordination, or near-edge fine structure;
those require the planned FEFF stage. GOSH also supplies an aperture-integrated
energy cross section, not an outgoing angular distribution, so the kernel's
small Gaussian angular kicks remain explicitly identified proxies. The
low-loss LMTO mixture is unchanged and remains provisional.
