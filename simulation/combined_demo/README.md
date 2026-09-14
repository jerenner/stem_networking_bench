# Combined LMTO + DAQIRI demonstration

This is a separate movie that joins the LMTO element-identification story to
the DAQIRI acquisition path. It does not replace or modify either existing
demo:

- `simulation/demo/` remains the detailed element-identification movie;
- the repository-level `demo/` remains the NiO acquisition movie;
- this directory renders to its own `.manim/` and `renders/` directories with
  the prefix `lmto_daqiri_combined`.

The combined movie follows one continuous workflow:

1. a focused electron probe crosses the explicit LMTO lattice;
2. synchronized HAADF and DOEELS branches measure structure and energy loss;
3. eight detector sources send native equal-payload tiles to DAQIRI;
4. packet metadata places the 192 ZLP and 768 CoreLoss tiles into their
   different native geometries in a GPU-resident frame;
5. GPU correction, electron counting, edge fitting, and accumulation overlap
   acquisition;
6. registered HAADF, Mn, O, and Ti maps update in the simulation's
   left-to-right, `x`-fast raster order.

The raw pixels, spectrum, structure, and final maps are the same generated
200 keV simulation products used by `simulation/demo`. The native target
layout has 960 tiles per frame: 192 `128 x 32`-pixel ZLP tiles over the first
768 columns and 768 `32 x 128`-pixel CoreLoss tiles over the remaining 3,072
columns. Every tile carries 4,096 samples. Packet motion and the rate at which
maps are revealed are explanatory animation, not a measured timing trace.

The receiver also has a temporary compatibility mode for the row-shaped
3,840-sample payloads emitted by the current test transmitter. That mode maps
the first 120 packet ordinals from each source onto the same native tile
geometry and fills the missing 256 samples by repeating the payload prefix. It
is deliberately not presented as the target detector format in the movie.

## Render

The compact shared 200 keV presentation assets are committed under
`simulation/demo/assets/generated_200keV/`. A fresh clone can edit or render
this movie without Geant4, GPAW, abTEM, the bulk HDF5 products, or the measured
NiO inputs. Create only the Manim environment from the repository root:

```bash
simulation/scripts/setup_demo_env.sh
```

Then render the new movie:

```bash
cd simulation
combined_demo/render_demo.sh preview
combined_demo/render_demo.sh full
```

To regenerate or change the physics rather than only the presentation, follow
`simulation/README.md` and run `simulation/scripts/reproduce_200kev_demo.sh`.

The outputs are:

```text
combined_demo/renders/lmto_daqiri_combined_FullDemo.mp4
```

To use an existing asset directory without copying it:

```bash
LMTO_COMBINED_ASSETS=/path/to/generated_200keV \
  combined_demo/render_demo.sh preview
```

Individual sections are independently renderable:

```bash
combined_demo/render_demo.sh scene PacketAssemblyScene
combined_demo/render_demo.sh scene LiveMapsScene
```

Available scenes are `MicroscopeScene`, `PacketAssemblyScene`,
`StreamingScene`, `LiveMapsScene`, `ResultsScene`, and `FullDemo`.

## Review and validation

```bash
.venv-demo/bin/python -m py_compile combined_demo/scenes.py
.conda-envs/eels-sim-gpaw/bin/python -m unittest combined_demo/test_combined_demo.py -v
```

See `STORYBOARD.md` for narration, timing, and scientific claim boundaries.
