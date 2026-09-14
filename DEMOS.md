# Video demo handoff

This is the entry point for editing and reproducing the repository's Manim
movies. Coding agents should read this file first, then the README and
storyboard for the selected movie. No prior chat history is required.

## Choose a movie

| Movie | Source and storyboard | Main output |
| --- | --- | --- |
| Recorded NiO acquisition and DAQIRI processing | [`demo/`](demo/README.md), [`demo/STORYBOARD.md`](demo/STORYBOARD.md) | `demo/renders/stem_daqiri_nio_FullDemo.mp4` |
| 200 keV LMTO element identification | [`simulation/demo/`](simulation/demo/README.md), [`simulation/demo/STORYBOARD.md`](simulation/demo/STORYBOARD.md) | `simulation/demo/renders/lmto_doeels_workflow_200keV_FullDemo.mp4` |
| Combined LMTO microscope, detector tiling, DAQIRI, and live maps | [`simulation/combined_demo/`](simulation/combined_demo/README.md), [`simulation/combined_demo/STORYBOARD.md`](simulation/combined_demo/STORYBOARD.md) | `simulation/combined_demo/renders/lmto_daqiri_combined_FullDemo.mp4` |

The scene implementations are `demo/scenes.py`, `simulation/demo/scenes.py`,
and `simulation/combined_demo/scenes.py`. Keep a new concept in a new scene or
demo directory unless the request explicitly asks to revise an existing cut.

## Fast path: edit or render without rerunning physics

The repository versions two compact derived asset packs:

- `demo/assets/generated/` contains the recorded-NiO images, spectrum data,
  and metadata used by the acquisition movie;
- `simulation/demo/assets/generated_200keV/` contains the LMTO structure,
  simulated detector, spectrum, elemental-map images, and scientific metadata
  shared by the LMTO and combined movies.

These packs contain presentation-ready PNG/NPZ/JSON files, not raw microscope
data or bulk HDF5 simulation products. Their manifests record the processing
settings and scientific provenance used by the scenes.

Install the rendering environment from the repository root. Manim also needs
its normal platform dependencies, including FFmpeg and Cairo/Pango:

```bash
simulation/scripts/setup_demo_env.sh
```

Render low-resolution previews of all three movies:

```bash
PYTHON=simulation/.venv-demo/bin/python demo/render_demo.sh preview
(
  cd simulation
  demo/render_demo.sh preview
  combined_demo/render_demo.sh preview
)
```

Replace `preview` with `full` for 1920 x 1080 output. Each render script also
supports `scene <SceneName>` for quick iteration; its README lists the useful
scene names. MP4 files, Manim caches, and review frames remain local and are
excluded from Git.

For consistent text layout, use the documented font environment variable. If
the original macOS fonts are unavailable, use `DejaVu Sans`:

```bash
STEM_DEMO_FONT="DejaVu Sans" \
  PYTHON=simulation/.venv-demo/bin/python demo/render_demo.sh preview
(
  cd simulation
  LMTO_DEMO_FONT="DejaVu Sans" demo/render_demo.sh preview
  LMTO_DEMO_FONT="DejaVu Sans" combined_demo/render_demo.sh preview
)
```

## Suggested coding-agent prompt

> Read `AGENTS.md` and `DEMOS.md` completely, followed by the README and
> STORYBOARD for the movie I name. Use the committed compact asset packs; do
> not rerun or alter the physics simulation unless I explicitly request it.
> Preserve the documented scientific claim boundaries and keep generated MP4,
> Manim cache, and review files out of Git. Make the requested scene changes,
> run the relevant tests, render an individual low-resolution scene for visual
> inspection, and then render the complete preview. Do not overwrite either
> of the other demo projects.

Give the agent the audience, desired message, requested edits, target duration,
and any font or branding constraints after that paragraph.

## Advanced path: regenerate or modify the simulation

Read [`simulation/README.md`](simulation/README.md) and its linked physics
documents. The full 200 keV workflow requires Conda, a C/C++ toolchain, CMake,
Geant4 with its data files, GPAW, abTEM, eXSpy, the checksum-verified GOSH
download, and two measured NiO HDF5 inputs supplied outside Git.

```bash
cd simulation
scripts/setup_gpaw_env.sh
scripts/setup_demo_env.sh

export EELS_GEANT4_PREFIX=/path/to/geant4/install
export EELS_DARK_HDF5=/path/to/nio_15pa_dark_frames_float32_uncompressed.h5
export EELS_SPECTRUM_HDF5=/path/to/nio_15pa_spectrum_frames_float32_uncompressed.h5
scripts/reproduce_200kev_demo.sh
```

That script rebuilds the GOSH library, abTEM spatial response, spectrum image,
Geant4 silicon response, NiO-constrained detector calibration, raw DOEELS
products, compact presentation assets, and final LMTO movie. Bulk HDF5/CSV
outputs remain ignored. Use `demo/render_demo.sh assets` after changing the
recorded-NiO processing configuration to refresh its compact asset pack.

Before committing regenerated assets, review the metadata, storyboards, and
claim boundaries; derived image changes are part of the scientific review, not
only a visual refresh.
