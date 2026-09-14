# DAQIRI NiO animation demo

This directory builds a reproducible 2D animation that follows one 128-frame
NiO detector bucket through acquisition, correction, spectrum construction,
burst output, and the DigitalMicrograph thinned display. Scientific assets are
generated from local HDF5 files using `stem_analysis`; the animation never
embeds or commits raw detector data.

## Quick start: edit or render the committed result

The compact `assets/generated/` presentation pack is versioned with the demo.
It contains derived images, spectrum data, and metadata, but no raw microscope
frames. A fresh clone can therefore render or edit this movie without access
to the NiO HDF5 files.

Create a dedicated environment from the repository root:

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r demo/requirements.txt
```

Render a low-resolution draft or the complete 1080p composition:

```bash
demo/render_demo.sh preview
demo/render_demo.sh full
```

The scenes use `Helvetica Neue` explicitly so small diagram labels have stable,
readable spacing instead of inheriting Manim's platform-dependent default.
Override it with another installed sans-serif when needed:

```bash
STEM_DEMO_FONT="DejaVu Sans" demo/render_demo.sh full
```

The final MP4 is written under `demo/renders/`. Manim caches, review images,
and video outputs are ignored by Git.

## Regenerate the assets from recorded data

Review `demo/demo_config.json`, especially the paths to the NiO spectrum and
dark-frame files. Relative paths are resolved from the repository root.
`detector.apply_stitch` is false by default: the accepted stitch procedure
requires an independently established no-BLR calibration, so the demo must not
fit a new gain directly from its grouped-BLR bucket.

When those external HDF5 inputs are available, regenerate the compact assets:

```bash
demo/render_demo.sh assets
```

This deliberately updates the versioned presentation pack. Review the changed
assets and metadata before committing them.

The bucket-sum detector image uses an inverse-hyperbolic-sine (`asinh`)
display transform. It is approximately linear around zero and compresses large
positive and negative values, allowing weak residuals and strong peaks to remain
visible together. This affects only the rendered color scale, not the processed
or saved detector values.

## Useful commands

Generate assets directly or override the configuration:

```bash
python demo/prepare_demo_assets.py --config demo/demo_config.json
python demo/prepare_demo_assets.py --config my_demo_config.json --force
```

Render one scene while iterating:

```bash
python -m manim -ql demo/scenes.py ProcessingScene
```

Run hardware-free tests:

```bash
python -m unittest demo/test_prepare_demo_assets.py
python -m py_compile demo/prepare_demo_assets.py demo/scenes.py
```

## Files

- `STORYBOARD.md`: timed visual and narration plan, plus claim checks.
- `demo_config.json`: input paths, processor-equivalent settings, and counters.
- `prepare_demo_assets.py`: streamed HDF5 processing and deterministic plots.
- `scenes.py`: individual Manim scenes and the complete composition.
- `render_demo.sh`: asset, preview, scene, and full-render commands.

The animation is a communication artifact, not a benchmark. Update measured
performance values in the configuration rather than hard-coding new claims in
the scene source.
