#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${LMTO_DEMO_CONFIG:-$ROOT/demo/demo_config.json}"
ASSETS="${LMTO_DEMO_ASSETS:-$ROOT/demo/assets/generated_200keV}"
MEDIA="$ROOT/demo/.manim"
RENDERS="$ROOT/demo/renders"
RENDER_PREFIX="${LMTO_DEMO_RENDER_PREFIX:-lmto_doeels_workflow_200keV}"
MODE="${1:-preview}"
SCENE="${2:-FullDemo}"

DEFAULT_ASSET_PYTHON="$ROOT/.conda-envs/eels-sim-gpaw/bin/python"
DEFAULT_MANIM_PYTHON="$ROOT/.venv-demo/bin/python"
ASSET_PYTHON="${LMTO_ASSET_PYTHON:-$DEFAULT_ASSET_PYTHON}"
MANIM_PYTHON="${LMTO_MANIM_PYTHON:-$DEFAULT_MANIM_PYTHON}"

if [[ ! -x "$ASSET_PYTHON" ]]; then
  ASSET_PYTHON="${PYTHON:-python3}"
fi
if [[ ! -x "$MANIM_PYTHON" ]]; then
  MANIM_PYTHON="${PYTHON:-python3}"
fi

mkdir -p "$RENDERS"

prepare_assets() {
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/eels-lmto-demo-mpl}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/eels-lmto-demo-cache}" \
  "$ASSET_PYTHON" "$ROOT/demo/prepare_demo_assets.py" \
    --config "$CONFIG" "${@:1}"
}

render_scene() {
  local quality="$1"
  local scene="$2"
  LMTO_DEMO_ASSETS="$ASSETS" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/eels-lmto-demo-manim}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/eels-lmto-demo-cache}" \
  "$MANIM_PYTHON" -m manim "$quality" --fps 30 \
    --media_dir "$MEDIA" --output_file "$scene" \
    "$ROOT/demo/scenes.py" "$scene"
  local rendered
  rendered="$(find "$MEDIA/videos" -type f -name "${scene}.mp4" -print0 | xargs -0 ls -t | head -n 1)"
  if [[ -z "$rendered" ]]; then
    echo "Could not locate rendered ${scene}.mp4 under $MEDIA/videos" >&2
    exit 1
  fi
  ffmpeg -hide_banner -loglevel error -y -i "$rendered" -c copy \
    "$RENDERS/${RENDER_PREFIX}_${scene}.mp4"
  echo "Wrote $RENDERS/${RENDER_PREFIX}_${scene}.mp4"
}

case "$MODE" in
  assets)
    prepare_assets --force
    ;;
  preview)
    prepare_assets
    render_scene -ql FullDemo
    ;;
  full)
    prepare_assets
    render_scene -qh FullDemo
    ;;
  scene)
    prepare_assets
    render_scene -ql "$SCENE"
    ;;
  *)
    echo "Usage: $0 {assets|preview|full|scene [SceneName]}" >&2
    exit 2
    ;;
esac
