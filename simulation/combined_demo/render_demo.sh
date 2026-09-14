#!/usr/bin/env bash
set -euo pipefail

SIMULATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_ROOT="$SIMULATION_ROOT/combined_demo"
ASSETS="${LMTO_COMBINED_ASSETS:-$SIMULATION_ROOT/demo/assets/generated_200keV}"
MEDIA="$DEMO_ROOT/.manim"
RENDERS="$DEMO_ROOT/renders"
RENDER_PREFIX="${LMTO_COMBINED_RENDER_PREFIX:-lmto_daqiri_combined}"
MODE="${1:-preview}"
SCENE="${2:-FullDemo}"
MANIM_PYTHON="${LMTO_MANIM_PYTHON:-$SIMULATION_ROOT/.venv-demo/bin/python}"

if [[ ! -x "$MANIM_PYTHON" ]]; then
  MANIM_PYTHON="${PYTHON:-python3}"
fi

prepare_assets() {
  LMTO_DEMO_ASSETS="$ASSETS" "$SIMULATION_ROOT/demo/render_demo.sh" assets
}

ensure_assets() {
  if [[ ! -f "$ASSETS/demo_metadata.json" ]]; then
    prepare_assets
  fi
}

render_scene() {
  local quality="$1"
  local scene="$2"
  mkdir -p "$RENDERS"
  LMTO_COMBINED_ASSETS="$ASSETS" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/eels-lmto-combined-manim}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/eels-lmto-combined-cache}" \
    "$MANIM_PYTHON" -m manim "$quality" --fps 30 \
      --media_dir "$MEDIA" --output_file "$scene" \
      "$DEMO_ROOT/scenes.py" "$scene"
  local rendered
  rendered="$(find "$MEDIA/videos" -type f -name "${scene}.mp4" -print0 | \
    xargs -0 ls -t | head -n 1)"
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
    prepare_assets
    ;;
  preview)
    ensure_assets
    render_scene -ql FullDemo
    ;;
  full)
    ensure_assets
    render_scene -qh FullDemo
    ;;
  scene)
    ensure_assets
    render_scene -ql "$SCENE"
    ;;
  *)
    echo "Usage: $0 {assets|preview|full|scene [SceneName]}" >&2
    exit 2
    ;;
esac
