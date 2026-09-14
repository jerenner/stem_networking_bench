#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${STEM_DEMO_CONFIG:-$ROOT/demo/demo_config.json}"
PYTHON="${PYTHON:-python3}"
ASSETS="${STEM_DEMO_ASSETS:-$ROOT/demo/assets/generated}"
MEDIA="$ROOT/demo/.manim"
RENDERS="$ROOT/demo/renders"
MODE="${1:-preview}"
SCENE="${2:-FullDemo}"

mkdir -p "$RENDERS"

prepare_assets() {
  "$PYTHON" "$ROOT/demo/prepare_demo_assets.py" --config "$CONFIG" "${@:1}"
}

render_scene() {
  local quality="$1"
  local scene="$2"
  STEM_DEMO_ASSETS="$ASSETS" "$PYTHON" -m manim "$quality" \
    --fps 30 \
    --media_dir "$MEDIA" \
    --output_file "${scene}" \
    "$ROOT/demo/scenes.py" "$scene"
  local rendered
  rendered="$(find "$MEDIA/videos" -type f -name "${scene}.mp4" -print0 | xargs -0 ls -t | head -n 1)"
  if [[ -z "$rendered" ]]; then
    echo "Could not locate rendered ${scene}.mp4 under $MEDIA/videos" >&2
    exit 1
  fi
  ffmpeg -hide_banner -loglevel error -y -i "$rendered" -c copy \
    "$RENDERS/stem_daqiri_nio_${scene}.mp4"
  echo "Wrote $RENDERS/stem_daqiri_nio_${scene}.mp4"
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
