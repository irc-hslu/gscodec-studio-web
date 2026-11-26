#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:-/mnt/d/LiveStream/GSCodec_Studio}"
FRAME_START="${2:-0}"
FRAME_END="${3:-0}"

COLMAP_SRC="$PROJECT/colmap1/workspace"
IMAGES_SRC="$PROJECT/colmap1/images"
DATA_DIR="$PROJECT/colmap_seq"

if [[ ! -d "$COLMAP_SRC/sparse/0" ]]; then
  echo "Missing COLMAP sparse model: $COLMAP_SRC/sparse/0"
  exit 1
fi

if [[ ! -d "$IMAGES_SRC" ]]; then
  echo "Missing reference images: $IMAGES_SRC"
  exit 1
fi

mkdir -p "$DATA_DIR"

for i in $(seq "$FRAME_START" "$FRAME_END"); do
  frame=$(printf "000__%04d" "$i")
  mkdir -p "$DATA_DIR/$frame/sparse"
  ln -sfn "$COLMAP_SRC/sparse/0" "$DATA_DIR/$frame/sparse/0"
  ln -sfn "$IMAGES_SRC" "$DATA_DIR/$frame/images"
  echo "Prepared $DATA_DIR/$frame"
done

echo "Done. Created $((FRAME_END - FRAME_START + 1)) frame folder(s) under $DATA_DIR"
