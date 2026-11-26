#!/usr/bin/env bash
# One-time setup for the external ContextGS baseline (separate conda env).
#
# ContextGS is NOT bundled inside GSCodec Studio. This clones upstream next to
# other third_party tools. You still need:
#   - conda env from ContextGS/environment.yml  (env name: contextgs)
#   - COLMAP datasets under ContextGS/data/
#
# Upstream: https://github.com/wyf0912/ContextGS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
CTX_DIR="$TP/ContextGS"

mkdir -p "$TP"

if [ ! -d "$CTX_DIR/.git" ]; then
  echo "=== cloning ContextGS ==="
  git clone --depth 1 https://github.com/wyf0912/ContextGS.git "$CTX_DIR"
else
  echo "=== ContextGS already cloned: $CTX_DIR ==="
fi

echo "=== unpacking submodules ==="
cd "$CTX_DIR/submodules"
for z in diff-gaussian-rasterization.zip simple-knn.zip; do
  if [ -f "$z" ] && [ ! -d "${z%.zip}" ]; then
    unzip -q -o "$z"
  fi
done
cd "$CTX_DIR"

echo ""
echo "Next steps (manual — ContextGS uses its own conda env):"
echo "  1. conda env create --file $CTX_DIR/environment.yml"
echo "  2. conda activate contextgs"
echo "  3. Place datasets under $CTX_DIR/data/  (e.g. data/tandt/train)"
echo "  4. From GSCodec examples/: bash scripts/contextgs_tt_baseline.sh"
echo ""
echo "See docs/contextgs_baseline.md for comparison notes vs seq-HEVC."
