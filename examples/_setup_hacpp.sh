#!/usr/bin/env bash
# One-time setup for the external HAC++ baseline (separate conda env).
#
# HAC++ is NOT bundled inside GSCodec Studio. This script clones the upstream
# repo next to GSCodec and prepares submodules. You still need:
#   - conda env from HAC++/environment.yml  (env name: HAC_env)
#   - tmc3 (MPEG GPCC) on PATH for anchor geometry coding
#   - COLMAP datasets under HAC-plus/data/ (see run script)
#
# Upstream: https://github.com/YihangChen-ee/HAC-plus
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
HAC_DIR="$TP/HAC-plus"

mkdir -p "$TP"

if [ ! -d "$HAC_DIR/.git" ]; then
  echo "=== cloning HAC++ ==="
  git clone --depth 1 https://github.com/YihangChen-ee/HAC-plus.git "$HAC_DIR"
else
  echo "=== HAC++ already cloned: $HAC_DIR ==="
fi

echo "=== unpacking submodules (diff-gaussian-rasterization, gridencoder, ...) ==="
cd "$HAC_DIR/submodules"
for z in diff-gaussian-rasterization.zip gridencoder.zip simple-knn.zip arithmetic.zip; do
  if [ -f "$z" ] && [ ! -d "${z%.zip}" ]; then
    unzip -q -o "$z"
  fi
done
cd "$HAC_DIR"

echo ""
echo "Next steps (manual — HAC++ uses its own conda env):"
echo "  1. conda env create --file $HAC_DIR/environment.yml"
echo "  2. conda activate HAC_env"
echo "  3. Install tmc3 (MPEG GPCC) and add to PATH"
echo "  4. Place datasets under $HAC_DIR/data/  (e.g. data/tandt/train)"
echo "  5. From GSCodec examples/: bash scripts/hacpp_tt_baseline.sh"
echo ""
echo "See docs/hacpp_baseline.md for comparison notes vs seq-HEVC."
