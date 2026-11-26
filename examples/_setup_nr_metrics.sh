#!/usr/bin/env bash
# One-time setup for no-reference video quality metrics (VSFA + DOVER).
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate gscodec_studio

ROOT="/mnt/d/LiveStream/GSCodec_Studio/examples"
TP="$ROOT/third_party"
mkdir -p "$TP"

echo "=== installing python deps ==="
# h5py/tensorboardX/scipy are only needed because VSFA.py imports them at module
# scope for its training entry point.
pip install -q einops timm decord h5py tensorboardX scipy scikit-video

echo "=== cloning DOVER ==="
if [ ! -d "$TP/DOVER/.git" ]; then
  git clone --depth 1 https://github.com/VQAssessment/DOVER.git "$TP/DOVER"
fi

echo "=== cloning VSFA ==="
if [ ! -d "$TP/VSFA/.git" ]; then
  git clone --depth 1 https://github.com/lidq92/VSFA.git "$TP/VSFA"
fi

echo "=== downloading DOVER weights ==="
mkdir -p "$TP/DOVER/pretrained_weights"
DOVER_PTH="$TP/DOVER/pretrained_weights/DOVER.pth"
if [ ! -s "$DOVER_PTH" ]; then
  curl -L --fail -o "$DOVER_PTH" \
    https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
fi

echo "=== inventory ==="
ls -la "$TP/DOVER/pretrained_weights"
ls -la "$TP/VSFA/models" 2>/dev/null || echo "VSFA: no models dir"
ls "$TP/VSFA"
python -c "import einops, timm, decord; print('deps ok')"
