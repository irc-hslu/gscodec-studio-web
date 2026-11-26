#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/d/LiveStream/GSCodec_Studio"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gscodec_studio

# CUDA 11.8 (matches PyTorch cu118). cicc lives under nvvm/bin.
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"

echo "=== Verifying PyTorch + CUDA ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "=== Installing gsplat (GSCodec Studio) ==="
cd "$PROJECT_ROOT"
pip install ninja
pip install -e . --no-build-isolation

echo "=== Installing example dependencies ==="
cd "$PROJECT_ROOT/examples"
# requirements.txt pins numpy<1.26; torch may have installed a newer numpy.
pip install "numpy<1.26"
pip install -r requirements.txt --no-build-isolation

echo "=== Installing third-party packages ==="
cd "$PROJECT_ROOT"
pip install third_party/python-fpnge-master --no-build-isolation
pip install third_party/gridencoder --no-build-isolation

echo "=== Verifying gsplat import ==="
python - <<'PY'
import gsplat
print("gsplat version:", gsplat.__version__)
PY

echo "=== Setup complete ==="
