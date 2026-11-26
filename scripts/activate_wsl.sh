#!/usr/bin/env bash
# Source this file in WSL before working on GSCodec Studio:
#   source /mnt/d/LiveStream/GSCodec_Studio/scripts/activate_wsl.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gscodec_studio

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"

echo "gscodec_studio env active (CUDA 11.8, project: /mnt/d/LiveStream/GSCodec_Studio)"
