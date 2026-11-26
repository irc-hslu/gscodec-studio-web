#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate gscodec_studio
cd /mnt/d/LiveStream/GSCodec_Studio/examples
python score_nr_video_metrics.py "$@"
