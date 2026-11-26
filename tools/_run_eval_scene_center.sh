#!/usr/bin/env bash
set -eu
source /mnt/d/LiveStream/GSCodec_Studio/scripts/activate_wsl.sh
cd /mnt/d/LiveStream/GSCodec_Studio/examples
SCENE_PLY="../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply"
LOG="eva/_eval_100views/run_scene_center_merge_env.log"
mkdir -p eva/_eval_100views
echo "Starting eval at $(date)" | tee "$LOG"
python eval_encoded_results.py \
  --baseline-ply "$SCENE_PLY" \
  --center-from-ply "$SCENE_PLY" \
  --merge-environment-baseline \
  --num-views 104 \
  --views-per-ring 8 \
  --radius-step-m 1.2 \
  --ring-height-step-m 0.2 \
  --pitch-down-deg 0 \
  --width 1920 \
  --height 1080 \
  --spiral-frames 600 \
  --render-spiral-video \
  --no-skip-existing \
  2>&1 | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
