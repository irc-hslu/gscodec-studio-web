#!/usr/bin/env bash
# Generate JPEG RF ES5 trajectories from PCQ report presets and optional videos.
set -euo pipefail

cd "$(dirname "$0")/.."
REGISTRY="configs/jpeg_rf_trajectories.json"
TRAIN_CAM="eva/_eval_100views/cameras_100.json"
OUT="eva/jpeg_rf_trajectories"

# Single recommended preset + video
python generate_jpeg_rf_trajectory.py \
  --cfg.trajectory mellifont_recommended \
  --cfg.registry "$REGISTRY" \
  --cfg.training-cameras "$TRAIN_CAM" \
  --cfg.output-dir "$OUT" \
  --cfg.merge-environment-baseline \
  --cfg.center-from-ply ../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply \
  --cfg.n-frames 600 --cfg.fps 25

# All PCQ presets (cameras JSON only):
# python generate_jpeg_rf_trajectory.py --cfg.all-presets \
#   --cfg.registry "$REGISTRY" --cfg.training-cameras "$TRAIN_CAM" \
#   --cfg.output-dir "$OUT" --cfg.n-frames 600 --cfg.fps 25

# Render video (CUDA):
# python generate_jpeg_rf_trajectory.py --cfg.trajectory mellifont_recommended \
#   --cfg.render-video --cfg.n-frames 600 --cfg.fps 25
