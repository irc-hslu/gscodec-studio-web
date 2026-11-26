#!/usr/bin/env bash
# SOGS (PNG + PLAS) baseline on MellifontAbbey — same scene/LOD/env setup as HEVC eval.
set -euo pipefail

cd "$(dirname "$0")/.."
ITER="../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100"
RATE_POINTS=(rp0 rp1 rp2 rp3)
LODS=("" 1 2 3)   # empty = full point_cloud.ply

echo "=== SOGS compression ==="
for lod in "${LODS[@]}"; do
  for rp in "${RATE_POINTS[@]}"; do
    args=(--rate-point "$rp" --iteration-dir "$ITER" --merge-environment)
    if [[ -n "$lod" ]]; then
      args+=(--ply-lod "$lod")
    fi
    echo "--- LOD=${lod:-full} $rp ---"
    python compress_ply_sogs.py "${args[@]}"
  done
done

echo "=== SOGS evaluation (104 views, same cameras as HEVC baseline) ==="
python eval_sogs_results.py \
  --merge-environment-baseline \
  --center-from-ply "$ITER/point_cloud.ply" \
  --num-views 104 \
  --views-per-ring 8 \
  --radius-step-m 1.2 \
  --ring-height-step-m 0.2 \
  --pitch-down-deg 0 \
  --width 1920 \
  --height 1080

echo "Done. Summary: eva/_eval_100views_sogs/summary.json"
