#!/usr/bin/env bash
# Dynamic GS (Spacetime Gaussians) on Cagliari Nov 2025 capture: 6 calibrated cams, 1920x1080 @ 30 fps.
#
# Prepare data once (from repo root, COLMAP 3.9 on PATH):
#   python mpeg_gsc_utils/multiview_video_preprocess/prepare_calibrated_multiview.py \
#       --base_dir examples/data/Cagliari/seated_3300 --video_dir <Captures_Cagliari_Nov_2025>/all \
#       --start_frame 3300 --frame_num 120
#   PYTHONPATH=. python mpeg_gsc_utils/multiview_video_preprocess/run_per_frame_colmap.py \
#       --scene Cagliari --base_dir examples/data/Cagliari/seated_3300 --frame_num 120
#
# Then from examples/:  bash scripts/train_dyngs_cagliari.sh
set -e

SCENE_DIR=${SCENE_DIR:-data/Cagliari/seated_3300/colmap}
RESULT_DIR=${RESULT_DIR:-results/dyngs/Cagliari_seated_3300_v2}
GOF=${GOF:-60}                    # frames per group (one model per group)
TOTAL_FRAMES=${TOTAL_FRAMES:-120}
MAX_STEPS=${MAX_STEPS:-25000}     # must be > 10k: best ckpt is only saved after step 10k
# Only 6 cameras: train on all of them (--no-eval); "val" metrics are then computed on
# training camera v00 and only serve as a sanity check.
# --prune_z_below None: cameras are ~2.5 m from the subject, the N3D z<4.5 cleanup would delete it
COMMON="--downscale_factor 1 --no-eval --no-pin-memory --prune_z_below None"
# STG_Strategy stops growing after 6 rounds (~45k GSs -> blurry, esp. moving parts). Modified_STG_Strategy
# keeps growing until refine_stop_iter; the higher grad threshold and earlier stop limit floaters.
STRATEGY="--strategy Modified_STG_Strategy --grow_grad2d 0.0004 --refine_start_iter 1500 \
    --refine_stop_iter 15000 --reset_every 5000 --pause_refine_after_reset 2000"

python simple_trainer_dyngs.py default \
    --data_dir $SCENE_DIR/colmap_0 \
    --model_path $RESULT_DIR \
    --result_dir $RESULT_DIR \
    --duration $GOF \
    --total_frames $TOTAL_FRAMES \
    --max_steps $MAX_STEPS \
    --eval_steps 10000 20000 $MAX_STEPS \
    $STRATEGY \
    $COMMON

# Export one static .ply per frame from each group's best checkpoint.
for ((g=0; g*GOF<TOTAL_FRAMES; g++)); do
    start=$((g*GOF))
    frames=$(( TOTAL_FRAMES-start < GOF ? TOTAL_FRAMES-start : GOF ))
    group=$RESULT_DIR/$(printf gof_%03d $GOF)/$(printf group_%03d $g)
    python simple_trainer_dyngs.py default \
        --data_dir $SCENE_DIR/colmap_$start \
        --model_path $group \
        --result_dir $group \
        --duration $frames \
        --ckpt $group/ckpts/ckpt_best_rank0.pt \
        --enable_dyn_splats_export \
        $COMMON
done
