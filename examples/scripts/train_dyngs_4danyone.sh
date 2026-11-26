#!/usr/bin/env bash
# Dynamic GS (Spacetime Gaussians) on 4DAnyone 010_multicam: 21 cams x 121 frames @ 30 fps.
#
# Prepare data once (from repo root):
#   python mpeg_gsc_utils/multiview_video_preprocess/convert_4dgs_multipleview.py \
#       --src <4DAnyone>/data/4dgs/010_multicam --out examples/data/4DAnyone/010_multicam
#
# Then from examples/:  bash scripts/train_dyngs_4danyone.sh
set -e

SCENE_DIR=data/4DAnyone/010_multicam/colmap
RESULT_DIR=${RESULT_DIR:-results/dyngs/4DAnyone_010_multicam}
GOF=${GOF:-60}                    # frames per group (one model per group)
TOTAL_FRAMES=${TOTAL_FRAMES:-120}
MAX_STEPS=${MAX_STEPS:-30000}     # must be > 10k: best ckpt is only saved after step 10k
TEST_VIEW=0                       # image1 (cam01) held out for evaluation
# --prune_z_below None: this scene lies at depth ~1-5.6, so the N3D z<4.5 cleanup would delete everything
COMMON="--downscale_factor 1 --test_view_id $TEST_VIEW --no-pin-memory --white-background --prune_z_below None"

python simple_trainer_dyngs.py default \
    --data_dir $SCENE_DIR/colmap_0 \
    --model_path $RESULT_DIR \
    --result_dir $RESULT_DIR \
    --duration $GOF \
    --total_frames $TOTAL_FRAMES \
    --max_steps $MAX_STEPS \
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
