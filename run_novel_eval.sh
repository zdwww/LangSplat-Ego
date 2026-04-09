#!/bin/bash
# Novel-view segmentation evaluation for LangSplat v2 experiments
# Stage 1: Generate SAM2 GT masks for moved_050 views (da3 env)
# Stage 2: Evaluate all 6 v2 configs (langsplat_v2 env)
# Usage: bash run_novel_eval.sh
set -e

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
WS_BASE=/mnt/raptor/daiwei/LangSplat-workspace

# Two Python environments
PFX_LANG=/home/daiwei/miniconda3/envs/langsplat_v2
PFX_SAM2=/home/daiwei/miniconda3/envs/da3
export LD_LIBRARY_PATH=$PFX_LANG/lib:$LD_LIBRARY_PATH
PY_LANG=$PFX_LANG/bin/python
PY_SAM2=$PFX_SAM2/bin/python

# Data paths (read-only)
ADT_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/ADT/Apartment_release_clean_seq131_M1292
HDEPIC_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250

# GPUs
GPUS=(0 4 6 7)

# Text weights
TEXT_WEIGHTS=(0.0 0.5 1.0)

# Config arrays (6 configs: ADT×3weights + HDEPIC×3weights)
DATASETS=(ADT ADT ADT HDEPIC HDEPIC HDEPIC)
DATA_ROOTS=($ADT_DATA $ADT_DATA $ADT_DATA $HDEPIC_DATA $HDEPIC_DATA $HDEPIC_DATA)
WEIGHTS=(0.0 0.5 1.0 0.0 0.5 1.0)
DS_DIRS=(ADT_seq131 ADT_seq131 ADT_seq131 HDEPIC_P01 HDEPIC_P01 HDEPIC_P01)

# Workspace paths per config
declare -a WS_PATHS
for i in 0 1 2 3 4 5; do
  WS_PATHS[$i]=$WS_BASE/v2_sam2_tw${WEIGHTS[$i]}/${DS_DIRS[$i]}
done

# AE checkpoint paths
declare -a AE_CKPTS
for i in 0 1 2 3 4 5; do
  AE_CKPTS[$i]=$LANGSPLAT/autoencoder/ckpt/${DS_DIRS[$i]}_tw${WEIGHTS[$i]}/best_ckpt.pth
done

# GT masks output (shared per dataset, not per text_weight)
NOVEL_MASKS_BASE=$WS_BASE/v2_sam2_shared
ADT_NOVEL_MASKS=$NOVEL_MASKS_BASE/ADT_novel_masks
HDEPIC_NOVEL_MASKS=$NOVEL_MASKS_BASE/HDEPIC_novel_masks
NOVEL_MASK_DIRS=($ADT_NOVEL_MASKS $ADT_NOVEL_MASKS $ADT_NOVEL_MASKS \
                 $HDEPIC_NOVEL_MASKS $HDEPIC_NOVEL_MASKS $HDEPIC_NOVEL_MASKS)

# Log directory
LOG_DIR=$WS_BASE/v2_logs/novel_eval
mkdir -p $LOG_DIR $ADT_NOVEL_MASKS $HDEPIC_NOVEL_MASKS

cd $LANGSPLAT

echo "$(date) =========================================="
echo "  Novel-View Segmentation Evaluation"
echo "  Configs: 6 (2 datasets x 3 text_weights)"
echo "  GPUs: ${GPUS[@]}"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 1: Generate GT SAM2 masks for novel views ==="
# =========================================================================

CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_SAM2 generate_novel_gt_masks.py \
  --captions_json $ADT_DATA/vlm-captions/captions.json \
  --rgb_dir $ADT_DATA/vlm-data/moved_050/rgb \
  --output_dir $ADT_NOVEL_MASKS \
  2>&1 | tee $LOG_DIR/stage1_ADT.log &
PID1=$!

CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY_SAM2 generate_novel_gt_masks.py \
  --captions_json $HDEPIC_DATA/vlm-captions/captions.json \
  --rgb_dir $HDEPIC_DATA/vlm-data/moved_050/rgb \
  --output_dir $HDEPIC_NOVEL_MASKS \
  2>&1 | tee $LOG_DIR/stage1_HDEPIC.log &
PID2=$!

wait $PID1 $PID2
echo "$(date) GT mask generation complete"

# =========================================================================
echo "$(date) === STAGE 2: Evaluate novel views ==="
# =========================================================================

run_novel_eval() {
  local idx=$1 gpu=$2
  local ws=${WS_PATHS[$idx]}
  local dr=${DATA_ROOTS[$idx]}

  CUDA_VISIBLE_DEVICES=$gpu $PY_LANG eval_novel_views.py \
    --workspace $ws \
    --ae_ckpt ${AE_CKPTS[$idx]} \
    --metadata_json $dr/vlm-data/moved_050/metadata.json \
    --captions_json $dr/vlm-captions/captions.json \
    --gt_masks_dir ${NOVEL_MASK_DIRS[$idx]} \
    --moved_rgb_dir $dr/vlm-data/moved_050/rgb \
    --num_vis_frames 10
}

# Round 1: 4 configs on 4 GPUs
for gi in 0 1 2 3; do
  run_novel_eval $gi ${GPUS[$gi]} \
    2>&1 | tee $LOG_DIR/stage2_${DS_DIRS[$gi]}_tw${WEIGHTS[$gi]}.log &
  eval "PID_NE_$gi=\$!"
done
wait $PID_NE_0 $PID_NE_1 $PID_NE_2 $PID_NE_3
echo "$(date) Round 1 complete"

# Round 2: remaining 2 configs
for gi in 0 1; do
  i=$((gi + 4))
  run_novel_eval $i ${GPUS[$gi]} \
    2>&1 | tee $LOG_DIR/stage2_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_NE2_$gi=\$!"
done
wait $PID_NE2_0 $PID_NE2_1
echo "$(date) Round 2 complete"

# =========================================================================
echo ""
echo "$(date) =========================================="
echo "  NOVEL VIEW EVALUATION COMPLETE"
echo "  Results in each workspace's eval_novel_results/"
echo "  Logs: $LOG_DIR/"
echo "=========================================="
