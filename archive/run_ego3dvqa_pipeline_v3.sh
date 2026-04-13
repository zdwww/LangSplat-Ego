#!/bin/bash
# Pipeline v3: SLERP + Adaptive Per-Segment Text Weights
# Runs 2 max_tw values × 2 datasets = 4 configs on 3 GPUs
# Usage: bash run_ego3dvqa_pipeline_v3.sh
set -e

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
WS_BASE=/mnt/raptor/daiwei/LangSplat-workspace

# Python environment
PFX_LANG=/home/daiwei/miniconda3/envs/langsplat_v2
export LD_LIBRARY_PATH=$PFX_LANG/lib:$LD_LIBRARY_PATH
PY_LANG=$PFX_LANG/bin/python

# Data paths
ADT_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/ADT/Apartment_release_clean_seq131_M1292
HDEPIC_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250
ADT_CAPTIONS=$ADT_DATA/vlm-captions/captions.json
HDEPIC_CAPTIONS=$HDEPIC_DATA/vlm-captions/captions.json
ADT_CKPT=$ADT_DATA/gs-output/chkpnt45000.pth
HDEPIC_CKPT=$HDEPIC_DATA/gs-output/chkpnt45000.pth

# Shared SAM2 masks (reuse from v2)
SHARED=$WS_BASE/v2_sam2_shared
ADT_MASKS=$SHARED/ADT_masks
HDEPIC_MASKS=$SHARED/HDEPIC_masks

# Novel view GT masks (reuse from v2)
ADT_NOVEL_MASKS=$SHARED/ADT_novel_masks
HDEPIC_NOVEL_MASKS=$SHARED/HDEPIC_novel_masks

# GPUs
GPUS=(1 6 7)

# 4 configs: 2 max_tw × 2 datasets
# Index: 0=ADT_max0.5, 1=ADT_max1.0, 2=HDEPIC_max0.5, 3=HDEPIC_max1.0
DATASETS=(ADT ADT HDEPIC HDEPIC)
DATA_ROOTS=($ADT_DATA $ADT_DATA $HDEPIC_DATA $HDEPIC_DATA)
CAPTIONS_FILES=($ADT_CAPTIONS $ADT_CAPTIONS $HDEPIC_CAPTIONS $HDEPIC_CAPTIONS)
MASK_DIRS=($ADT_MASKS $ADT_MASKS $HDEPIC_MASKS $HDEPIC_MASKS)
CKPTS=($ADT_CKPT $ADT_CKPT $HDEPIC_CKPT $HDEPIC_CKPT)
MAX_TWS=(0.5 1.0 0.5 1.0)
DS_DIRS=(ADT_seq131 ADT_seq131 HDEPIC_P01 HDEPIC_P01)
NOVEL_MASK_DIRS=($ADT_NOVEL_MASKS $ADT_NOVEL_MASKS $HDEPIC_NOVEL_MASKS $HDEPIC_NOVEL_MASKS)

# Workspace paths
declare -a WS_PATHS
for i in 0 1 2 3; do
  WS_PATHS[$i]=$WS_BASE/v3_slerp_adaptive_max${MAX_TWS[$i]}/${DS_DIRS[$i]}
done

# AE dataset names
declare -a AE_NAMES
for i in 0 1 2 3; do
  AE_NAMES[$i]=${DS_DIRS[$i]}_v3_max${MAX_TWS[$i]}
done

# AE checkpoint paths
declare -a AE_CKPTS
for i in 0 1 2 3; do
  AE_CKPTS[$i]=$LANGSPLAT/autoencoder/ckpt/${AE_NAMES[$i]}/best_ckpt.pth
done

# Log directory
LOG_DIR=$WS_BASE/v3_logs
mkdir -p $LOG_DIR

cd $LANGSPLAT

echo "$(date) =========================================="
echo "  LangSplat v3: SLERP + Adaptive Per-Segment Weights"
echo "  Configs: 4 (2 datasets × 2 max_tw)"
echo "  GPUs: ${GPUS[@]}"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 1: Prepare Workspaces ==="
# =========================================================================
for i in 0 1 2 3; do
  WS=${WS_PATHS[$i]}
  DR=${DATA_ROOTS[$i]}
  CAP=${CAPTIONS_FILES[$i]}
  echo "  Preparing $WS..."
  $PY_LANG prepare_ego3dvqa_workspace_v2.py \
    --data_root $DR --workspace $WS --captions_json $CAP \
    2>&1 | tee $LOG_DIR/stage1_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log
done
echo "$(date) Workspaces ready"

# =========================================================================
echo "$(date) === STAGE 2: CLIP Encoding + SLERP Adaptive Blending ==="
# =========================================================================
# Round 1: 3 configs on 3 GPUs
for gi in 0 1 2; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG preprocess_sam2_ego3dvqa.py \
    --segments_json ${MASK_DIRS[$i]}/segments.json \
    --images_dir ${WS_PATHS[$i]}/images \
    --output_dir ${WS_PATHS[$i]}/language_features \
    --viz_dir ${WS_PATHS[$i]}/diagnostics \
    --blend_mode slerp_adaptive \
    --max_text_weight ${MAX_TWS[$i]} \
    --text_weight 0.5 \
    2>&1 | tee $LOG_DIR/stage2_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log &
  eval "PID_S2_$gi=\$!"
done
wait $PID_S2_0 $PID_S2_1 $PID_S2_2
echo "$(date) CLIP round 1 complete"

# Round 2: 1 config
i=3
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG preprocess_sam2_ego3dvqa.py \
  --segments_json ${MASK_DIRS[$i]}/segments.json \
  --images_dir ${WS_PATHS[$i]}/images \
  --output_dir ${WS_PATHS[$i]}/language_features \
  --viz_dir ${WS_PATHS[$i]}/diagnostics \
  --blend_mode slerp_adaptive \
  --max_text_weight ${MAX_TWS[$i]} \
  --text_weight 0.5 \
  2>&1 | tee $LOG_DIR/stage2_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log
echo "$(date) CLIP encoding complete"

# =========================================================================
echo "$(date) === STAGE 3: Post-process Seg Maps ==="
# =========================================================================
for i in 0 1 2 3; do
  $PY_LANG postprocess_segmaps.py --workspace ${WS_PATHS[$i]} \
    2>&1 | tee $LOG_DIR/stage3_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log
done
echo "$(date) Post-processing complete"

# =========================================================================
echo "$(date) === STAGE 4: Autoencoder Train + Compress ==="
# =========================================================================
cd $LANGSPLAT/autoencoder

# Training round 1: 3 configs
for gi in 0 1 2; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG train.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    --num_epochs 100 --lr 0.0007 \
    2>&1 | tee $LOG_DIR/stage4_train_${AE_NAMES[$i]}.log &
  eval "PID_AE_$gi=\$!"
done
wait $PID_AE_0 $PID_AE_1 $PID_AE_2

# Training round 2: 1 config
i=3
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG train.py \
  --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
  --num_epochs 100 --lr 0.0007 \
  2>&1 | tee $LOG_DIR/stage4_train_${AE_NAMES[$i]}.log
echo "$(date) Autoencoder training complete"

# Compression round 1: 3 configs
for gi in 0 1 2; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG test.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    2>&1 | tee $LOG_DIR/stage4_compress_${AE_NAMES[$i]}.log &
  eval "PID_AC_$gi=\$!"
done
wait $PID_AC_0 $PID_AC_1 $PID_AC_2

# Compression round 2
i=3
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG test.py \
  --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
  2>&1 | tee $LOG_DIR/stage4_compress_${AE_NAMES[$i]}.log
echo "$(date) Autoencoder compression complete"

# =========================================================================
echo "$(date) === STAGE 5: Language Feature Training ==="
# =========================================================================
cd $LANGSPLAT

# Round 1: 3 configs on 3 GPUs
for gi in 0 1 2; do
  i=$gi
  PORT=$((55610 + i))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG train.py \
    -s ${WS_PATHS[$i]} -m ${WS_PATHS[$i]}/output \
    --start_checkpoint ${CKPTS[$i]} --feature_level 1 \
    --resolution 1 --test_iterations -1 --port $PORT \
    2>&1 | tee $LOG_DIR/stage5_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log &
  eval "PID_T_$gi=\$!"
done
wait $PID_T_0 $PID_T_1 $PID_T_2
echo "$(date) Training round 1 complete"

# Round 2: 1 config
i=3
PORT=$((55610 + i))
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG train.py \
  -s ${WS_PATHS[$i]} -m ${WS_PATHS[$i]}/output \
  --start_checkpoint ${CKPTS[$i]} --feature_level 1 \
  --resolution 1 --test_iterations -1 --port $PORT \
  2>&1 | tee $LOG_DIR/stage5_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log
echo "$(date) All feature training complete"

# =========================================================================
echo "$(date) === STAGE 6: Render ==="
# =========================================================================
# Round 1
for gi in 0 1 2; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG render.py \
    -m ${WS_PATHS[$i]}/output_1 --include_feature --skip_test \
    2>&1 | tee $LOG_DIR/stage6_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log &
  eval "PID_R_$gi=\$!"
done
wait $PID_R_0 $PID_R_1 $PID_R_2

# Round 2
i=3
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG render.py \
  -m ${WS_PATHS[$i]}/output_1 --include_feature --skip_test \
  2>&1 | tee $LOG_DIR/stage6_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log
echo "$(date) Rendering complete"

# =========================================================================
echo "$(date) === STAGE 7: Novel-View Evaluation ==="
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

# Round 1: 3 configs
for gi in 0 1 2; do
  run_novel_eval $gi ${GPUS[$gi]} \
    2>&1 | tee $LOG_DIR/stage7_${DS_DIRS[$gi]}_max${MAX_TWS[$gi]}.log &
  eval "PID_NE_$gi=\$!"
done
wait $PID_NE_0 $PID_NE_1 $PID_NE_2

# Round 2
i=3
run_novel_eval $i ${GPUS[0]} \
  2>&1 | tee $LOG_DIR/stage7_${DS_DIRS[$i]}_max${MAX_TWS[$i]}.log
echo "$(date) Novel-view evaluation complete"

# =========================================================================
echo "$(date) === STAGE 8: Generate Report ==="
# =========================================================================
$PY_LANG $LANGSPLAT/generate_v3_report.py --ws_base $WS_BASE \
  2>&1 | tee $LOG_DIR/stage8_report.log

echo ""
echo "$(date) =========================================="
echo "  V3 PIPELINE COMPLETE"
echo "  Report: $LANGSPLAT/docs/experiments/04-09_v3-slerp-adaptive-results.md"
echo "  Logs: $LOG_DIR/"
echo "=========================================="
