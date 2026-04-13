#!/bin/bash
# Master pipeline: LangSplat v2 with SAM2 + Text Feature Blending
# Runs comprehensive experiments: 2 datasets × 3 text_weights = 6 configs
# Usage: bash run_ego3dvqa_pipeline_v2.sh
set -e

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
WS_BASE=/mnt/raptor/daiwei/LangSplat-workspace

# Two Python environments
PFX_LANG=/home/daiwei/miniconda3/envs/langsplat_v2
PFX_SAM2=/home/daiwei/miniconda3/envs/da3
export LD_LIBRARY_PATH=$PFX_LANG/lib:$LD_LIBRARY_PATH
PY_LANG=$PFX_LANG/bin/python
PY_SAM2=$PFX_SAM2/bin/python

# Data paths
ADT_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/ADT/Apartment_release_clean_seq131_M1292
HDEPIC_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250
ADT_CAPTIONS=$ADT_DATA/vlm-captions/captions.json
HDEPIC_CAPTIONS=$HDEPIC_DATA/vlm-captions/captions.json
ADT_CKPT=$ADT_DATA/gs-output/chkpnt45000.pth
HDEPIC_CKPT=$HDEPIC_DATA/gs-output/chkpnt45000.pth

# Shared SAM2 masks output
SHARED=$WS_BASE/v2_sam2_shared
ADT_MASKS=$SHARED/ADT_masks
HDEPIC_MASKS=$SHARED/HDEPIC_masks

# GPUs
GPUS=(0 4 6 7)

# Text weights to test
TEXT_WEIGHTS=(0.0 0.5 1.0)

# Config arrays for batch scheduling
# 6 configs: ADT×3weights + HDEPIC×3weights
DATASETS=(ADT ADT ADT HDEPIC HDEPIC HDEPIC)
DATA_ROOTS=($ADT_DATA $ADT_DATA $ADT_DATA $HDEPIC_DATA $HDEPIC_DATA $HDEPIC_DATA)
CAPTIONS_FILES=($ADT_CAPTIONS $ADT_CAPTIONS $ADT_CAPTIONS $HDEPIC_CAPTIONS $HDEPIC_CAPTIONS $HDEPIC_CAPTIONS)
MASK_DIRS=($ADT_MASKS $ADT_MASKS $ADT_MASKS $HDEPIC_MASKS $HDEPIC_MASKS $HDEPIC_MASKS)
CKPTS=($ADT_CKPT $ADT_CKPT $ADT_CKPT $HDEPIC_CKPT $HDEPIC_CKPT $HDEPIC_CKPT)
WEIGHTS=(0.0 0.5 1.0 0.0 0.5 1.0)
DS_DIRS=(ADT_seq131 ADT_seq131 ADT_seq131 HDEPIC_P01 HDEPIC_P01 HDEPIC_P01)

# Workspace paths per config
declare -a WS_PATHS
for i in 0 1 2 3 4 5; do
  WS_PATHS[$i]=$WS_BASE/v2_sam2_tw${WEIGHTS[$i]}/${DS_DIRS[$i]}
done

# AE dataset names per config
declare -a AE_NAMES
for i in 0 1 2 3 4 5; do
  AE_NAMES[$i]=${DS_DIRS[$i]}_tw${WEIGHTS[$i]}
done

# Create log directory
LOG_DIR=$WS_BASE/v2_logs
mkdir -p $LOG_DIR $SHARED

cd $LANGSPLAT

echo "$(date) =========================================="
echo "  LangSplat v2: SAM2 + Text Feature Blending"
echo "  Configs: ${#WEIGHTS[@]} (2 datasets × 3 text_weights)"
echo "  GPUs: ${GPUS[@]}"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 1: Generate SAM2 Masks (da3 env) ==="
# =========================================================================
mkdir -p $ADT_MASKS $HDEPIC_MASKS

CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_SAM2 generate_sam2_masks.py \
  --captions_json $ADT_CAPTIONS --data_root $ADT_DATA --output_dir $ADT_MASKS \
  2>&1 | tee $LOG_DIR/stage1_ADT.log &
PID1=$!

CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY_SAM2 generate_sam2_masks.py \
  --captions_json $HDEPIC_CAPTIONS --data_root $HDEPIC_DATA --output_dir $HDEPIC_MASKS \
  2>&1 | tee $LOG_DIR/stage1_HDEPIC.log &
PID2=$!

wait $PID1 $PID2
echo "$(date) SAM2 mask generation complete"

# =========================================================================
echo "$(date) === STAGE 2: Prepare Workspaces ==="
# =========================================================================
for tw in ${TEXT_WEIGHTS[@]}; do
  for ds_idx in 0 1; do
    if [ $ds_idx -eq 0 ]; then
      DR=$ADT_DATA; CAP=$ADT_CAPTIONS; DS=ADT_seq131
    else
      DR=$HDEPIC_DATA; CAP=$HDEPIC_CAPTIONS; DS=HDEPIC_P01
    fi
    WS=$WS_BASE/v2_sam2_tw${tw}/$DS
    echo "  Preparing $WS..."
    $PY_LANG prepare_ego3dvqa_workspace_v2.py \
      --data_root $DR --workspace $WS --captions_json $CAP \
      2>&1 | tee $LOG_DIR/stage2_${DS}_tw${tw}.log
  done
done
echo "$(date) Workspaces ready"

# =========================================================================
echo "$(date) === STAGE 3: CLIP Encoding + Text Blending ==="
# =========================================================================
# Round 1: 4 configs on 4 GPUs
for gi in 0 1 2 3; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG preprocess_sam2_ego3dvqa.py \
    --segments_json ${MASK_DIRS[$i]}/segments.json \
    --images_dir ${WS_PATHS[$i]}/images \
    --output_dir ${WS_PATHS[$i]}/language_features \
    --viz_dir ${WS_PATHS[$i]}/diagnostics \
    --text_weight ${WEIGHTS[$i]} \
    2>&1 | tee $LOG_DIR/stage3_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_S3_$gi=\$!"
done
wait $PID_S3_0 $PID_S3_1 $PID_S3_2 $PID_S3_3
echo "$(date) CLIP round 1 complete"

# Round 2: 2 configs on 2 GPUs
for gi in 0 1; do
  i=$((gi + 4))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG preprocess_sam2_ego3dvqa.py \
    --segments_json ${MASK_DIRS[$i]}/segments.json \
    --images_dir ${WS_PATHS[$i]}/images \
    --output_dir ${WS_PATHS[$i]}/language_features \
    --viz_dir ${WS_PATHS[$i]}/diagnostics \
    --text_weight ${WEIGHTS[$i]} \
    2>&1 | tee $LOG_DIR/stage3_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_S3R2_$gi=\$!"
done
wait $PID_S3R2_0 $PID_S3R2_1
echo "$(date) CLIP encoding complete"

# =========================================================================
echo "$(date) === STAGE 4: Post-process Seg Maps ==="
# =========================================================================
for i in 0 1 2 3 4 5; do
  $PY_LANG postprocess_segmaps.py --workspace ${WS_PATHS[$i]} \
    2>&1 | tee $LOG_DIR/stage4_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log
done
echo "$(date) Post-processing complete"

# =========================================================================
echo "$(date) === STAGE 5: Autoencoder Train + Compress ==="
# =========================================================================
cd $LANGSPLAT/autoencoder

# Training round 1: 4 configs
for gi in 0 1 2 3; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG train.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    --num_epochs 100 --lr 0.0007 \
    2>&1 | tee $LOG_DIR/stage5_train_${AE_NAMES[$i]}.log &
  eval "PID_AE_$gi=\$!"
done
wait $PID_AE_0 $PID_AE_1 $PID_AE_2 $PID_AE_3

# Training round 2: 2 configs
for gi in 0 1; do
  i=$((gi + 4))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG train.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    --num_epochs 100 --lr 0.0007 \
    2>&1 | tee $LOG_DIR/stage5_train_${AE_NAMES[$i]}.log &
  eval "PID_AE2_$gi=\$!"
done
wait $PID_AE2_0 $PID_AE2_1
echo "$(date) Autoencoder training complete"

# Compression round 1
for gi in 0 1 2 3; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG test.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    2>&1 | tee $LOG_DIR/stage5_compress_${AE_NAMES[$i]}.log &
  eval "PID_AC_$gi=\$!"
done
wait $PID_AC_0 $PID_AC_1 $PID_AC_2 $PID_AC_3

# Compression round 2
for gi in 0 1; do
  i=$((gi + 4))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG test.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    2>&1 | tee $LOG_DIR/stage5_compress_${AE_NAMES[$i]}.log &
  eval "PID_AC2_$gi=\$!"
done
wait $PID_AC2_0 $PID_AC2_1
echo "$(date) Autoencoder compression complete"

# =========================================================================
echo "$(date) === STAGE 6: Language Feature Training ==="
# =========================================================================
cd $LANGSPLAT

# Round 1: 4 configs on 4 GPUs
for gi in 0 1 2 3; do
  i=$gi
  PORT=$((55600 + i))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG train.py \
    -s ${WS_PATHS[$i]} -m ${WS_PATHS[$i]}/output \
    --start_checkpoint ${CKPTS[$i]} --feature_level 1 \
    --resolution 1 --test_iterations -1 --port $PORT \
    2>&1 | tee $LOG_DIR/stage6_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_T_$gi=\$!"
done
wait $PID_T_0 $PID_T_1 $PID_T_2 $PID_T_3
echo "$(date) Training round 1 complete"

# Round 2: 2 configs
for gi in 0 1; do
  i=$((gi + 4))
  PORT=$((55600 + i))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG train.py \
    -s ${WS_PATHS[$i]} -m ${WS_PATHS[$i]}/output \
    --start_checkpoint ${CKPTS[$i]} --feature_level 1 \
    --resolution 1 --test_iterations -1 --port $PORT \
    2>&1 | tee $LOG_DIR/stage6_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_T2_$gi=\$!"
done
wait $PID_T2_0 $PID_T2_1
echo "$(date) All feature training complete"

# =========================================================================
echo "$(date) === STAGE 7: Render ==="
# =========================================================================
# Round 1
for gi in 0 1 2 3; do
  i=$gi
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG render.py \
    -m ${WS_PATHS[$i]}/output_1 --include_feature --skip_test \
    2>&1 | tee $LOG_DIR/stage7_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_R_$gi=\$!"
done
wait $PID_R_0 $PID_R_1 $PID_R_2 $PID_R_3

# Round 2
for gi in 0 1; do
  i=$((gi + 4))
  CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY_LANG render.py \
    -m ${WS_PATHS[$i]}/output_1 --include_feature --skip_test \
    2>&1 | tee $LOG_DIR/stage7_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_R2_$gi=\$!"
done
wait $PID_R2_0 $PID_R2_1
echo "$(date) Rendering complete"

# =========================================================================
echo "$(date) === STAGE 8: Evaluation ==="
# =========================================================================
# Prompts per dataset
ADT_PROMPTS=("kitchen counter" "refrigerator" "sofa" "bookshelf" "sink" "table")
HDEPIC_PROMPTS=("cutting board" "knife" "pan" "stove" "kettle" "fridge")

run_eval() {
  local idx=$1 gpu=$2
  local ae=$LANGSPLAT/autoencoder/ckpt/${AE_NAMES[$idx]}/best_ckpt.pth
  if [ "${DATASETS[$idx]}" == "ADT" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY_LANG eval_ego3dvqa.py \
      --workspace ${WS_PATHS[$idx]} --ae_ckpt $ae \
      --prompts "${ADT_PROMPTS[@]}" --num_vis_frames 10
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY_LANG eval_ego3dvqa.py \
      --workspace ${WS_PATHS[$idx]} --ae_ckpt $ae \
      --prompts "${HDEPIC_PROMPTS[@]}" --num_vis_frames 10
  fi
}

# Round 1
for gi in 0 1 2 3; do
  run_eval $gi ${GPUS[$gi]} \
    2>&1 | tee $LOG_DIR/stage8_${DS_DIRS[$gi]}_tw${WEIGHTS[$gi]}.log &
  eval "PID_E_$gi=\$!"
done
wait $PID_E_0 $PID_E_1 $PID_E_2 $PID_E_3

# Round 2
for gi in 0 1; do
  i=$((gi + 4))
  run_eval $i ${GPUS[$gi]} \
    2>&1 | tee $LOG_DIR/stage8_${DS_DIRS[$i]}_tw${WEIGHTS[$i]}.log &
  eval "PID_E2_$gi=\$!"
done
wait $PID_E2_0 $PID_E2_1
echo "$(date) Evaluation complete"

# =========================================================================
echo "$(date) === STAGE 9: Generate Report ==="
# =========================================================================
$PY_LANG generate_v2_report.py --ws_base $WS_BASE \
  2>&1 | tee $LOG_DIR/stage9_report.log

echo ""
echo "$(date) =========================================="
echo "  PIPELINE COMPLETE"
echo "  Results: $WS_BASE/v2_report.md"
echo "  Logs: $LOG_DIR/"
echo "=========================================="
