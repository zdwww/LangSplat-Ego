#!/bin/bash
# Pipeline v4: Qwen3-VL-Embedding (unified multimodal embeddings)
# Runs 2 encode modes (image_only, multimodal) × 1 dataset (HD-EPIC) = 2 configs on 2 GPUs
# Usage: bash run_ego3dvqa_pipeline_v4.sh
set -e

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
WS_BASE=/mnt/raptor/daiwei/LangSplat-workspace

# Python environments
PFX_LANG=/home/daiwei/miniconda3/envs/langsplat_v2
export LD_LIBRARY_PATH=$PFX_LANG/lib:$LD_LIBRARY_PATH
PY_LANG=$PFX_LANG/bin/python

PFX_QWEN=/home/daiwei/miniconda3/envs/qwen3vl
PY_QWEN=$PFX_QWEN/bin/python
QWEN_SRC=/home/daiwei/Ego3DVQA-GS/Qwen3-VL-Embedding/src
export PYTHONPATH=$QWEN_SRC:$LANGSPLAT:$PYTHONPATH

# Data paths (HD-EPIC only)
HDEPIC_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250
HDEPIC_CAPTIONS=$HDEPIC_DATA/vlm-captions/captions.json
HDEPIC_CKPT=$HDEPIC_DATA/gs-output/chkpnt45000.pth

# Shared SAM2 masks and novel GT masks (reuse from v2)
SHARED=$WS_BASE/v2_sam2_shared
HDEPIC_MASKS=$SHARED/HDEPIC_masks
HDEPIC_NOVEL_MASKS=$SHARED/HDEPIC_novel_masks

# GPUs
GPUS=(6 7)

# 2 configs: image_only and multimodal
MODES=(image_only multimodal)
DS_DIR=HDEPIC_P01

# Workspace paths
declare -a WS_PATHS
for i in 0 1; do
  WS_PATHS[$i]=$WS_BASE/v4_qwen3vl_${MODES[$i]}/$DS_DIR
done

# AE dataset names
declare -a AE_NAMES
for i in 0 1; do
  AE_NAMES[$i]=${DS_DIR}_v4_${MODES[$i]}
done

# AE checkpoint paths
declare -a AE_CKPTS
for i in 0 1; do
  AE_CKPTS[$i]=$LANGSPLAT/autoencoder/ckpt/${AE_NAMES[$i]}/best_ckpt.pth
done

# Log directory
LOG_DIR=$WS_BASE/v4_logs
mkdir -p $LOG_DIR

cd $LANGSPLAT

echo "$(date) =========================================="
echo "  LangSplat v4: Qwen3-VL-Embedding"
echo "  Configs: 2 (image_only, multimodal) × HD-EPIC"
echo "  GPUs: ${GPUS[@]}"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 1: Prepare Workspaces ==="
# =========================================================================
for i in 0 1; do
  WS=${WS_PATHS[$i]}
  echo "  Preparing $WS..."
  $PY_LANG prepare_ego3dvqa_workspace_v2.py \
    --data_root $HDEPIC_DATA --workspace $WS --captions_json $HDEPIC_CAPTIONS \
    2>&1 | tee $LOG_DIR/stage1_${MODES[$i]}.log
done
echo "$(date) Workspaces ready"

# =========================================================================
echo "$(date) === STAGE 2: Qwen3-VL-Embedding Preprocessing ==="
# =========================================================================
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_QWEN preprocess_qwen3vl_ego3dvqa.py \
    --segments_json $HDEPIC_MASKS/segments.json \
    --images_dir ${WS_PATHS[$i]}/images \
    --output_dir ${WS_PATHS[$i]}/language_features \
    --viz_dir ${WS_PATHS[$i]}/diagnostics \
    --encode_mode ${MODES[$i]} \
    --embed_dim 512 \
    --batch_size 4 \
    2>&1 | tee $LOG_DIR/stage2_${MODES[$i]}.log &
  eval "PID_S2_$i=\$!"
done
wait $PID_S2_0 $PID_S2_1
echo "$(date) Qwen3-VL preprocessing complete"

# =========================================================================
echo "$(date) === STAGE 3: Post-process Seg Maps ==="
# =========================================================================
for i in 0 1; do
  $PY_LANG postprocess_segmaps.py --workspace ${WS_PATHS[$i]} \
    2>&1 | tee $LOG_DIR/stage3_${MODES[$i]}.log
done
echo "$(date) Post-processing complete"

# =========================================================================
echo "$(date) === STAGE 4: Autoencoder Train + Compress ==="
# =========================================================================
cd $LANGSPLAT/autoencoder

for i in 0 1; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_LANG train.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    --num_epochs 100 --lr 0.0007 \
    2>&1 | tee $LOG_DIR/stage4_train_${AE_NAMES[$i]}.log &
  eval "PID_AE_$i=\$!"
done
wait $PID_AE_0 $PID_AE_1
echo "$(date) Autoencoder training complete"

for i in 0 1; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_LANG test.py \
    --dataset_path ${WS_PATHS[$i]} --dataset_name ${AE_NAMES[$i]} \
    2>&1 | tee $LOG_DIR/stage4_compress_${AE_NAMES[$i]}.log &
  eval "PID_AC_$i=\$!"
done
wait $PID_AC_0 $PID_AC_1
echo "$(date) Autoencoder compression complete"

# =========================================================================
echo "$(date) === STAGE 5: Language Feature Training ==="
# =========================================================================
cd $LANGSPLAT

for i in 0 1; do
  PORT=$((55620 + i))
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_LANG train.py \
    -s ${WS_PATHS[$i]} -m ${WS_PATHS[$i]}/output \
    --start_checkpoint $HDEPIC_CKPT --feature_level 1 \
    --resolution 1 --test_iterations -1 --port $PORT \
    2>&1 | tee $LOG_DIR/stage5_${MODES[$i]}.log &
  eval "PID_T_$i=\$!"
done
wait $PID_T_0 $PID_T_1
echo "$(date) Feature training complete"

# =========================================================================
echo "$(date) === STAGE 6: Render ==="
# =========================================================================
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_LANG render.py \
    -m ${WS_PATHS[$i]}/output_1 --include_feature --skip_test \
    2>&1 | tee $LOG_DIR/stage6_${MODES[$i]}.log &
  eval "PID_R_$i=\$!"
done
wait $PID_R_0 $PID_R_1
echo "$(date) Rendering complete"

# =========================================================================
echo "$(date) === STAGE 7: Novel-View Evaluation ==="
# =========================================================================
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_QWEN $LANGSPLAT/eval_novel_views.py \
    --workspace ${WS_PATHS[$i]} \
    --ae_ckpt ${AE_CKPTS[$i]} \
    --metadata_json $HDEPIC_DATA/vlm-data/moved_050/metadata.json \
    --captions_json $HDEPIC_CAPTIONS \
    --gt_masks_dir $HDEPIC_NOVEL_MASKS \
    --moved_rgb_dir $HDEPIC_DATA/vlm-data/moved_050/rgb \
    --num_vis_frames 10 \
    --encoder_type qwen3vl \
    --embed_dim 512 \
    2>&1 | tee $LOG_DIR/stage7_${MODES[$i]}.log &
  eval "PID_NE_$i=\$!"
done
wait $PID_NE_0 $PID_NE_1
echo "$(date) Novel-view evaluation complete"

# =========================================================================
echo "$(date) === STAGE 8: Generate Report ==="
# =========================================================================
$PY_LANG $LANGSPLAT/generate_v4_report.py --ws_base $WS_BASE \
  2>&1 | tee $LOG_DIR/stage8_report.log

echo ""
echo "$(date) =========================================="
echo "  V4 PIPELINE COMPLETE"
echo "  Report: $LANGSPLAT/docs/experiments/04-10_v4-qwen3vl-embedding-results.md"
echo "  Logs: $LOG_DIR/"
echo "=========================================="
