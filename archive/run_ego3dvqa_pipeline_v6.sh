#!/bin/bash
# Pipeline v6: Expanded codebook experiment
# Changes from v5:
#   - 30K iterations (vs v5's 10K, matches v2 training budget)
#   - Codebook size 128 (vs v5's 64, expanded multi-modal capacity)
#   - topk 8 (vs v5's 4, finer sparse representation)
# Note: Resolution kept at 2 (half-res). Resolution 1 was tested but OOM'd at 24GB VRAM
# (even with codebook 64). User confirmed resolution is not the critical factor.
# 2 configs: CLIP codebook + Qwen3-VL codebook × HD-EPIC on 2 GPUs
# Usage: bash run_ego3dvqa_pipeline_v6.sh
set -e

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
LANGSPLATV2=/home/daiwei/LangSplat-variants/LangSplatV2
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
DS_DIR=HDEPIC_P01

# v6 hyperparameters
V6_CODEBOOK_SIZE=128
V6_TOPK=8
V6_ITERATIONS=30000
V6_RESOLUTION=2

# 2 configs
LABELS=("v6_clip_codebook" "v6_qwen_codebook")

# Workspace paths
declare -a WS_PATHS
for i in 0 1; do
  WS_PATHS[$i]=$WS_BASE/${LABELS[$i]}/$DS_DIR
done

# Log directory
LOG_DIR=$WS_BASE/v6_logs
mkdir -p $LOG_DIR

echo "$(date) =========================================="
echo "  LangSplat v6: Expanded Codebook Experiment"
echo "  Config: codebook=$V6_CODEBOOK_SIZE, topk=$V6_TOPK,"
echo "          iters=$V6_ITERATIONS, res=$V6_RESOLUTION"
echo "  Note: Resolution kept at 2 (res=1 OOMs at 24GB VRAM)"
echo "  GPUs: ${GPUS[@]}"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 1: Prepare Workspaces ==="
# =========================================================================
cd $LANGSPLAT
for i in 0 1; do
  WS=${WS_PATHS[$i]}
  echo "  Preparing $WS..."
  $PY_LANG prepare_ego3dvqa_workspace_v2.py \
    --data_root $HDEPIC_DATA --workspace $WS --captions_json $HDEPIC_CAPTIONS \
    2>&1 | tee $LOG_DIR/stage1_${LABELS[$i]}.log
done
echo "$(date) Workspaces ready"

# =========================================================================
echo "$(date) === STAGE 2: Feature Extraction ==="
# =========================================================================

# Config 0: CLIP LERP tw=0.5
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG $LANGSPLAT/preprocess_sam2_ego3dvqa.py \
  --segments_json $HDEPIC_MASKS/segments.json \
  --images_dir ${WS_PATHS[0]}/images \
  --output_dir ${WS_PATHS[0]}/language_features \
  --viz_dir ${WS_PATHS[0]}/diagnostics \
  --text_weight 0.5 \
  --batch_size 16 \
  2>&1 | tee $LOG_DIR/stage2_clip.log &
PID_S2_0=$!

# Config 1: Qwen3-VL LERP tw=0.5
CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY_QWEN $LANGSPLAT/preprocess_qwen3vl_ego3dvqa.py \
  --segments_json $HDEPIC_MASKS/segments.json \
  --images_dir ${WS_PATHS[1]}/images \
  --output_dir ${WS_PATHS[1]}/language_features \
  --viz_dir ${WS_PATHS[1]}/diagnostics \
  --encode_mode lerp \
  --text_weight 0.5 \
  --embed_dim 512 \
  --batch_size 4 \
  2>&1 | tee $LOG_DIR/stage2_qwen.log &
PID_S2_1=$!

wait $PID_S2_0 $PID_S2_1
echo "$(date) Feature extraction complete"

# =========================================================================
echo "$(date) === STAGE 3: Post-process Seg Maps ==="
# =========================================================================
cd $LANGSPLAT
for i in 0 1; do
  $PY_LANG postprocess_segmaps.py --workspace ${WS_PATHS[$i]} \
    2>&1 | tee $LOG_DIR/stage3_${LABELS[$i]}.log
done
echo "$(date) Post-processing complete"

# =========================================================================
echo "$(date) === STAGE 4: SKIP (no autoencoder needed for codebook) ==="
# =========================================================================

# =========================================================================
echo "$(date) === STAGE 5: Codebook Training (LangSplatV2) ==="
# =========================================================================
cd $LANGSPLATV2

for i in 0 1; do
  PORT=$((55650 + i))
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY_LANG train.py \
    -s ${WS_PATHS[$i]} -m ${WS_PATHS[$i]}/output \
    --start_checkpoint $HDEPIC_CKPT \
    --include_feature --feature_level 1 \
    --resolution $V6_RESOLUTION --test_iterations -1 \
    --iterations $V6_ITERATIONS \
    --l1_loss --normalize --topk $V6_TOPK \
    --codebook_size $V6_CODEBOOK_SIZE --vq_layer_num 1 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    --port $PORT \
    2>&1 | tee $LOG_DIR/stage5_${LABELS[$i]}.log &
  eval "PID_T_$i=\$!"
done
wait $PID_T_0 $PID_T_1
echo "$(date) Codebook training complete"

# =========================================================================
echo "$(date) === STAGE 6: Novel-View Evaluation ==="
# =========================================================================
cd $LANGSPLAT

# Pre-compute Qwen3-VL text embeddings (needed for stage 6 config 1)
$PY_QWEN precompute_qwen3vl_text_embeds.py \
  --captions_json $HDEPIC_CAPTIONS \
  --metadata_json $HDEPIC_DATA/vlm-data/moved_050/metadata.json \
  --output $LOG_DIR/qwen3vl_text_embeds.npz \
  2>&1 | tee $LOG_DIR/stage6_precompute.log

# Config 0: CLIP codebook (uses CLIP text encoder in langsplat_v2 env)
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY_LANG eval_novel_views_codebook.py \
  --workspace ${WS_PATHS[0]} \
  --langsplatv2_dir $LANGSPLATV2 \
  --metadata_json $HDEPIC_DATA/vlm-data/moved_050/metadata.json \
  --captions_json $HDEPIC_CAPTIONS \
  --gt_masks_dir $HDEPIC_NOVEL_MASKS \
  --moved_rgb_dir $HDEPIC_DATA/vlm-data/moved_050/rgb \
  --num_vis_frames 10 \
  --encoder_type clip \
  --embed_dim 512 \
  --topk $V6_TOPK \
  2>&1 | tee $LOG_DIR/stage6_clip.log &
PID_E_0=$!

# Config 1: Qwen3-VL codebook (uses precomputed embeddings in langsplat_v2 env)
CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY_LANG eval_novel_views_codebook.py \
  --workspace ${WS_PATHS[1]} \
  --langsplatv2_dir $LANGSPLATV2 \
  --metadata_json $HDEPIC_DATA/vlm-data/moved_050/metadata.json \
  --captions_json $HDEPIC_CAPTIONS \
  --gt_masks_dir $HDEPIC_NOVEL_MASKS \
  --moved_rgb_dir $HDEPIC_DATA/vlm-data/moved_050/rgb \
  --num_vis_frames 10 \
  --encoder_type precomputed \
  --precomputed_text_embeds $LOG_DIR/qwen3vl_text_embeds.npz \
  --embed_dim 512 \
  --topk $V6_TOPK \
  2>&1 | tee $LOG_DIR/stage6_qwen.log &
PID_E_1=$!

wait $PID_E_0 $PID_E_1
echo "$(date) Novel-view evaluation complete"

# =========================================================================
echo "$(date) === STAGE 7: Generate Report ==="
# =========================================================================
cd $LANGSPLAT
$PY_LANG generate_v6_report.py --ws_base $WS_BASE \
  2>&1 | tee $LOG_DIR/stage7_report.log

echo ""
echo "$(date) =========================================="
echo "  V6 PIPELINE COMPLETE"
echo "  Report: $LANGSPLAT/docs/experiments/04-10_v6-codebook-fullscale-results.md"
echo "  Logs: $LOG_DIR/"
echo "=========================================="
