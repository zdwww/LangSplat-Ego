#!/bin/bash
# Pipeline v4_lerp: Qwen3-VL LERP blend (separate image + text, then LERP)
# Single config: lerp tw=0.5 × HD-EPIC on 1 GPU
# Usage: bash run_ego3dvqa_pipeline_v4_lerp.sh
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

# Config
GPU=6
TW=0.5
DS_DIR=HDEPIC_P01
WS=$WS_BASE/v4_qwen3vl_lerp_tw${TW}/$DS_DIR
AE_NAME=${DS_DIR}_v4_lerp_tw${TW}
AE_CKPT=$LANGSPLAT/autoencoder/ckpt/${AE_NAME}/best_ckpt.pth

# Log directory
LOG_DIR=$WS_BASE/v4_lerp_logs
mkdir -p $LOG_DIR

cd $LANGSPLAT

echo "$(date) =========================================="
echo "  LangSplat v4_lerp: Qwen3-VL LERP Blend"
echo "  Config: lerp tw=${TW} × HD-EPIC"
echo "  GPU: ${GPU}"
echo "  Workspace: ${WS}"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 1: Prepare Workspace ==="
# =========================================================================
echo "  Preparing $WS..."
$PY_LANG prepare_ego3dvqa_workspace_v2.py \
  --data_root $HDEPIC_DATA --workspace $WS --captions_json $HDEPIC_CAPTIONS \
  2>&1 | tee $LOG_DIR/stage1.log
echo "$(date) Workspace ready"

# =========================================================================
echo "$(date) === STAGE 2: Qwen3-VL LERP Preprocessing ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=$GPU $PY_QWEN preprocess_qwen3vl_ego3dvqa.py \
  --segments_json $HDEPIC_MASKS/segments.json \
  --images_dir $WS/images \
  --output_dir $WS/language_features \
  --viz_dir $WS/diagnostics \
  --encode_mode lerp \
  --text_weight $TW \
  --embed_dim 512 \
  --batch_size 4 \
  2>&1 | tee $LOG_DIR/stage2.log
echo "$(date) LERP preprocessing complete"

# =========================================================================
echo "$(date) === STAGE 3: Post-process Seg Maps ==="
# =========================================================================
$PY_LANG postprocess_segmaps.py --workspace $WS \
  2>&1 | tee $LOG_DIR/stage3.log
echo "$(date) Post-processing complete"

# =========================================================================
echo "$(date) === STAGE 4: Autoencoder Train + Compress ==="
# =========================================================================
cd $LANGSPLAT/autoencoder

CUDA_VISIBLE_DEVICES=$GPU $PY_LANG train.py \
  --dataset_path $WS --dataset_name $AE_NAME \
  --num_epochs 100 --lr 0.0007 \
  2>&1 | tee $LOG_DIR/stage4_train.log
echo "$(date) Autoencoder training complete"

CUDA_VISIBLE_DEVICES=$GPU $PY_LANG test.py \
  --dataset_path $WS --dataset_name $AE_NAME \
  2>&1 | tee $LOG_DIR/stage4_compress.log
echo "$(date) Autoencoder compression complete"

# =========================================================================
echo "$(date) === STAGE 5: Language Feature Training ==="
# =========================================================================
cd $LANGSPLAT

CUDA_VISIBLE_DEVICES=$GPU $PY_LANG train.py \
  -s $WS -m $WS/output \
  --start_checkpoint $HDEPIC_CKPT --feature_level 1 \
  --resolution 1 --test_iterations -1 --port 55630 \
  2>&1 | tee $LOG_DIR/stage5.log
echo "$(date) Feature training complete"

# =========================================================================
echo "$(date) === STAGE 6: Render ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=$GPU $PY_LANG render.py \
  -m $WS/output_1 --include_feature --skip_test \
  2>&1 | tee $LOG_DIR/stage6.log
echo "$(date) Rendering complete"

# =========================================================================
echo "$(date) === STAGE 7: Novel-View Evaluation ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=$GPU $PY_QWEN $LANGSPLAT/eval_novel_views.py \
  --workspace $WS \
  --ae_ckpt $AE_CKPT \
  --metadata_json $HDEPIC_DATA/vlm-data/moved_050/metadata.json \
  --captions_json $HDEPIC_CAPTIONS \
  --gt_masks_dir $HDEPIC_NOVEL_MASKS \
  --moved_rgb_dir $HDEPIC_DATA/vlm-data/moved_050/rgb \
  --num_vis_frames 10 \
  --encoder_type qwen3vl \
  --embed_dim 512 \
  2>&1 | tee $LOG_DIR/stage7.log
echo "$(date) Novel-view evaluation complete"

# =========================================================================
echo "$(date) === STAGE 8: Generate Report ==="
# =========================================================================
$PY_LANG $LANGSPLAT/generate_v4_report.py --ws_base $WS_BASE \
  2>&1 | tee $LOG_DIR/stage8_report.log

echo ""
echo "$(date) =========================================="
echo "  V4_LERP PIPELINE COMPLETE"
echo "  Report: $LANGSPLAT/docs/experiments/04-10_v4-qwen3vl-embedding-results.md"
echo "  Logs: $LOG_DIR/"
echo "=========================================="
