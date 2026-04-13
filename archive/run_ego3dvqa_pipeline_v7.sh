#!/bin/bash
# Pipeline v7: Detailed text descriptions experiment
#
# Same architecture as primary pipeline (CLIP CB-64, top-4, tw=0.5) but uses
# vlm-captions-w-desc/ where render-branch description != category (more detailed).
# Tests whether richer text during CLIP feature extraction improves AP.
#
# Differences from primary pipeline:
#   - Captions: vlm-captions-w-desc/ (render descriptions are detailed)
#   - SAM2 masks: generated fresh (bboxes differ from old captions)
#   - Text field: --text_field description (uses detailed description for CLIP)
#   - Workspace: v7_detailed_desc/
#
# Usage:
#   bash run_ego3dvqa_pipeline_v7.sh                 # full run
#   bash run_ego3dvqa_pipeline_v7.sh --gpu 4         # specify GPU
#   bash run_ego3dvqa_pipeline_v7.sh --skip_masks    # skip SAM2 mask generation
set -e

# ── Arguments ───────────────────────────────────────────────────────────
GPU=6
SKIP_MASKS=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)         GPU="$2";      shift 2;;
    --skip_masks)  SKIP_MASKS=true; shift;;
    *)             echo "Unknown arg: $1"; exit 1;;
  esac
done

# ── Paths ───────────────────────────────────────────────────────────────
LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
LANGSPLATV2=/home/daiwei/LangSplat-variants/LangSplatV2
WS_BASE=/mnt/raptor/daiwei/LangSplat-workspace

PFX=/home/daiwei/miniconda3/envs/langsplat_v2
export LD_LIBRARY_PATH=$PFX/lib:$LD_LIBRARY_PATH
PY=$PFX/bin/python

# SAM2 mask generation uses da3 env
PFX_DA3=/home/daiwei/miniconda3/envs/da3
PY_DA3=$PFX_DA3/bin/python

# ── Dataset config (HD-EPIC only) ──────────────────────────────────────
DATA_ROOT=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250
DS_DIR=HDEPIC_P01

# v7: use new captions with detailed descriptions
CAPTIONS=$DATA_ROOT/vlm-captions-w-desc/captions.json
GS_CKPT=$DATA_ROOT/gs-output/chkpnt45000.pth
WS=$WS_BASE/v7_detailed_desc/$DS_DIR

# v7-specific SAM2 masks (bboxes differ from shared masks)
MASKS=$WS_BASE/v7_detailed_desc/HDEPIC_masks

# GT masks reused — moved_050 is identical between old/new captions
NOVEL_MASKS=$WS_BASE/v2_sam2_shared/HDEPIC_novel_masks

# ── v7 hyperparameters (same as v5 CLIP CB-64, except text_field) ──────
TEXT_WEIGHT=0.5
TEXT_FIELD=description
CODEBOOK_SIZE=64
TOPK=4
ITERATIONS=10000
RESOLUTION=2

# ── Logging ─────────────────────────────────────────────────────────────
LOG_DIR=$WS_BASE/v7_logs
mkdir -p $LOG_DIR

echo "$(date) =========================================="
echo "  Ego3DVQA Pipeline v7: Detailed Text Descriptions"
echo "  Dataset:    $DS_DIR"
echo "  GPU:        $GPU"
echo "  Captions:   vlm-captions-w-desc"
echo "  Text field: $TEXT_FIELD (detailed descriptions for render branch)"
echo "  Config:     tw=$TEXT_WEIGHT, CB-$CODEBOOK_SIZE, top-$TOPK, ${ITERATIONS} iters"
echo "  Workspace:  $WS"
echo "=========================================="

# =========================================================================
echo "$(date) === STAGE 0: Generate SAM2 Masks ==="
# =========================================================================
if [ "$SKIP_MASKS" = true ]; then
  echo "  Skipping (--skip_masks). Using existing masks at $MASKS"
else
  cd $LANGSPLAT
  CUDA_VISIBLE_DEVICES=$GPU $PY_DA3 generate_sam2_masks.py \
    --captions_json $CAPTIONS \
    --data_root $DATA_ROOT \
    --output_dir $MASKS \
    --device cuda \
    2>&1 | tee $LOG_DIR/stage0_masks.log
  echo "$(date) SAM2 mask generation complete"
fi

# =========================================================================
echo "$(date) === STAGE 1: Prepare Workspace ==="
# =========================================================================
cd $LANGSPLAT
$PY prepare_ego3dvqa_workspace_v2.py \
  --data_root $DATA_ROOT --workspace $WS --captions_json $CAPTIONS \
  2>&1 | tee $LOG_DIR/stage1.log

# =========================================================================
echo "$(date) === STAGE 2: CLIP Feature Extraction (LERP tw=$TEXT_WEIGHT, field=$TEXT_FIELD) ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=$GPU $PY preprocess_sam2_ego3dvqa.py \
  --segments_json $MASKS/segments.json \
  --images_dir $WS/images \
  --output_dir $WS/language_features \
  --viz_dir $WS/diagnostics \
  --text_weight $TEXT_WEIGHT \
  --text_field $TEXT_FIELD \
  --batch_size 16 \
  2>&1 | tee $LOG_DIR/stage2.log
echo "$(date) Feature extraction complete"

# =========================================================================
echo "$(date) === STAGE 3: Post-process Seg Maps ==="
# =========================================================================
$PY postprocess_segmaps.py --workspace $WS \
  2>&1 | tee $LOG_DIR/stage3.log
echo "$(date) Post-processing complete"

# =========================================================================
echo "$(date) === STAGE 4: (skipped — codebook replaces autoencoder) ==="
# =========================================================================

# =========================================================================
echo "$(date) === STAGE 5: Codebook Training (LangSplatV2) ==="
# =========================================================================
cd $LANGSPLATV2
PORT=$((55640 + RANDOM % 100))
CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
  -s $WS -m $WS/output \
  --start_checkpoint $GS_CKPT \
  --include_feature --feature_level 1 \
  --resolution $RESOLUTION --test_iterations -1 \
  --iterations $ITERATIONS \
  --l1_loss --normalize --topk $TOPK \
  --codebook_size $CODEBOOK_SIZE --vq_layer_num 1 \
  --port $PORT \
  2>&1 | tee $LOG_DIR/stage5.log
echo "$(date) Codebook training complete"

# =========================================================================
echo "$(date) === STAGE 6: Novel-View Evaluation ==="
# =========================================================================
cd $LANGSPLAT
CUDA_VISIBLE_DEVICES=$GPU $PY eval_novel_views_codebook.py \
  --workspace $WS \
  --langsplatv2_dir $LANGSPLATV2 \
  --metadata_json $DATA_ROOT/vlm-data/moved_050/metadata.json \
  --captions_json $CAPTIONS \
  --gt_masks_dir $NOVEL_MASKS \
  --moved_rgb_dir $DATA_ROOT/vlm-data/moved_050/rgb \
  --num_vis_frames 10 \
  --encoder_type clip \
  --embed_dim 512 \
  2>&1 | tee $LOG_DIR/stage6.log
echo "$(date) Novel-view evaluation complete"

# =========================================================================
echo ""
echo "$(date) =========================================="
echo "  V7 PIPELINE COMPLETE"
echo "  Workspace:    $WS"
echo "  Eval results: $WS/eval_results/"
echo "  Logs:         $LOG_DIR/"
echo "=========================================="
