#!/bin/bash
# Unified re-evaluation of v2-v6 novel-view segmentation with the corrected
# eval scripts:
#   - per-object mean IoU (new metrics.mean_iou_object)
#   - threshold-free AP + ROC-AUC + FG/BG saliency
#   - eval_novel_views_codebook.py loads highest checkpoint (fixes v6 10K->30K)
#
# Two-pass execution because v5 (codebook=64) and v6 (codebook=128) require
# different NUM_CHANNELS_language_feature builds of the efficient-langsplat
# rasterizer:
#   MODE=pass1 -> v2 + v3 + v4 + v6    (uses current 128-channel build)
#   MODE=pass2 -> v5 only              (requires 64-channel rebuild first)
#   MODE=all   -> pass1 then pass2      (assumes 64-ch rebuild was done manually)
#
# Backs up each existing eval_novel_summary.json to eval_novel_summary.legacy.json
# before the rerun. Logs to reeval_v2_v6_logs/.
#
# Usage:
#   GPU_A=6 GPU_B=7 bash reeval_v2_v6_unified.sh [pass1|pass2|all]
set -u
MODE=${1:-${MODE:-pass1}}

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
WS_BASE=/mnt/raptor/daiwei/LangSplat-workspace
SHARED=$WS_BASE/v2_sam2_shared
LANGSPLATV2=/home/daiwei/LangSplat-variants/LangSplatV2

PFX_LANG=/home/daiwei/miniconda3/envs/langsplat_v2
PFX_QWEN=/home/daiwei/miniconda3/envs/qwen3vl
export LD_LIBRARY_PATH=$PFX_LANG/lib:${LD_LIBRARY_PATH:-}
PY_LANG=$PFX_LANG/bin/python
PY_QWEN=$PFX_QWEN/bin/python

ADT_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/ADT/Apartment_release_clean_seq131_M1292
HDEPIC_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250

ADT_META=$ADT_DATA/vlm-data/moved_050/metadata.json
ADT_CAPS=$ADT_DATA/vlm-captions/captions.json
ADT_RGB=$ADT_DATA/vlm-data/moved_050/rgb
ADT_GT=$SHARED/ADT_novel_masks

HDEPIC_META=$HDEPIC_DATA/vlm-data/moved_050/metadata.json
HDEPIC_CAPS=$HDEPIC_DATA/vlm-captions/captions.json
HDEPIC_RGB=$HDEPIC_DATA/vlm-data/moved_050/rgb
HDEPIC_GT=$SHARED/HDEPIC_novel_masks

# Precomputed text embeddings — required because langsplat_v2 env now has the
# efficient-langsplat-rasterization installed (which breaks the AE-decode path),
# so v2/v3/v4 must run in the qwen3vl env (which has the original
# langsplat-rasterization). qwen3vl env lacks open_clip, so CLIP text queries
# are precomputed into .npz files by precompute_clip_text_embeds.py.
QWEN_EMB=$WS_BASE/v6_logs/qwen3vl_text_embeds.npz
CLIP_HDEPIC_EMB=$WS_BASE/reeval_v2_v6_logs/clip_hdepic_text_embeds.npz
CLIP_ADT_EMB=$WS_BASE/reeval_v2_v6_logs/clip_adt_text_embeds.npz

GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}

LOG_DIR=$WS_BASE/reeval_v2_v6_logs
mkdir -p $LOG_DIR

backup() {
  local ws=$1
  local f=$ws/eval_novel_results/eval_novel_summary.json
  if [ -f "$f" ] && [ ! -f "$ws/eval_novel_results/eval_novel_summary.legacy.json" ]; then
    cp "$f" "$ws/eval_novel_results/eval_novel_summary.legacy.json"
    echo "  backed up $f -> .legacy.json"
  fi
}

# =====================================================================
# v2 + v3 (AE decode) — runs in qwen3vl env (has the original
# langsplat-rasterization; langsplat_v2's diff_gaussian_rasterization was
# overwritten by efficient-langsplat-rasterization and no longer works for
# 3-channel feature rendering). Uses precomputed CLIP text embeds because
# qwen3vl env lacks open_clip.
# 6 v2 runs (3 tw × 2 datasets) + 4 v3 runs (2 max × 2 datasets) = 10
# =====================================================================
run_v2_v3() {
  local ws=$1 ae_name=$2 dataset=$3 gpu=$4 label=$5
  local wsdir=$WS_BASE/$ws/$dataset
  local ae_ckpt=$LANGSPLAT/autoencoder/ckpt/$ae_name/best_ckpt.pth
  local meta caps rgb gt clip_emb
  if [ "$dataset" = "ADT_seq131" ]; then
    meta=$ADT_META caps=$ADT_CAPS rgb=$ADT_RGB gt=$ADT_GT clip_emb=$CLIP_ADT_EMB
  else
    meta=$HDEPIC_META caps=$HDEPIC_CAPS rgb=$HDEPIC_RGB gt=$HDEPIC_GT clip_emb=$CLIP_HDEPIC_EMB
  fi
  backup $wsdir
  echo "$(date) [$label GPU=$gpu] $wsdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY_QWEN $LANGSPLAT/eval_novel_views.py \
    --workspace $wsdir --ae_ckpt $ae_ckpt \
    --metadata_json $meta --captions_json $caps \
    --gt_masks_dir $gt --moved_rgb_dir $rgb \
    --num_vis_frames 0 \
    --encoder_type precomputed --precomputed_text_embeds $clip_emb \
    --embed_dim 512 \
    2>&1 | tee $LOG_DIR/reeval_${label}.log
}

# =====================================================================
# v4 Qwen3-VL (AE decode, qwen3vl env, eval_novel_views.py --encoder_type qwen3vl)
# 3 runs (image_only, multimodal, lerp_tw0.5) × HDEPIC
# =====================================================================
run_v4() {
  local ws=$1 ae_name=$2 gpu=$3 label=$4
  local wsdir=$WS_BASE/$ws/HDEPIC_P01
  local ae_ckpt=$LANGSPLAT/autoencoder/ckpt/$ae_name/best_ckpt.pth
  backup $wsdir
  echo "$(date) [$label GPU=$gpu] $wsdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY_QWEN $LANGSPLAT/eval_novel_views.py \
    --workspace $wsdir --ae_ckpt $ae_ckpt \
    --metadata_json $HDEPIC_META --captions_json $HDEPIC_CAPS \
    --gt_masks_dir $HDEPIC_GT --moved_rgb_dir $HDEPIC_RGB \
    --num_vis_frames 0 \
    --encoder_type qwen3vl --embed_dim 512 \
    2>&1 | tee $LOG_DIR/reeval_${label}.log
}

# =====================================================================
# v5 Codebook (2 runs: CLIP + Qwen)
# =====================================================================
run_v5_clip() {
  local gpu=$1
  local wsdir=$WS_BASE/v5_clip_codebook/HDEPIC_P01
  backup $wsdir
  echo "$(date) [v5_clip GPU=$gpu] $wsdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY_LANG $LANGSPLAT/eval_novel_views_codebook.py \
    --workspace $wsdir --langsplatv2_dir $LANGSPLATV2 \
    --metadata_json $HDEPIC_META --captions_json $HDEPIC_CAPS \
    --gt_masks_dir $HDEPIC_GT --moved_rgb_dir $HDEPIC_RGB \
    --num_vis_frames 0 \
    --encoder_type clip --embed_dim 512 --topk 4 \
    2>&1 | tee $LOG_DIR/reeval_v5_clip.log
}

run_v5_qwen() {
  local gpu=$1
  local wsdir=$WS_BASE/v5_qwen_codebook/HDEPIC_P01
  backup $wsdir
  echo "$(date) [v5_qwen GPU=$gpu] $wsdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY_LANG $LANGSPLAT/eval_novel_views_codebook.py \
    --workspace $wsdir --langsplatv2_dir $LANGSPLATV2 \
    --metadata_json $HDEPIC_META --captions_json $HDEPIC_CAPS \
    --gt_masks_dir $HDEPIC_GT --moved_rgb_dir $HDEPIC_RGB \
    --num_vis_frames 0 \
    --encoder_type precomputed --precomputed_text_embeds $QWEN_EMB \
    --embed_dim 512 --topk 4 \
    2>&1 | tee $LOG_DIR/reeval_v5_qwen.log
}

# =====================================================================
# v6 Codebook (2 runs: CLIP + Qwen). Checkpoint auto-selected to 30K
# by the fixed eval_novel_views_codebook.py.
# =====================================================================
run_v6_clip() {
  local gpu=$1
  local wsdir=$WS_BASE/v6_clip_codebook/HDEPIC_P01
  backup $wsdir
  echo "$(date) [v6_clip GPU=$gpu] $wsdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY_LANG $LANGSPLAT/eval_novel_views_codebook.py \
    --workspace $wsdir --langsplatv2_dir $LANGSPLATV2 \
    --metadata_json $HDEPIC_META --captions_json $HDEPIC_CAPS \
    --gt_masks_dir $HDEPIC_GT --moved_rgb_dir $HDEPIC_RGB \
    --num_vis_frames 0 \
    --encoder_type clip --embed_dim 512 --topk 8 \
    2>&1 | tee $LOG_DIR/reeval_v6_clip.log
}

run_v6_qwen() {
  local gpu=$1
  local wsdir=$WS_BASE/v6_qwen_codebook/HDEPIC_P01
  backup $wsdir
  echo "$(date) [v6_qwen GPU=$gpu] $wsdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY_LANG $LANGSPLAT/eval_novel_views_codebook.py \
    --workspace $wsdir --langsplatv2_dir $LANGSPLATV2 \
    --metadata_json $HDEPIC_META --captions_json $HDEPIC_CAPS \
    --gt_masks_dir $HDEPIC_GT --moved_rgb_dir $HDEPIC_RGB \
    --num_vis_frames 0 \
    --encoder_type precomputed --precomputed_text_embeds $QWEN_EMB \
    --embed_dim 512 --topk 8 \
    2>&1 | tee $LOG_DIR/reeval_v6_qwen.log
}

echo "$(date) ============================================"
echo "  Unified v2-v6 re-evaluation  MODE=$MODE"
echo "  GPU_A=$GPU_A  GPU_B=$GPU_B"
echo "  Logs: $LOG_DIR/"
echo "============================================"

run_pass1() {
  # ---------------- v2 (6 runs) — pair by dataset, 2 GPUs ----------------
  run_v2_v3 v2_sam2_tw0.0 ADT_seq131_tw0.0  ADT_seq131  $GPU_A v2_tw0.0_ADT &
  run_v2_v3 v2_sam2_tw0.0 HDEPIC_P01_tw0.0  HDEPIC_P01  $GPU_B v2_tw0.0_HDEPIC &
  wait

  run_v2_v3 v2_sam2_tw0.5 ADT_seq131_tw0.5  ADT_seq131  $GPU_A v2_tw0.5_ADT &
  run_v2_v3 v2_sam2_tw0.5 HDEPIC_P01_tw0.5  HDEPIC_P01  $GPU_B v2_tw0.5_HDEPIC &
  wait

  run_v2_v3 v2_sam2_tw1.0 ADT_seq131_tw1.0  ADT_seq131  $GPU_A v2_tw1.0_ADT &
  run_v2_v3 v2_sam2_tw1.0 HDEPIC_P01_tw1.0  HDEPIC_P01  $GPU_B v2_tw1.0_HDEPIC &
  wait

  # ---------------- v3 (4 runs) ----------------
  run_v2_v3 v3_slerp_adaptive_max0.5 ADT_seq131_v3_max0.5 ADT_seq131 $GPU_A v3_max0.5_ADT &
  run_v2_v3 v3_slerp_adaptive_max0.5 HDEPIC_P01_v3_max0.5 HDEPIC_P01 $GPU_B v3_max0.5_HDEPIC &
  wait

  run_v2_v3 v3_slerp_adaptive_max1.0 ADT_seq131_v3_max1.0 ADT_seq131 $GPU_A v3_max1.0_ADT &
  run_v2_v3 v3_slerp_adaptive_max1.0 HDEPIC_P01_v3_max1.0 HDEPIC_P01 $GPU_B v3_max1.0_HDEPIC &
  wait

  # ---------------- v4 Qwen (3 runs, qwen3vl env) ----------------
  run_v4 v4_qwen3vl_image_only HDEPIC_P01_v4_image_only $GPU_A v4_image_only &
  run_v4 v4_qwen3vl_multimodal HDEPIC_P01_v4_multimodal $GPU_B v4_multimodal &
  wait

  run_v4 v4_qwen3vl_lerp_tw0.5 HDEPIC_P01_v4_lerp_tw0.5 $GPU_A v4_lerp_tw0.5

  # ---------------- v6 Codebook (2 runs, needs 128-channel rasterizer) ----
  run_v6_clip $GPU_A &
  run_v6_qwen $GPU_B &
  wait
}

run_pass2() {
  # ---------------- v5 Codebook (2 runs, needs 64-channel rasterizer) ----
  run_v5_clip $GPU_A &
  run_v5_qwen $GPU_B &
  wait
}

case "$MODE" in
  pass1) run_pass1 ;;
  pass2) run_pass2 ;;
  all)   run_pass1; run_pass2 ;;
  *)     echo "Unknown MODE: $MODE (expected pass1|pass2|all)"; exit 2 ;;
esac

echo "$(date) ============================================"
echo "  $MODE complete"
echo "  Summaries overwritten; legacy backups in .legacy.json"
echo "============================================"
