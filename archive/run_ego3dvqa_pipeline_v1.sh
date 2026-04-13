#!/bin/bash
# Master pipeline: LangSplat on Ego3DVQA datasets
# Runs everything from frame selection through evaluation.
# Usage: bash run_ego3dvqa_pipeline.sh
set -e

LANGSPLAT=/home/daiwei/Ego3DVQA-GS/LangSplat
WS=/mnt/raptor/daiwei/LangSplat-workspace
PFX=/home/daiwei/miniconda3/envs/langsplat_v2
export LD_LIBRARY_PATH=$PFX/lib:$LD_LIBRARY_PATH
PY=$PFX/bin/python

ADT_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/ADT/Apartment_release_clean_seq131_M1292
HDEPIC_DATA=/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250
ADT_WS=$WS/ADT_seq131
HDEPIC_WS=$WS/HDEPIC_P01
ADT_CKPT=$ADT_DATA/gs-output/chkpnt45000.pth
HDEPIC_CKPT=$HDEPIC_DATA/gs-output/chkpnt45000.pth
SAM=$WS/weights/sam_vit_h_4b8939.pth

mkdir -p $WS/logs $ADT_WS $HDEPIC_WS
cd $LANGSPLAT

# =========================================================================
echo "$(date) === STEP 1: Frame Selection ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=0 $PY select_frames.py --data_root $ADT_DATA \
  --output $ADT_WS/selected_frames.json \
  --max_frames 200 --diversity_threshold 0.5 \
  2>&1 | tee $WS/logs/step1_ADT.log &
PID1=$!

CUDA_VISIBLE_DEVICES=4 $PY select_frames.py --data_root $HDEPIC_DATA \
  --output $HDEPIC_WS/selected_frames.json \
  --max_frames 200 --diversity_threshold 0.5 \
  2>&1 | tee $WS/logs/step1_HDEPIC.log &
PID2=$!
wait $PID1 $PID2

# =========================================================================
echo "$(date) === STEP 2: Prepare Workspaces ==="
# =========================================================================
$PY prepare_ego3dvqa_workspace.py --data_root $ADT_DATA --workspace $ADT_WS \
  --selected_frames $ADT_WS/selected_frames.json \
  2>&1 | tee $WS/logs/step2_ADT.log

$PY prepare_ego3dvqa_workspace.py --data_root $HDEPIC_DATA --workspace $HDEPIC_WS \
  --selected_frames $HDEPIC_WS/selected_frames.json \
  2>&1 | tee $WS/logs/step2_HDEPIC.log

# =========================================================================
echo "$(date) === STEP 3a: SAM+CLIP Preprocessing ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=0 $PY preprocess.py --dataset_path $ADT_WS --sam_ckpt_path $SAM \
  2>&1 | tee $WS/logs/step3a_ADT.log &
PID1=$!

CUDA_VISIBLE_DEVICES=4 $PY preprocess.py --dataset_path $HDEPIC_WS --sam_ckpt_path $SAM \
  2>&1 | tee $WS/logs/step3a_HDEPIC.log &
PID2=$!
wait $PID1 $PID2

echo "$(date) === STEP 3b: Post-process Seg Maps ==="
$PY postprocess_segmaps.py --workspace $ADT_WS 2>&1 | tee $WS/logs/step3b_ADT.log
$PY postprocess_segmaps.py --workspace $HDEPIC_WS 2>&1 | tee $WS/logs/step3b_HDEPIC.log

echo "$(date) === STEP 3c: Visualize ==="
$PY visualize_preprocessing.py --workspace $ADT_WS 2>&1 | tee $WS/logs/step3c_ADT.log
$PY visualize_preprocessing.py --workspace $HDEPIC_WS 2>&1 | tee $WS/logs/step3c_HDEPIC.log

# =========================================================================
echo "$(date) === STEP 4: Autoencoder ==="
# =========================================================================
cd $LANGSPLAT/autoencoder
CUDA_VISIBLE_DEVICES=0 $PY train.py --dataset_path $ADT_WS --dataset_name ADT_seq131 --num_epochs 100 --lr 0.0007 \
  2>&1 | tee $WS/logs/step4_train_ADT.log &
PID1=$!
CUDA_VISIBLE_DEVICES=4 $PY train.py --dataset_path $HDEPIC_WS --dataset_name HDEPIC_P01 --num_epochs 100 --lr 0.0007 \
  2>&1 | tee $WS/logs/step4_train_HDEPIC.log &
PID2=$!
wait $PID1 $PID2

CUDA_VISIBLE_DEVICES=0 $PY test.py --dataset_path $ADT_WS --dataset_name ADT_seq131 \
  2>&1 | tee $WS/logs/step4_compress_ADT.log &
PID1=$!
CUDA_VISIBLE_DEVICES=4 $PY test.py --dataset_path $HDEPIC_WS --dataset_name HDEPIC_P01 \
  2>&1 | tee $WS/logs/step4_compress_HDEPIC.log &
PID2=$!
wait $PID1 $PID2

# =========================================================================
echo "$(date) === STEP 5: Language Feature Training ==="
# =========================================================================
cd $LANGSPLAT

# ADT L1 + HDEPIC L1 on GPUs 0,4 | ADT L2 + HDEPIC L2 on GPUs 6,7
CUDA_VISIBLE_DEVICES=0 $PY train.py -s $ADT_WS -m $ADT_WS/output \
  --start_checkpoint $ADT_CKPT --feature_level 1 --test_iterations -1 --port 55501 \
  2>&1 | tee $WS/logs/step5_ADT_L1.log &
PID_A1=$!
CUDA_VISIBLE_DEVICES=6 $PY train.py -s $ADT_WS -m $ADT_WS/output \
  --start_checkpoint $ADT_CKPT --feature_level 2 --test_iterations -1 --port 55502 \
  2>&1 | tee $WS/logs/step5_ADT_L2.log &
PID_A2=$!
CUDA_VISIBLE_DEVICES=4 $PY train.py -s $HDEPIC_WS -m $HDEPIC_WS/output \
  --start_checkpoint $HDEPIC_CKPT --feature_level 1 --test_iterations -1 --port 55511 \
  2>&1 | tee $WS/logs/step5_HDEPIC_L1.log &
PID_H1=$!
CUDA_VISIBLE_DEVICES=7 $PY train.py -s $HDEPIC_WS -m $HDEPIC_WS/output \
  --start_checkpoint $HDEPIC_CKPT --feature_level 2 --test_iterations -1 --port 55512 \
  2>&1 | tee $WS/logs/step5_HDEPIC_L2.log &
PID_H2=$!

# When L1 finishes, start L3 on the same GPU
wait $PID_A1
echo "$(date) ADT L1 done, starting L3..."
CUDA_VISIBLE_DEVICES=0 $PY train.py -s $ADT_WS -m $ADT_WS/output \
  --start_checkpoint $ADT_CKPT --feature_level 3 --test_iterations -1 --port 55503 \
  2>&1 | tee $WS/logs/step5_ADT_L3.log &
PID_A3=$!

wait $PID_H1
echo "$(date) HDEPIC L1 done, starting L3..."
CUDA_VISIBLE_DEVICES=4 $PY train.py -s $HDEPIC_WS -m $HDEPIC_WS/output \
  --start_checkpoint $HDEPIC_CKPT --feature_level 3 --test_iterations -1 --port 55513 \
  2>&1 | tee $WS/logs/step5_HDEPIC_L3.log &
PID_H3=$!

wait $PID_A2 $PID_A3 $PID_H2 $PID_H3
echo "$(date) All feature training complete."

# =========================================================================
echo "$(date) === STEP 6: Render ==="
# =========================================================================
for LV in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 $PY render.py -m $ADT_WS/output_${LV} --include_feature --skip_test \
    2>&1 | tee $WS/logs/step6_ADT_L${LV}.log
done &
PID1=$!
for LV in 1 2 3; do
  CUDA_VISIBLE_DEVICES=4 $PY render.py -m $HDEPIC_WS/output_${LV} --include_feature --skip_test \
    2>&1 | tee $WS/logs/step6_HDEPIC_L${LV}.log
done &
PID2=$!
wait $PID1 $PID2

# =========================================================================
echo "$(date) === STEP 7: Evaluation ==="
# =========================================================================
CUDA_VISIBLE_DEVICES=0 $PY eval_ego3dvqa.py --workspace $ADT_WS \
  --ae_ckpt $LANGSPLAT/autoencoder/ckpt/ADT_seq131/best_ckpt.pth \
  --prompts "kitchen counter" "refrigerator" "sofa" "bookshelf" "sink" "table" \
  --num_vis_frames 10 2>&1 | tee $WS/logs/step7_ADT.log &
PID1=$!

CUDA_VISIBLE_DEVICES=4 $PY eval_ego3dvqa.py --workspace $HDEPIC_WS \
  --ae_ckpt $LANGSPLAT/autoencoder/ckpt/HDEPIC_P01/best_ckpt.pth \
  --prompts "cutting board" "knife" "pan" "stove" "kettle" "fridge" \
  --num_vis_frames 10 2>&1 | tee $WS/logs/step7_HDEPIC.log &
PID2=$!
wait $PID1 $PID2

echo "$(date) === PIPELINE COMPLETE ==="
echo "Results at: $WS/"
echo "  ADT diagnostics: $ADT_WS/diagnostics/"
echo "  ADT eval: $ADT_WS/eval_results/"
echo "  HDEPIC diagnostics: $HDEPIC_WS/diagnostics/"
echo "  HDEPIC eval: $HDEPIC_WS/eval_results/"
echo "  Logs: $WS/logs/"
