#!/bin/bash
# One-shot: rebuild efficient-langsplat-rasterization for NUM_CHANNELS=64 so
# v5 codebook evals can run (v5 trained with codebook_size=64; current build
# uses 128 for v6 which gives v5 a dimension mismatch). Backs up the 128-ch
# .so first and restores it after the run. Run this from the LangSplat dir.
#
# Usage: bash rebuild_rasterizer_64ch.sh [rebuild_to_64|restore_to_128|status]
set -e
set -u

MODE=${1:-status}
RAST_DIR=/home/daiwei/LangSplat-variants/LangSplatV2/submodules/efficient-langsplat-rasterization
CONFIG_H=$RAST_DIR/cuda_rasterizer/config.h
BUILD_SO=$RAST_DIR/build/lib.linux-x86_64-cpython-39/diff_gaussian_rasterization/_C.cpython-39-x86_64-linux-gnu.so
INPLACE_SO=$RAST_DIR/diff_gaussian_rasterization/_C.cpython-39-x86_64-linux-gnu.so
BACKUP_128=$RAST_DIR/build/lib.linux-x86_64-cpython-39/diff_gaussian_rasterization/_C.128.so.bak
BACKUP_64=$RAST_DIR/build/lib.linux-x86_64-cpython-39/diff_gaussian_rasterization/_C.64.so.bak
PY=/home/daiwei/miniconda3/envs/langsplat_v2/bin/python

status() {
  echo "config.h:"
  grep 'NUM_CHANNELS_language_feature' $CONFIG_H
  echo ""
  echo "Built .so sizes / timestamps:"
  ls -la $BUILD_SO $INPLACE_SO 2>&1 | grep -v "^total"
  echo ""
  echo "Backups:"
  ls -la $BACKUP_128 $BACKUP_64 2>&1 | grep -v "cannot access" | grep -v "^total"
}

rebuild_to_64() {
  # Back up 128-channel .so if the backup is missing (should already exist)
  if [ ! -f $BACKUP_128 ]; then
    echo "Creating 128-channel backup..."
    cp $BUILD_SO $BACKUP_128
  fi
  echo "Editing config.h: 128 -> 64..."
  sed -i 's|#define NUM_CHANNELS_language_feature 128|#define NUM_CHANNELS_language_feature 64|' $CONFIG_H
  grep 'NUM_CHANNELS_language_feature' $CONFIG_H

  echo "Rebuilding (this may take a few minutes)..."
  cd $RAST_DIR
  $PY setup.py build_ext --inplace 2>&1 | tail -5

  echo "Verifying new .so..."
  ls -la $BUILD_SO

  # Save a 64-channel backup
  cp $BUILD_SO $BACKUP_64
  echo "64-channel .so saved to $BACKUP_64"
}

restore_to_128() {
  if [ ! -f $BACKUP_128 ]; then
    echo "ERROR: $BACKUP_128 not found — cannot restore"
    exit 1
  fi
  echo "Restoring 128-channel .so from backup..."
  cp $BACKUP_128 $BUILD_SO
  cp $BACKUP_128 $INPLACE_SO
  echo "Editing config.h: 64 -> 128..."
  sed -i 's|#define NUM_CHANNELS_language_feature 64|#define NUM_CHANNELS_language_feature 128|' $CONFIG_H
  grep 'NUM_CHANNELS_language_feature' $CONFIG_H
  echo "Restore complete."
}

case "$MODE" in
  status)          status ;;
  rebuild_to_64)   rebuild_to_64 ;;
  restore_to_128)  restore_to_128 ;;
  *) echo "Unknown MODE: $MODE"; exit 2 ;;
esac
