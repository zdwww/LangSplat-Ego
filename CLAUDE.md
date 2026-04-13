# CLAUDE.md

## Project

Fork of LangSplat for the Ego3DVQA benchmark. The main pipeline runs v5 CLIP CB-64 (best AP=0.3818) via `run_ego3dvqa_pipeline.sh`.

## Architecture

- **Feature extraction:** OpenCLIP ViT-B-16 encodes SAM2-segmented image crops, blended with text features (LERP tw=0.5)
- **Compression:** LangSplatV2 codebook (64 entries, top-4) replaces the original autoencoder
- **Training:** 3DGS with codebook feature head, 10K iterations, half-resolution
- **Evaluation:** Novel-view segmentation on moved_050 frames (IoU, AP, ROC-AUC)

## Key files

- `run_ego3dvqa_pipeline.sh` -- Main entry point (6 stages)
- `prepare_ego3dvqa_workspace_v2.py` -- Stage 1: workspace setup
- `preprocess_sam2_ego3dvqa.py` -- Stage 2: CLIP features + text blending
- `postprocess_segmaps.py` -- Stage 3: mask post-processing
- `eval_novel_views_codebook.py` -- Stage 6: novel-view evaluation
- `generate_sam2_masks.py` -- Prerequisite: SAM2 mask generation (da3 env)
- `generate_novel_gt_masks.py` -- Prerequisite: GT mask generation (da3 env)

Stage 5 training runs LangSplatV2's `train.py` at `/home/daiwei/LangSplat-variants/LangSplatV2/`.

## Environments

- `langsplat_v2` -- Main pipeline (Python 3.11, PyTorch, OpenCLIP)
- `da3` -- SAM2 mask generation (Python 3.x, PyTorch 2.5.1, SAM2)

## Data paths

- Dataset: `/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250/`
- Workspace: `/mnt/raptor/daiwei/LangSplat-workspace/v5_clip_codebook/HDEPIC_P01/`
- Shared masks: `/mnt/raptor/daiwei/LangSplat-workspace/v2_sam2_shared/`
- LangSplatV2: `/home/daiwei/LangSplat-variants/LangSplatV2/`

## Rasterizer

The CUDA rasterizer `NUM_CHANNELS_language_feature` in `config.h` MUST match the codebook size (64 for main pipeline). If it was changed (e.g., for a 128-codebook experiment), use `rebuild_rasterizer_64ch.sh` and `pip install .` from the rasterizer submodule directory. The .so loaded at runtime comes from site-packages, not from `build_ext --inplace`.

## Common tasks

```bash
# Run full pipeline
bash run_ego3dvqa_pipeline.sh --gpu 6

# Generate SAM2 masks for a new dataset (requires da3 env)
/home/daiwei/miniconda3/envs/da3/bin/python generate_sam2_masks.py \
  --captions_json <captions.json> --data_root <dataset_root> --output_dir <output>

# Check rasterizer channel config
bash rebuild_rasterizer_64ch.sh status
```

## Archive

`archive/` contains scripts from previous experiments (v1-v7). These are kept for reference but are not used by the main pipeline. See `docs/experiments/04-11_unified-v2-v6-analysis.md` for the comprehensive experiment comparison.
