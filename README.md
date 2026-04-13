# LangSplat for Ego3DVQA

Fork of [LangSplat](https://github.com/minghanqin/LangSplat) (CVPR 2024) adapted for egocentric 3D visual question answering on the **Ego3DVQA** benchmark. Uses [LangSplatV2](https://langsplat-v2.github.io/) codebook-based feature compression instead of the original autoencoder.

**Best configuration (v5 CLIP CB-64):** CLIP LERP@0.5 text blending + 64-entry codebook, achieving AP=0.3818 on HD-EPIC novel-view segmentation.

## Pipeline Overview

The main pipeline (`run_ego3dvqa_pipeline.sh`) runs 6 stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `prepare_ego3dvqa_workspace_v2.py` | Symlinks images, COLMAP data, and masks into a workspace directory |
| 2 | `preprocess_sam2_ego3dvqa.py` | CLIP-encodes SAM2-segmented crops, blends image+text features (LERP tw=0.5) |
| 3 | `postprocess_segmaps.py` | Masks out dynamic objects and hands from segmentation maps |
| 4 | *(skipped)* | Autoencoder replaced by codebook |
| 5 | LangSplatV2 `train.py` | Trains 3D Gaussian splatting with codebook feature compression (CB-64, top-4) |
| 6 | `eval_novel_views_codebook.py` | Evaluates on novel views: IoU, AP, ROC-AUC, FG-BG saliency gap |

### Prerequisites (one-time setup)

These scripts prepare the data that the pipeline consumes. They are NOT called by the pipeline itself.

| Script | Conda Env | Description |
|--------|-----------|-------------|
| `generate_sam2_masks.py` | `da3` | Generates SAM2 masks from VLM caption bounding boxes |
| `generate_novel_gt_masks.py` | `da3` | Generates ground-truth SAM2 masks for novel-view evaluation |

## Quick Start

```bash
# Run the full pipeline on HD-EPIC (default)
bash run_ego3dvqa_pipeline.sh

# Specify GPU
bash run_ego3dvqa_pipeline.sh --gpu 4

# Run on ADT dataset
bash run_ego3dvqa_pipeline.sh --dataset adt
```

## Directory Structure

```
.
├── run_ego3dvqa_pipeline.sh       # Main pipeline entry point
├── prepare_ego3dvqa_workspace_v2.py  # Stage 1: workspace setup
├── preprocess_sam2_ego3dvqa.py       # Stage 2: CLIP feature extraction
├── postprocess_segmaps.py            # Stage 3: segmap post-processing
├── eval_novel_views_codebook.py      # Stage 6: novel-view evaluation
├── generate_sam2_masks.py            # Prerequisite: SAM2 mask generation
├── generate_novel_gt_masks.py        # Prerequisite: GT mask generation
├── rebuild_rasterizer_64ch.sh        # Tool: switch rasterizer between 64/128 channels
│
├── gaussian_renderer/     # LangSplat rendering (used by old eval; codebook eval uses LangSplatV2)
├── scene/                 # Camera, dataset readers, Gaussian model
├── arguments/             # Argument parsing
├── utils/                 # Utility functions
├── autoencoder/           # Original LangSplat autoencoder (replaced by codebook)
├── eval/                  # Original LangSplat evaluation
├── lpipsPyTorch/          # LPIPS perceptual loss
├── submodules/            # Git submodules (rasterization, SAM, simple-knn)
├── docs/                  # Experiment reports and analysis
├── archive/               # Previous experiment scripts (v1-v7, reports, analysis)
└── assets/                # Images
```

## Input Data

The pipeline expects data organized under a dataset root. Example for HD-EPIC:

```
/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/P01-20240202-110250/
├── images/                        # Full-resolution RGB images (camera-rgb_*.jpg)
├── sparse/0/                      # COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)
├── gs-output/chkpnt45000.pth      # Pre-trained RGB Gaussian splatting checkpoint
├── vlm-captions/captions.json     # VLM-generated captions with bounding boxes
└── vlm-data/moved_050/            # Novel-view evaluation data
    ├── metadata.json              #   Frame metadata and object annotations
    └── rgb/                       #   Novel-view RGB images
```

### Pre-computed shared data

SAM2 masks and GT masks are pre-computed and shared across experiments:

```
/mnt/raptor/daiwei/LangSplat-workspace/v2_sam2_shared/
├── HDEPIC_masks/           # SAM2 masks for training frames
│   ├── masks/              #   Binary mask PNGs per object
│   └── segments.json       #   Metadata (categories, bboxes, IoU scores)
├── HDEPIC_novel_masks/     # GT masks for novel-view evaluation
├── ADT_masks/              # SAM2 masks for ADT dataset
└── ADT_novel_masks/        # GT masks for ADT novel views
```

## Output

The pipeline writes all outputs to a workspace directory:

```
/mnt/raptor/daiwei/LangSplat-workspace/v5_clip_codebook/HDEPIC_P01/
├── images/                        # Symlinked training images
├── sparse/                        # Symlinked COLMAP data
├── language_features/             # CLIP features (_f.npy) and segmaps (_s.npy) per frame
├── diagnostics/                   # CLIP input montage visualizations
├── output/                        # Trained model
│   └── output_1/
│       ├── point_cloud/           # Final Gaussian point cloud
│       ├── chkpnt10000.pth        # Training checkpoint
│       └── cfg_args               # Training configuration
└── eval_results/                  # Evaluation output
    ├── results.json               # Per-object and aggregate metrics (IoU, AP, ROC-AUC)
    └── vis/                       # Side-by-side visualizations
```

Pipeline logs are written to `/mnt/raptor/daiwei/LangSplat-workspace/pipeline_logs/`.

## Configuration

Key hyperparameters (set in `run_ego3dvqa_pipeline.sh`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TEXT_WEIGHT` | 0.5 | LERP blend: `feat = normalize(0.5*image + 0.5*text)` |
| `CODEBOOK_SIZE` | 64 | Number of codebook entries in LangSplatV2 |
| `TOPK` | 4 | Top-k codebook entries per Gaussian |
| `ITERATIONS` | 10000 | Training iterations |
| `RESOLUTION` | 2 | Half-resolution training (full-res OOMs on 24GB GPUs) |

## Environment

The pipeline uses the `langsplat_v2` conda environment:

```
conda env: langsplat_v2
python:    /home/daiwei/miniconda3/envs/langsplat_v2/bin/python
```

Prerequisite mask generation (SAM2) uses the `da3` environment:

```
conda env: da3
python:    /home/daiwei/miniconda3/envs/da3/bin/python
```

### Rasterizer

The pipeline depends on `diff_gaussian_rasterization` from LangSplatV2's custom CUDA rasterizer at:

```
/home/daiwei/LangSplat-variants/LangSplatV2/submodules/efficient-langsplat-rasterization/
```

The rasterizer's `NUM_CHANNELS_language_feature` in `config.h` must match the codebook size (64 for the main pipeline). Use the maintenance tool to check or switch:

```bash
bash rebuild_rasterizer_64ch.sh status         # Check current config
bash rebuild_rasterizer_64ch.sh rebuild_to_64  # Switch to 64 channels
bash rebuild_rasterizer_64ch.sh restore_to_128 # Switch to 128 channels
```

After switching, reinstall to site-packages:
```bash
cd /home/daiwei/LangSplat-variants/LangSplatV2/submodules/efficient-langsplat-rasterization/
pip install .
```

## Experiment History

Previous experiments are archived in `archive/`. The v2-v6 unified analysis is documented in `docs/experiments/04-11_unified-v2-v6-analysis.md`.

| Version | Architecture | AP | Notes |
|---------|-------------|-----|-------|
| v1 | SAM1 + autoencoder (3 feature levels) | 0.0975 | Original LangSplat pipeline |
| v2 | SAM2 + CLIP image-only | 0.2684 | Baseline with SAM2 masks |
| v2-lerp | SAM2 + CLIP LERP@0.5 + autoencoder | 0.3302 | Text blending helps |
| v3 | SLERP + adaptive text weights | 0.2779 | Over-engineering hurt |
| v4 | Qwen3-VL embeddings | 0.0524 | Qwen3-VL embeddings incompatible |
| **v5** | **CLIP LERP@0.5 + CB-64** | **0.3818** | **Best -- main pipeline** |
| v6 | CLIP LERP@0.5 + CB-128 | 0.3680 | Larger codebook slightly worse |
| v7 | CLIP detailed descriptions + CB-64 | 0.3437 | Richer text hurt (train-test asymmetry) |

## References

- [LangSplat](https://github.com/minghanqin/LangSplat) (CVPR 2024) -- Qin et al.
- [LangSplatV2](https://langsplat-v2.github.io/) (NeurIPS 2025) -- Li et al.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) -- Kerbl et al.
