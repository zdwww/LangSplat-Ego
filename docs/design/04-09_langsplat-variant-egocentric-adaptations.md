# LangSplat Variant — Egocentric Adaptations

**Date**: 2026-04-09
**Source**: `/home/daiwei/LangSplat-variants/LangSplat/` (old archive workspace)
**Related**: [Quality Analysis](../experiments/04-09_langsplat-egocentric-quality-analysis.md), [Pipeline Report](../pipeline/04-09_ego3dvqa-langsplat-pipeline-report.md)

---

## Overview

The old LangSplat variant adapted the CVPR 2024 paper for in-the-wild egocentric video (HD-EPIC kitchen, ADT apartment). It has **3 modified files** and **10 new files** (7 code + 3 docs). The key design principle: keep all changes in preprocessing so the downstream pipeline (autoencoder, Gaussian training, rendering) works unmodified.

---

## 1. SAM2 Replaces SAM (`preprocess_sam2.py` — NEW, 392 lines)

The biggest adaptation. Original LangSplat runs SAM with a 32x32 point grid, producing 3 hierarchical segmentation levels per frame. This fails on egocentric video because:
- SAM's point-grid produces too many overlapping masks in cluttered indoor scenes
- All masks are anonymous — no semantic labels

**Replacement**: Pre-computed SAM2 segments from an upstream VLM captioning pipeline, which provides:
- **Category names** per segment (e.g., "wooden countertop", "stainless steel microwave") from `segments.json`
- **Single-level segmentation** replicated across all 4 channels (trains only `feature_level=1` instead of 1,2,3)
- **Static mask AND-filtering** (`--static_masks_dir`): filters out dynamic objects (hands, held objects) by AND-ing SAM2 masks with pre-computed static/dynamic masks (0=dynamic, 255=keep)
- **CW 90 rotation handling** (`--sam2_rotation cw90`): SAM2 was run on rotated images while 3DGS was trained on originals; the script rotates images/masks to match SAM2 coordinates during CLIP encoding, then rotates seg_map back for training

### Output Format

Same as original LangSplat — `_f.npy` [4xM, 512] and `_s.npy` [4, H, W] — so everything downstream works unchanged.

---

## 2. Text Feature Blending (`--text_weight`)

The key experimental innovation. Original LangSplat uses only CLIP image features from masked crops. The variant optionally blends in CLIP text features from VLM-provided category names:

```python
image_feat = normalize(CLIP_image_encoder(masked_crop))    # [M, 512]
text_feat  = normalize(CLIP_text_encoder(category_name))   # [M, 512]

combined = (1 - alpha) * image_feat + alpha * text_feat
combined = combined / ||combined||    # L2-renormalize to unit sphere
```

Both encoders are from the same OpenCLIP ViT-B/16 model, so embeddings share the same 512D joint space. L2-renormalization after blending is critical because the weighted average of two unit vectors is not itself a unit vector.

### Experiment Configurations

| Experiment | `--text_weight` | Feature Source | Rationale |
|---|---|---|---|
| `img_only` | 0.0 | Pure CLIP image features | Baseline (original LangSplat behavior) |
| `text_050` | 0.5 | 50% image + 50% text | Hedges between visual fidelity and semantic precision |
| `text_100` | 1.0 | Pure CLIP text features | Maximally aligned with text queries; loses visual info |

Each experiment runs the full pipeline independently with its own autoencoder (different 512D feature distributions require different compression).

### Why Text Blending Helps Egocentric Data

- **Masked crops are often poor quality**: partial occlusion, hands in frame, motion blur, black-background artifacts that deviate from CLIP's training distribution
- **Text features provide clean semantic anchors**: not affected by crop quality, occlusion, or lighting
- **Query-time alignment**: since evaluation computes cosine similarity against CLIP text embeddings, training features partially derived from text are already closer to the query space
- **Risk**: text features are only as good as the VLM labels; blending (alpha < 1.0) hedges against mislabeling

---

## 3. Subset Dataset Support (`scene/dataset_readers.py` — MODIFIED)

`readColmapCameras()` was modified to scan which images actually exist on disk and **skip COLMAP cameras whose images are missing**, instead of crashing:

```python
existing_image_files = set()
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', ...)):
        existing_image_files.add(filename)

for key in cam_extrinsics:
    image_filename = os.path.basename(extr.name)
    if image_filename not in existing_image_files:
        skipped_count += 1
        continue  # skip this camera
```

**Why needed**: SAM2 only processes a subset of frames. Working directories contain symlinks to only those frames, but COLMAP's `images.bin` still references ALL original frames.

---

## 4. Query Comparison Tool (`query_features.py` — NEW, 287 lines)

Cross-experiment text query tool for evaluating results:

1. Loads rendered 3D features from multiple experiments simultaneously
2. Decodes each via experiment-specific autoencoders (3D -> 512D)
3. Computes CLIP relevancy maps for user-provided text prompts
4. Applies 30px box filter smoothing (matching the paper's eval pipeline)
5. Generates per-experiment heatmaps/masks AND side-by-side comparisons

Output structure:
```
query_results/
  per_experiment/{exp}/frame_XXXXX_{prompt}_composited.jpg
  per_experiment/{exp}/frame_XXXXX_{prompt}_mask.jpg
  comparisons/frame_XXXXX_{prompt}.jpg   (RGB | img_only | text_050 | text_100)
```

---

## 5. Clustering Analysis Tools (NEW, 566 lines total)

Post-training analysis answering "what semantic groups did the Gaussians learn?"

### `cluster_extract.py` (114 lines)
- Loads 13-tuple checkpoint with `_language_feature` [N, 3]
- Filters background Gaussians (feature norm > 0.1 threshold)
- Decodes foreground 3D features to 512D via autoencoder
- Saves as `cluster_data.npz` (xyz, feat_3d, feat_512, fg_mask)

### `cluster_segment.py` (452 lines)
Runs 4 configurations (2 methods x 2 feature spaces):

| | 3D (bottleneck) | 512D (decoded CLIP) |
|---|---|---|
| **K-Means** (K=20, MiniBatch) | Fast, coarse | Balanced semantic view |
| **HDBSCAN** (auto-K, subsampled) | Very coarse (~5 clusters) | Fine-grained (~25 clusters) |

Outputs per configuration:
- `colored_gaussians.ply` — full-scene point cloud colored by cluster (HSV-spaced hues)
- `maps_2d/` — 2D cluster overlays on RGB frames (40% cluster color + 60% RGB)
- `summary.json`, `cluster_labels.npy`, `cluster_centers.npy`

**Key finding**: HDBSCAN in 3D found only 5 clusters (showing the 3D bottleneck collapse), while HDBSCAN in 512D found 25 (the richest segmentation). This directly demonstrated the autoencoder information loss.

---

## 6. Compatibility Fixes

| File | Change | Reason |
|---|---|---|
| `train.py` | `PIL.Image.ANTIALIAS = PIL.Image.LANCZOS` | Pillow 10+ removed `ANTIALIAS` |
| `train.py` | Commented out `add_histogram` | numpy compatibility with newer versions |
| `autoencoder/train.py` | Commented out `add_histogram` | Same |

---

## 7. Pipeline Orchestration Scripts (NEW)

| Script | Lines | Purpose |
|---|---|---|
| `run_pipeline.sh` | 69 | Full training for one experiment: setup -> preprocess -> AE -> train -> render |
| `run_query.sh` | 63 | Text queries across all 3 experiments, generate comparisons |
| `run_cluster.sh` | 59 | Extract features + cluster with all 4 configs |

Target dataset: HD-EPIC P01 kitchen (`/mnt/raptor/daiwei/HD-EPIC-processed/colmap-data/P01/P01-20240202-110250-full`)

---

## Complete File Inventory

### Modified from Original LangSplat
| File | Change |
|---|---|
| `scene/dataset_readers.py` | Skip missing images for subset datasets (~25 lines) |
| `train.py` | Pillow compat + histogram fix (2 lines) |
| `autoencoder/train.py` | Histogram fix (1 line) |

### New Files
| File | Lines | Category |
|---|---|---|
| `preprocess_sam2.py` | 392 | Preprocessing (SAM2 + text blending) |
| `query_features.py` | 287 | Evaluation (cross-experiment query) |
| `cluster_extract.py` | 114 | Analysis (feature extraction) |
| `cluster_segment.py` | 452 | Analysis (clustering + visualization) |
| `run_pipeline.sh` | 69 | Orchestration |
| `run_query.sh` | 63 | Orchestration |
| `run_cluster.sh` | 59 | Orchestration |
| `docs/modifications-summary.md` | 361 | Documentation |
| `docs/pipeline-explanation.md` | 169 | Documentation |
| `docs/clustering-explanation.md` | 249 | Documentation |

### Untouched
Renderer (`gaussian_renderer/__init__.py`), Gaussian model (`scene/gaussian_model.py`), autoencoder architecture (`autoencoder/model.py`), loss functions (`utils/loss_utils.py`), CUDA rasterizer (`submodules/langsplat-rasterization/`), and all other core files remain identical to the upstream LangSplat.

---

## Relationship to Current Pipeline (Ego3DVQA-GS/LangSplat)

The current LangSplat-Ego pipeline at `/home/daiwei/Ego3DVQA-GS/LangSplat/` takes a different approach:
- Uses **covisibility + blur frame selection** instead of SAM2 frame subsets
- Uses **original SAM** (not SAM2) with per-image processing and resume support
- Uses **hand/dynamic object mask postprocessing** on seg maps (vs. static mask AND-filtering during preprocessing)
- Trains **all 3 feature levels** (vs. single level in the variant)
- Does **not** use text feature blending (pure image features only)
- Adds its own evaluation script (`eval_ego3dvqa.py`) tailored to Ego3DVQA datasets

The variant's text blending approach remains unexplored in the current pipeline and could address the [feature quality issues](../experiments/04-09_langsplat-egocentric-quality-analysis.md) — particularly the autoencoder bottleneck collapse, since text features are already aligned with the query space.
