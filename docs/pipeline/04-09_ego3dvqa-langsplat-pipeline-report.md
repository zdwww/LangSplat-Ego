# LangSplat on Ego3DVQA Datasets - Pipeline Report

**Date**: 2026-04-09
**LangSplat code**: `/home/daiwei/Ego3DVQA-GS/LangSplat/` (fresh clone of `minghanqin/LangSplat`)
**Workspace**: `/mnt/raptor/daiwei/LangSplat-workspace/`
**Pipeline script**: `run_ego3dvqa_pipeline.sh` / `run_ego3dvqa_pipeline_resume.sh`
**Total wall time**: ~2 hours (preprocessing) + ~1.5 hours (training 6 models on 4 GPUs) = ~3.5 hours

---

## 1. Data Modifications

### Frame Selection: Covisibility + Blur Filtering

Instead of using all homography-filtered training frames (1,375 ADT / 3,911 HD-EPIC), we used **covisibility-based frame selection** (same algorithm as the VLM pipeline in `gaussian-splatting/utils/covisibility.py`):

1. **Greedy set cover** (phase 1): Select frames that maximize Gaussian visibility coverage
2. **IoU diversity sampling** (phase 2): Add frames with most novel viewpoints until `diversity_threshold` exceeded
3. **Blur rejection**: Remove bottom 15th percentile by Laplacian variance, with neighbor substitution

| Dataset | Total Frames | Covisibility Selected | After Blur | Coverage | Final Used |
|---------|-------------|----------------------|------------|----------|------------|
| **ADT** | 3,527 | 90 (threshold=0.7) | 76 | 100% | 76 |
| **HD-EPIC** | 11,892 | 138 (threshold=0.7) | 117 | 100% | 119* |

*119 includes 2 frames from an earlier selection run that remained in the workspace.

### 2x Image Downscale

All images downscaled by 2x before processing:
- **ADT**: 1172x1173 -> **586x586**
- **HD-EPIC**: 1187x1188 -> **593x594**

This is safe because `loadCam()` uses actual PIL image dimensions and FoV from COLMAP is angular (resolution-independent).

### Mask Application (Option A)

- SAM ran on **unmasked** images for better segmentation quality
- After SAM preprocessing, seg maps were post-processed: pixels in masked regions (hands + dynamic objects) set to `-1`
- Masks from `masks_rle.json` (dynamic objects) AND `hand_masks_rle.json` (hands) combined with AND
- ADT: 35/76 frames had masks applied, **1.9%** total pixels masked
- HD-EPIC: 92/119 frames had masks applied, **3.9%** total pixels masked

---

## 2. Pipeline Modifications to LangSplat

### New Scripts (in `/home/daiwei/Ego3DVQA-GS/LangSplat/`)

| Script | Purpose |
|--------|---------|
| `select_frames.py` | Covisibility + blur frame selection using trained 3DGS model |
| `prepare_ego3dvqa_workspace.py` | 2x downscale selected frames, symlink COLMAP, copy masks |
| `postprocess_segmaps.py` | Apply RLE masks to seg maps (set masked pixels to -1) |
| `visualize_preprocessing.py` | Generate diagnostic montages and coverage stats |
| `eval_ego3dvqa.py` | Qualitative CLIP relevancy evaluation without LERF ground truth |
| `run_ego3dvqa_pipeline.sh` | Master orchestration script |
| `utils/covisibility.py` | Copied from gaussian-splatting for frame selection |

### Modified Files

| File | Change |
|------|--------|
| `preprocess.py` | Per-image processing instead of bulk `torch.cat()` (avoids OOM, adds resume support) |
| `scene/dataset_readers.py` | `readColmapCameras()` now scans disk first, skips missing images |
| `autoencoder/dataset.py` | Added `.astype(np.float32)` to handle CLIP's fp16 output |
| `autoencoder/train.py` | Commented out `add_histogram` (numpy/tensorboard incompatibility) |

---

## 3. Errors Encountered and Resolved

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: pycocotools` | Not in langsplat_v2 env | `pip install pycocotools` |
| `RuntimeError: expected Float but found Half` | CLIP saves fp16, autoencoder expects fp32 | Cast to float32 in dataset.py |
| `FileNotFoundError` on camera loading | Fresh clone tries to open ALL COLMAP images | Patched `readColmapCameras` to skip missing |
| `TypeError: ufunc greater` in tensorboard | numpy 1.26 incompatible with old tensorboard histogram | Commented out `add_histogram` call |
| `GLIBCXX_3.4.29 not found` | System libstdc++ too old for conda packages | Set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| SAM `double free or corruption` crash | Sporadic CUDA memory corruption | Resume support in `create()` skips completed images |

---

## 4. Preprocessing Quality

### Segmentation Coverage (after mask post-processing)

| Level | ADT Mean | ADT Min | HD-EPIC Mean | HD-EPIC Min |
|-------|----------|---------|-------------|-------------|
| L0 (default) | 94.1% | 85.6% | 91.4% | 74.5% |
| L1 (small) | 86.7% | 59.3% | 83.5% | 61.1% |
| L2 (medium) | 92.9% | 79.4% | 90.5% | 71.4% |
| L3 (large) | 87.0% | 54.7% | 81.5% | 27.8% |

Diagnostic montages saved to:
- `ADT_seq131/diagnostics/preprocessing_samples.png`
- `HDEPIC_P01/diagnostics/preprocessing_samples.png`

### Autoencoder Performance

| Dataset | Best Epoch | Eval Loss (L2 + cosine) |
|---------|-----------|------------------------|
| ADT | 98 | **0.2415** |
| HD-EPIC | 98 | **0.2598** |

---

## 5. Language Feature Training

All models trained for 30,000 iterations with frozen geometry (`include_feature=True`).

| Experiment | GPU | Final Loss | Time |
|-----------|-----|-----------|------|
| ADT L1 (small) | 0 | 0.223 | ~45 min |
| ADT L2 (medium) | 6 | 0.194 | ~45 min |
| ADT L3 (large) | 0 | 0.174 | ~20 min |
| HDEPIC L1 (small) | 4 | 0.278 | ~45 min |
| HDEPIC L2 (medium) | 7 | 0.260 | ~45 min |
| HDEPIC L3 (large) | 4 | 0.214 | ~20 min |

Checkpoints saved at iterations 7,000 and 30,000 for each.

---

## 6. Evaluation Results

Qualitative evaluation using CLIP relevancy (softmax over positive vs. negative prompts). Higher = better localization.

### ADT Apartment

| Query | Best Level | Mean Relevancy | Max Relevancy |
|-------|-----------|---------------|--------------|
| sofa | L2 | **0.660** | 0.684 |
| table | L3 | **0.619** | 0.641 |
| bookshelf | L2 | **0.581** | 0.598 |
| sink | L3 | **0.519** | 0.522 |
| refrigerator | L2 | **0.502** | 0.506 |
| kitchen counter | L3 | **0.466** | 0.469 |

### HD-EPIC Kitchen

| Query | Best Level | Mean Relevancy | Max Relevancy |
|-------|-----------|---------------|--------------|
| stove | L3 | **0.680** | 0.709 |
| cutting board | L3 | **0.680** | 0.710 |
| pan | L2 | **0.634** | 0.637 |
| knife | L1 | **0.601** | 0.604 |
| fridge | L2 | **0.560** | 0.597 |
| kettle | L3 | **0.530** | 0.539 |

Heatmap visualizations saved to `*/eval_results/{query}/frame_*.png`.

**Observations**:
- Large, distinct objects (sofa, stove, cutting board) score highest (0.66-0.68)
- Small objects (knife) or texturally ambiguous surfaces (kitchen counter) score lower (0.47-0.60)
- Level 2 (medium) and Level 3 (large) are most frequently selected as best, consistent with egocentric scenes where objects occupy medium-to-large portions of the frame
- HD-EPIC generally shows higher scores than ADT, likely due to the kitchen scene having more distinct, nameable objects

---

## 7. Output Directory Structure

```
/mnt/raptor/daiwei/LangSplat-workspace/
├── REPORT.md
├── logs/                           # All step logs
│   ├── step1_ADT.log ... step7_HDEPIC.log
│   └── pipeline_resume2.log        # Master pipeline log
├── weights/
│   └── sam_vit_h_4b8939.pth
├── ADT_seq131/
│   ├── selected_frames.json        # Frame selection results
│   ├── images/                     # 76 half-res images
│   ├── sparse/0/                   # Symlink to COLMAP
│   ├── masks_rle.json
│   ├── hand_masks_rle.json
│   ├── language_features/          # 76 SAM+CLIP features
│   ├── language_features_dim3/     # 76 compressed (3D) features
│   ├── diagnostics/                # Preprocessing montage + stats
│   ├── output_1/                   # L1 model + renders
│   ├── output_2/                   # L2 model + renders
│   ├── output_3/                   # L3 model + renders
│   └── eval_results/               # Heatmap visualizations
└── HDEPIC_P01/
    └── (same structure, 119 frames)
```
