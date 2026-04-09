# LangSplat on Egocentric Data — Quality Analysis

**Date**: 2026-04-09
**Datasets**: ADT Apartment (seq131), HD-EPIC Kitchen (P01)
**Workspace**: `/mnt/raptor/daiwei/LangSplat-workspace/`
**Related**: [Pipeline Report](../pipeline/04-09_ego3dvqa-langsplat-pipeline-report.md)

---

## Summary

LangSplat (CVPR 2024) was run on two Ego3DVQA egocentric datasets. The pipeline completed successfully but **evaluation heatmaps show poor segmentation quality** — most text queries produce near-uniform relevancy maps with no clear object boundaries. This document traces the root causes through the full pipeline.

---

## 1. Symptom

Evaluation visualizations under `eval_results/` were uninterpretable:
- Heatmaps appeared as uniform cyan/green wash
- "refrigerator" query highlighted arbitrary furniture equally
- No clear object boundaries in any text query

### Visualization Bug (fixed)

The original `eval_ego3dvqa.py` used:
- `jet` colormap with hardcoded `vmin=0, vmax=1`
- Actual relevancy values occupied only a ~0.2-wide band (e.g., 0.23–0.44 for "sofa")
- Overlay alpha `rel_map * 2` → uniformly high (~0.6–0.7) everywhere

**Fix**: Switched to `turbo` colormap, percentile-based normalization (2nd–98th), and threshold-based overlay alpha. This revealed some spatial structure but confirmed the underlying features are weak.

---

## 2. Quantitative Diagnostics

### Pipeline Stage Measurements

| Stage | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| AE reconstruction | per-mask cosine sim (512D→3D→512D) | **0.91** | 9% semantic loss per mask |
| AE decoder output | random unit 3D → decoded 512D pairwise cos | **0.78** | Decoder maps entire sphere to small 512D region |
| AE decoder output | actual data 3D → decoded 512D pairwise cos | **0.78** | Same — intrinsic decoder limitation |
| Compressed 3D features | pairwise cosine (per-mask) | **0.05** (ADT) | Good diversity at mask level |
| GT pixel features (L1) | pairwise cosine | **0.17** | Reasonable pixel-level diversity |
| GT pixel features (L3) | pairwise cosine | **0.56** | Less diverse (fewer, larger segments) |
| Rendered 3D features (L1) | pairwise cosine | **0.28** | Some smoothing from 3DGS |
| Rendered 3D features (L3) | pairwise cosine | **0.63** | More smoothing at coarser level |
| **Decoded rendered (L1)** | **pairwise cosine (512D)** | **0.83** | **Critical: nearly identical** |
| **Decoded rendered (L3)** | **pairwise cosine (512D)** | **0.84** | **Critical: nearly identical** |

### Relevancy Score Distribution (ADT, "sofa", best query)

| Percentile | Value |
|-----------|-------|
| 1st | 0.23 |
| 25th | 0.32 |
| 50th | 0.34 |
| 75th | 0.39 |
| 95th | 0.42 |
| 99th | 0.44 |
| max | 0.69 |

The 1st-to-99th percentile range is only 0.21 — essentially no contrast for text-query discrimination.

### Feature Norm Mismatch

| Level | GT norm | Rendered norm (mean) |
|-------|---------|---------------------|
| L1 | 1.00 | 0.64 |
| L2 | 1.00 | 0.72 |
| L3 | 1.00 | 0.75 |

GT features are unit-normalized (encoder normalizes output). Rendered features have lower norm due to alpha-blending of diversely-directed unit vectors. L1 loss cannot close this gap.

---

## 3. Root Causes (ordered by impact)

### 3.1 Autoencoder 3D Bottleneck (FUNDAMENTAL)

The 512D→3D→512D autoencoder is the primary bottleneck. The decoder maps the **entire unit 3-sphere** onto a small region of 512D space:

```
Random 3D unit vectors → decoded 512D pairwise cosine = 0.78
```

This is an intrinsic property of the trained decoder, not a data issue. Even if 3DGS produced perfect 3D features, the decoder cannot produce diverse enough 512D outputs to discriminate semantically similar objects.

**Why it works for the paper's scenes**: The paper evaluates on tabletop scenes (figurines, ramen) and simple rooms (3D-OVS). Objects are visually and semantically distinct (Pikachu vs Xbox controller vs UNO cards). Their CLIP features are far apart in 512D, so even after the 3D bottleneck, relative ordering is preserved. Indoor egocentric scenes have many semantically similar surfaces (counter vs table vs fridge vs bookshelf) whose CLIP features are close — the 3D bottleneck collapses these distinctions.

**Paper's autoencoder dimension study** (Table 7, Appendix): d=3 gives 94.19% mIoU on 3D-OVS bench scene. But 3D-OVS has highly distinct long-tail objects, not similar indoor furniture.

### 3.2 Low Gaussian Count

| | Paper | Our Data |
|---|-------|----------|
| Gaussians per scene | ~2,500,000 | **329,549** (ADT) |
| Ratio | 1x | **0.13x** |

Each Gaussian covers ~7.5x more spatial area. Alpha-blending in the rasterizer averages features across larger regions, destroying fine-grained segment boundaries. This is visible in the norm reduction: rendered norms of 0.64 (L1) indicate significant direction cancellation from averaging differently-directed feature vectors.

**Why**: Our 3DGS models were trained with a different pipeline (Ego3DVQA-GS, 45K iterations) focused on visual quality. The paper trains 3DGS from scratch for 30K iterations with default densification, producing 8x more Gaussians for comparable scenes.

### 3.3 Training Signal Limitations

- **76/119 training views**: At the low end for multi-view supervision. The paper's scenes likely have 100-300 images with better angular coverage.
- **L1 loss on normalized features**: The norm mismatch (rendered ~0.7 vs GT 1.0) consumes a large fraction of the loss budget. Final training loss plateaus at 0.22 — significantly above 0 — indicating the model cannot fit the targets well.
- **Masked pixels**: 2-4% of pixels masked as -1, slightly reducing supervision signal. Minor factor.

### 3.4 Egocentric Scene Characteristics

- **Large spatial extent**: Apartment/kitchen spans many meters (vs paper's tabletop scenes spanning <1m)
- **Similar surfaces dominate**: Walls, floors, counters, tables are semantically close in CLIP space
- **Narrow FOV egocentric views**: Each frame sees only a portion of the scene; objects appear at varying scales and partial occlusions
- **Dynamic content**: Even after masking hands/objects, remaining scene geometry may be subtly different across frames

---

## 4. Decoder Sensitivity Analysis

Perturbation test: add noise to 3D unit vectors, measure effect on decoded 512D cosine similarity.

| 3D perturbation (eps) | 3D cosine | 512D cosine | Sensitivity ratio |
|----------------------|-----------|-------------|-------------------|
| 0.01 | 0.9999 | 0.9997 | 3.0x less sensitive |
| 0.05 | 0.9979 | 0.9946 | 2.6x less sensitive |
| 0.10 | 0.9909 | 0.9793 | 2.3x less sensitive |
| 0.20 | 0.9669 | 0.9561 | 1.3x less sensitive |
| 0.50 | 0.7495 | 0.8671 | 0.5x less sensitive |

The decoder is consistently **less sensitive** than the input — small 3D direction changes produce even smaller 512D changes. This smoothing behavior means fine-grained feature distinctions in 3D space are further suppressed in the decoded 512D evaluation space.

---

## 5. Comparison: Paper's Domain vs Ours

| Factor | LangSplat Paper (LERF/3D-OVS) | Our Egocentric Data |
|--------|-------------------------------|---------------------|
| Scene type | Tabletop, single room | Large apartment/kitchen |
| Spatial extent | <1m (tabletop) to ~5m (room) | 10-20m traversal |
| Object types | Visually distinct (toys, food, cards) | Semantically similar (furniture, surfaces) |
| CLIP feature separation | High (distinct visual appearance) | Low (similar indoor surfaces) |
| Image count | ~100-300 | 76 (ADT), 119 (HDEPIC) |
| Gaussian count | ~2.5M | 329K (ADT), ~500K (HDEPIC) |
| Camera trajectories | Dense, multi-orbit captures | Sparse, linear egocentric paths |
| Image resolution | 988x731 to 1440x1080 | 586x586 to 593x594 (2x downscaled) |

---

## 6. What We Changed vs Original LangSplat Pipeline

| Change | Impact on Quality | Severity |
|--------|-------------------|----------|
| Used pre-trained checkpoint (329K Gaussians) instead of training from scratch (~2.5M) | Major — reduces spatial feature resolution by ~7.5x | **High** |
| 76/119 frames instead of ~100-300 | Moderate — fewer multi-view constraints | Medium |
| 2x image downscale | Minor — paper handles similar resolutions | Low |
| Mask postprocessing (seg_map = -1) | Minor — only 2-4% of pixels affected | Low |
| `readColmapCameras` skip missing images | None — correctly handles subset datasets | None |
| Autoencoder fp16→fp32 cast | None — fixes a bug | None |
| Per-image SAM processing with resume | None — same output as batch processing | None |

---

## 7. Potential Improvements

### Short-term (improve visualization/evaluation)
- [x] Fix heatmap colormap and normalization (done: turbo + percentile-based)
- [ ] Add spatial smoothing (mean filter, as paper does for localization/segmentation)
- [ ] Try per-pixel thresholding with different cutoffs for binary segmentation

### Medium-term (improve feature quality)
- [ ] Increase autoencoder latent dimension from 3 to 16 or 32 (paper shows d=8 gives 95.61% mIoU)
- [ ] Train fresh 3DGS with default densification to get ~2M+ Gaussians
- [ ] Use all homography-filtered frames for autoencoder training, covisibility subset for 3DGS feature training

### Long-term (architectural changes)
- [ ] Replace scene-specific autoencoder with a fixed pre-trained dimension reduction (e.g., PCA on CLIP features)
- [ ] Use OpenSeg or SigLIP for pixel-level features instead of SAM crop → CLIP encode
- [ ] Explore higher-capacity per-Gaussian feature representations (e.g., feature SH coefficients)
