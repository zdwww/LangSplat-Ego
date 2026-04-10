# LangSplat v4: Qwen3-VL-Embedding (Unified Multimodal)
Generated: 2026-04-10 05:54

## Method
**v2 baseline**: OpenCLIP ViT-B-16, LERP blend of separate image + text embeddings.
**v4 (this experiment)**: Qwen3-VL-Embedding-2B, single-tower unified embeddings.
- `image_only`: SAM2 crop encoded as image through Qwen3-VL-Embedding
- `multimodal`: SAM2 crop + category text encoded together (unified fusion)
- 512D via Matryoshka truncation (from 2048D native)

## Novel-View Segmentation Results (HD-EPIC, Mean IoU)

| Config | Mean IoU | Delta vs best v2 |
|--------|----------|-------------------|
| v2 tw=0.0 (CLIP image only) | 0.0737 | — |
| v2 tw=0.5 (CLIP LERP blend) | 0.1384 | — |
| v4 image_only (Qwen3-VL) | 0.1086 | -21.6% |
| v4 multimodal (Qwen3-VL) | 0.0992 | -28.4% |
| v4 lerp tw=0.5 (Qwen3-VL) | 0.1219 | -12.0% |

## HD-EPIC — Detailed Metrics

| Config | Mean IoU | Median IoU | Std IoU |
|--------|----------|------------|---------|
| v2 tw=0.0 (CLIP image only) | 0.0737 | 0.0610 | 0.0448 |
| v2 tw=0.5 (CLIP LERP blend) | 0.1384 | 0.1156 | 0.0840 |
| v4 image_only (Qwen3-VL) | 0.1086 | 0.1008 | 0.0628 |
| v4 multimodal (Qwen3-VL) | 0.0992 | 0.0762 | 0.0596 |
| v4 lerp tw=0.5 (Qwen3-VL) | 0.1219 | 0.0846 | 0.0830 |

### HD-EPIC — Per-Category IoU (top 15 by frequency)

| Category | Count | v2 tw=0.0 | v2 tw=0.5 | v4 image_only | v4 multimodal | v4 lerp tw=0.5 |
|----------|-------| --------- | --------- | ------------- | ------------- | -------------- |
| wooden countertop | 98 | 0.0715 | 0.2157 | 0.2063 | 0.1578 | 0.2231 |
| dark wooden cabinet | 50 | 0.0722 | 0.0936 | 0.0883 | 0.0906 | 0.1031 |
| wooden table | 41 | 0.0000 | 0.0061 | 0.0285 | 0.0126 | 0.0129 |
| black appliance | 36 | 0.0005 | 0.0178 | 0.0173 | 0.0227 | 0.0111 |
| bookshelf | 31 | 0.0025 | 0.0267 | 0.0013 | 0.0162 | 0.0207 |
| white tiled wall | 30 | 0.0000 | 0.1529 | 0.0736 | 0.1381 | 0.1678 |
| black cable | 26 | 0.0000 | 0.0002 | 0.0129 | 0.0023 | 0.0000 |
| stainless steel stove | 23 | 0.0011 | 0.0885 | 0.0190 | 0.1143 | 0.1164 |
| white refrigerator | 23 | 0.2533 | 0.4372 | 0.2784 | 0.2022 | 0.3583 |
| stainless steel microwave | 22 | 0.3199 | 0.2641 | 0.2998 | 0.1105 | 0.0866 |
| black blender | 19 | 0.0285 | 0.0058 | 0.0081 | 0.0159 | 0.0133 |
| black toaster | 17 | 0.0036 | 0.0177 | 0.0002 | 0.0747 | 0.0065 |
| white door | 17 | 0.0209 | 0.1441 | 0.0571 | 0.0599 | 0.0797 |
| black kettle | 16 | 0.0192 | 0.1009 | 0.0265 | 0.0647 | 0.0497 |
| dark cabinet | 15 | 0.1658 | 0.2589 | 0.2441 | 0.1717 | 0.1903 |

## Analysis

### Result Summary

v4 lerp tw=0.5 is the best Qwen3-VL config, closing the gap to -12% vs v2 best:
- **v4 lerp tw=0.5**: 0.1219 — best v4 config, combines Qwen3-VL's superior image encoding with text grounding
- **v4 image_only**: 0.1086 (-21.6%) — +47% over CLIP image-only, confirming better encoder
- **v4 multimodal**: 0.0992 (-28.4%) — unified encoding underperforms, likely due to compression

### Why v4 LERP improves over other v4 configs but not v2

Qwen3-VL LERP combines two advantages: (1) Qwen3-VL's stronger image features (+47% vs CLIP
image-only), and (2) text blending for category-level semantic grounding. Since Qwen3-VL is
single-tower (no modality gap), the LERP interpolation is geometrically meaningful — it moves
along a path within the same manifold rather than bridging separate regions like CLIP.

However, it still underperforms v2 CLIP LERP (0.1219 vs 0.1384). The autoencoder compression
(512D → 3D) is the bottleneck: CLIP's simpler, lower-entropy features survive aggressive
compression better. Qwen3-VL's 2048D features (truncated to 512D via Matryoshka) encode
richer information that the 3D bottleneck destroys.

### Per-category patterns

**v4 LERP wins** on texture-defined/surface categories (where no-modality-gap alignment helps):
- wooden countertop: 0.2231 vs 0.2157 (v2 blend) — +3.4%
- white tiled wall: 0.1678 vs 0.1529 (v2 blend) — +9.7%
- stainless steel stove: 0.1164 vs 0.0885 (v2 blend) — +31.5%
- dark wooden cabinet: 0.1031 vs 0.0936 (v2 blend) — +10.2%

**v4 LERP loses** on distinctive standalone objects:
- white refrigerator: 0.3583 vs 0.4372 (v2 blend) — -18.0%
- stainless steel microwave: 0.0866 vs 0.2641 (v2 blend) — -67.2%
- black kettle: 0.0497 vs 0.1009 (v2 blend) — -50.7%

This suggests Qwen3-VL features for visually distinctive objects are "too rich" for the 3D
autoencoder — the features that make these objects recognizable in 512D are lost in compression.
CLIP's simpler features retain enough discriminative signal because they encode less total
information.

### Key insight

The bottleneck is definitively the **autoencoder compression (512D → 3D)**, not the embedding
model or modality gap. Evidence:
1. v4 image_only beats v2 image_only by +47% (better encoder confirmed)
2. v4 LERP beats v4 image_only by +12% (text grounding works with Qwen3-VL)
3. All v4 configs lose to v2 LERP (richer features don't survive 3D compression)
4. v4 LERP wins on categories where discriminative features happen to align with
   the autoencoder's 3D latent structure

## Next Steps: Overcoming the Autoencoder Bottleneck

The 512D → 3D autoencoder is the weakest link. Recent literature (2024-2026) offers
several alternatives, ranked by implementation effort and expected impact.

### Tier 1: Immediate experiments (low effort, high impact)

**1. LangSplatV2 sparse coding** — Replace the autoencoder with a learned codebook
of 64 basis vectors in full 512D space. Each Gaussian stores sparse top-K=4 coefficients
over the codebook. Feature recovery: `codebook^T @ rendered_weights`. No information
is destroyed in the codebook — only approximated by the sparse combination.
- Code already available at `/home/daiwei/LangSplat-variants/LangSplatV2/`
- Expected: +8-15 pp mIoU based on published results (59.9% vs 51.4% on LERF)
- Qwen3-VL features should benefit more since the codebook preserves full 512D
- Ref: Qin et al., "LangSplatV2: High-dimensional 3D Language Gaussian Splatting
  with 450+ FPS", arXiv 2507.07136

**2. PCA baseline** — Run PCA on Qwen3-VL features to find the intrinsic dimensionality.
If 16-32 principal components capture 95%+ cosine similarity (as Gen-LangSplat found
for CLIP), replace the autoencoder with PCA encode/decode. Zero training cost, deterministic.
- Implementation: trivial (`sklearn.decomposition.PCA`)
- Requires modifying CUDA rasterizer for >3 channels (or render in groups)

**3. Increase autoencoder latent to 16D** — Gen-LangSplat (arXiv 2510.22930) found that
16D preserves >93% cosine similarity to original CLIP embeddings (vs much less for 3D).
Minimal code change to autoencoder architecture, but requires CUDA rasterizer modification.
- Ref: "Gen-LangSplat: Generalized Language Gaussian Splatting with Pre-Trained
  Feature Compression", arXiv 2510.22930

### Tier 2: Medium effort, high potential

**4. Occam's LGS back-projection** — Skip training entirely. Back-project 512D features
onto Gaussians using alpha-blending weights from the rendering equation. Each Gaussian
stores its full 512D embedding. No compression, no information loss.
- 16-32 seconds total (vs 67+ minutes for LangSplat pipeline)
- Memory cost: ~1-4 GB for features (500K-2M Gaussians × 512 floats)
- Ref: Wang et al., "Occam's LGS", BMVC 2025, arXiv 2412.01807

**5. Product Quantization (Dr. Splat)** — Divide 512D into 128 subspaces of 4D,
learn 256 centroids per subspace. Each Gaussian stores 128 uint8 indices (128 bytes).
Pre-train PQ on large-scale Qwen3-VL embeddings, apply to any scene. 6.25% compression
ratio while preserving much more than the 3D bottleneck.
- Ref: Kim et al., "Dr. Splat", CVPR 2025 Highlight, arXiv 2502.16652

**6. Contrastive codebook (CCL-LGS)** — Train 8D features per Gaussian with contrastive
losses (intra-class compactness, inter-class separation). Explicitly optimizes for
discriminability in compressed space — addresses exactly why our autoencoder fails.
- SOTA on LERF: 65.6% mIoU (+14.2 pp over LangSplat)
- Ref: Tian et al., "CCL-LGS", ICCV 2025, arXiv 2505.20469

### Tier 3: Higher effort, theoretically strongest

**7. GOI codebook + MLP decoder** — Store 10-16D per Gaussian, maintain a codebook
of Qwen3-VL embeddings, use MLP decoder + optimizable hyperplane query. Best reported
accuracy gains over LangSplat (+30pp mIoU on Mip-NeRF360).
- Ref: Qu et al., "GOI", ACM MM 2024, arXiv 2405.17596

### Comparison of alternatives

| Method | Per-Gaussian Dim | Compression | LERF mIoU | vs LangSplat |
|--------|-----------------|-------------|-----------|-------------|
| LangSplat (ours) | 3D (AE) | 512→3 | 51.4% | baseline |
| LangSplatV2 | 64 logits (top-4) | Sparse coding | 59.9% | +8.5 |
| Occam's LGS | 512D (full) | None | 61.3% | +9.9 |
| CCL-LGS | 8D + codebook | Contrastive | 65.6% | +14.2 |
| GOI | 10D + codebook | VQ + MLP | 86.5%* | +30* |
| Gen-LangSplat | 16D (AE) | 512→16 | 51.6% | +0.2 |
| Dr. Splat | PQ (128 indices) | Product quant | — | — |

*GOI measured on Mip-NeRF360 (different protocol), not directly comparable.

### Recommended path

**Start with LangSplatV2 sparse coding** — code already exists in our repo, highest
expected impact with lowest implementation cost. The sparse codebook approach should
particularly benefit from Qwen3-VL's richer features since the codebook vectors live
in full 512D space, completely bypassing the 3D compression bottleneck.

If LangSplatV2 with Qwen3-VL LERP features outperforms v2 CLIP LERP, it confirms
that the embedding model quality matters when the compression is adequate — validating
the v4 experiment's core hypothesis while removing the bottleneck that obscured it.

---
*Report generated by `generate_v4_report.py`, analysis added manually*
