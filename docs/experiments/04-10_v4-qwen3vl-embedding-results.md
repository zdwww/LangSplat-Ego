# LangSplat v4: Qwen3-VL-Embedding (Unified Multimodal)
Generated: 2026-04-10 04:21

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

## HD-EPIC — Detailed Metrics

| Config | Mean IoU | Median IoU | Std IoU |
|--------|----------|------------|---------|
| v2 tw=0.0 (CLIP image only) | 0.0737 | 0.0610 | 0.0448 |
| v2 tw=0.5 (CLIP LERP blend) | 0.1384 | 0.1156 | 0.0840 |
| v4 image_only (Qwen3-VL) | 0.1086 | 0.1008 | 0.0628 |
| v4 multimodal (Qwen3-VL) | 0.0992 | 0.0762 | 0.0596 |

### HD-EPIC — Per-Category IoU (top 15 by frequency)

| Category | Count | v2 tw=0.0 | v2 tw=0.5 | v4 image_only | v4 multimodal |
|----------|-------| --------- | --------- | ------------- | ------------- |
| wooden countertop | 98 | 0.0715 | 0.2157 | 0.2063 | 0.1578 |
| dark wooden cabinet | 50 | 0.0722 | 0.0936 | 0.0883 | 0.0906 |
| wooden table | 41 | 0.0000 | 0.0061 | 0.0285 | 0.0126 |
| black appliance | 36 | 0.0005 | 0.0178 | 0.0173 | 0.0227 |
| bookshelf | 31 | 0.0025 | 0.0267 | 0.0013 | 0.0162 |
| white tiled wall | 30 | 0.0000 | 0.1529 | 0.0736 | 0.1381 |
| black cable | 26 | 0.0000 | 0.0002 | 0.0129 | 0.0023 |
| stainless steel stove | 23 | 0.0011 | 0.0885 | 0.0190 | 0.1143 |
| white refrigerator | 23 | 0.2533 | 0.4372 | 0.2784 | 0.2022 |
| stainless steel microwave | 22 | 0.3199 | 0.2641 | 0.2998 | 0.1105 |
| black blender | 19 | 0.0285 | 0.0058 | 0.0081 | 0.0159 |
| black toaster | 17 | 0.0036 | 0.0177 | 0.0002 | 0.0747 |
| white door | 17 | 0.0209 | 0.1441 | 0.0571 | 0.0599 |
| black kettle | 16 | 0.0192 | 0.1009 | 0.0265 | 0.0647 |
| dark cabinet | 15 | 0.1658 | 0.2589 | 0.2441 | 0.1717 |

## Analysis

### Result Summary

Neither v4 config beats v2 tw=0.5 (CLIP LERP blend):
- **v4 image_only**: 0.1086 (-21.6%) — significantly better than CLIP image-only (+47%), but worse than CLIP blend
- **v4 multimodal**: 0.0992 (-28.4%) — unexpectedly worse than v4 image_only

### Why v4 image_only improves over v2 image_only but not v2 blend

Qwen3-VL-Embedding-2B is a more powerful vision encoder than CLIP ViT-B-16 (2B vs 150M params,
SigLIP-2 + DeepStack fusion vs single ViT). This explains the +47% improvement in image-only
mode (0.1086 vs 0.0737). The richer image features capture more semantic information per crop.

However, it still underperforms v2 tw=0.5 because text blending in v2 provides explicit
category-level semantic grounding that pure image features lack, regardless of encoder quality.
The v2 blend essentially tells each pixel "you are a wooden countertop" via the text component,
which directly matches the text query at eval time.

### Why multimodal underperforms image_only

This is the most surprising result. Possible explanations:

1. **Autoencoder compression bottleneck (512D -> 3D)**: Multimodal embeddings encode richer
   information (visual + textual) in 512D, but the autoencoder compresses to just 3D. The
   additional text information may be lost in compression, while the image-specific patterns
   that survive compression are diluted by the text component.

2. **Matryoshka truncation (2048D -> 512D)**: The multimodal embedding may distribute
   image and text information across all 2048 dimensions. Truncating to 512D might
   disproportionately lose the visual component that's critical for spatial discrimination.

3. **Feature variance**: Multimodal embeddings for different objects with the same category
   label become more similar (text dominates), reducing the model's ability to discriminate
   between instances — similar to the precision drop seen in v2 tw=1.0 (text only).

### Per-category patterns

**v4 image_only wins** on distinctive/large objects:
- stainless steel microwave: 0.300 vs 0.264 (v2 blend) — +14%
- wooden table: 0.029 vs 0.006 (v2 blend) — +370%
- black cable: 0.013 vs 0.000 (v2 blend) — from zero

**v4 image_only loses** on texture-defined/generic objects:
- white tiled wall: 0.074 vs 0.153 (v2 blend) — -52%
- stainless steel stove: 0.019 vs 0.089 (v2 blend) — -79%
- white door: 0.057 vs 0.144 (v2 blend) — -60%

This pattern suggests image-only features from Qwen3-VL excel at distinguishing visually
unique objects but struggle with generic surfaces where text labels provide critical disambiguation.

**v4 multimodal wins** on text-disambiguated objects:
- stainless steel stove: 0.114 vs 0.089 (v2 blend) — +29%
- black toaster: 0.075 vs 0.018 (v2 blend) — +320%
- black appliance: 0.023 vs 0.018 (v2 blend) — +28%

This confirms multimodal encoding helps where text disambiguation matters, but the gains are
outweighed by losses on high-frequency categories like "wooden countertop" (0.158 vs 0.216).

## Conclusion

Replacing OpenCLIP with Qwen3-VL-Embedding-2B improves image-only features substantially (+47%),
confirming that encoder quality matters. However, neither encoding mode beats v2's simple
CLIP LERP blend (tw=0.5).

**Key insight**: The bottleneck is not the embedding model or modality gap — it's the
autoencoder compression (512D -> 3D). With only 3 dimensions to represent language features,
the simpler CLIP+text blend creates features that are more compressible and more directly
aligned with text queries.

**Implications for future work**:
- Increasing the autoencoder latent dimension (e.g., 3D -> 8D or 16D) could allow richer
  Qwen3-VL features to survive compression
- Using higher Matryoshka dimensions (768D, 1024D) with a matched autoencoder may help
- Direct feature assignment methods (Dr. Splat, Occam's LGS) that bypass the autoencoder
  entirely could benefit most from Qwen3-VL's richer embeddings
- The v2 LERP blend (tw=0.5) remains the best configuration in the current pipeline

---
*Report generated by `generate_v4_report.py`, analysis added manually*
