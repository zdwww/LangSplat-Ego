# LangSplat v3: SLERP + Adaptive Per-Segment Weights
Generated: 2026-04-09 23:54

## Method
**v2 baseline**: Linear interpolation (LERP) with fixed text_weight across all segments.
**v3 (this experiment)**: Spherical linear interpolation (SLERP) with adaptive per-segment
text weight based on image-text cosine similarity:

```
sim = cosine_similarity(image_feat, text_feat)
tw_adaptive = (1 - sim) * max_tw   # high sim → low text weight
combined = SLERP(image_feat, text_feat, tw_adaptive)
```

## Novel-View Segmentation Results (Mean IoU)

| Config | ADT IoU | ADT delta | HDEPIC IoU | HDEPIC delta |
|--------|---------|-----------|------------|--------------|
| v2 tw=0.0 (image only) | 0.0908 | — | 0.0737 | — |
| v2 tw=0.5 (LERP blend) | 0.0974 | — | 0.1384 | — |
| v2 tw=1.0 (text only) | 0.0840 | — | 0.1085 | — |
| v3 SLERP max=0.5 | 0.0928 | -4.7% | 0.1091 | -21.2% |
| v3 SLERP max=1.0 | 0.0751 | -22.9% | 0.1011 | -27.0% |

## ADT — Detailed Metrics

| Config | Mean IoU | Median IoU | Std IoU |
|--------|----------|------------|---------|
| v2 tw=0.0 (image only) | 0.0908 | 0.0706 | 0.0668 |
| v2 tw=0.5 (LERP blend) | 0.0974 | 0.0780 | 0.0719 |
| v2 tw=1.0 (text only) | 0.0840 | 0.0649 | 0.0651 |
| v3 SLERP max=0.5 | 0.0928 | 0.0750 | 0.0719 |
| v3 SLERP max=1.0 | 0.0751 | 0.0564 | 0.0642 |

### ADT — Per-Category IoU (top 15 by frequency)

| Category | Count | v2 tw=0.0 | v2 tw=0.5 | v2 tw=1.0 | v3 SLERP max=0.5 | v3 SLERP max=1.0 |
|----------|-------| --------- | --------- | --------- | ---------------- | ---------------- |
| blue glass bottle | 56 | 0.0000 | 0.0032 | 0.0011 | 0.0063 | 0.0029 |
| blue bottle | 43 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| black window | 30 | 0.0494 | 0.0782 | 0.0647 | 0.0674 | 0.0550 |
| dark wooden table | 20 | 0.0354 | 0.0978 | 0.1027 | 0.0860 | 0.0897 |
| glass door | 19 | 0.0587 | 0.1514 | 0.1265 | 0.1482 | 0.1182 |
| wooden table | 18 | 0.0619 | 0.1135 | 0.1017 | 0.0966 | 0.0830 |
| glass window | 17 | 0.0703 | 0.1024 | 0.0369 | 0.0394 | 0.0234 |
| window | 16 | 0.0489 | 0.0753 | 0.0554 | 0.0775 | 0.0489 |
| wooden countertop | 15 | 0.0771 | 0.1867 | 0.1676 | 0.1786 | 0.1560 |
| wooden cabinet | 13 | 0.1062 | 0.0776 | 0.0767 | 0.0721 | 0.0705 |
| stainless steel refrigerator | 11 | 0.3734 | 0.1365 | 0.1364 | 0.1292 | 0.1100 |
| large window | 9 | 0.2026 | 0.1803 | 0.1560 | 0.1915 | 0.1259 |
| large windows | 8 | 0.0227 | 0.0999 | 0.0814 | 0.0948 | 0.0672 |
| metal shelving unit | 7 | 0.0000 | 0.0761 | 0.0442 | 0.0859 | 0.0617 |
| stainless steel sink | 7 | 0.0000 | 0.0024 | 0.0013 | 0.0003 | 0.0013 |

## HD-EPIC — Detailed Metrics

| Config | Mean IoU | Median IoU | Std IoU |
|--------|----------|------------|---------|
| v2 tw=0.0 (image only) | 0.0737 | 0.0610 | 0.0448 |
| v2 tw=0.5 (LERP blend) | 0.1384 | 0.1156 | 0.0840 |
| v2 tw=1.0 (text only) | 0.1085 | 0.0865 | 0.0728 |
| v3 SLERP max=0.5 | 0.1091 | 0.0912 | 0.0696 |
| v3 SLERP max=1.0 | 0.1011 | 0.0737 | 0.0741 |

### HD-EPIC — Per-Category IoU (top 15 by frequency)

| Category | Count | v2 tw=0.0 | v2 tw=0.5 | v2 tw=1.0 | v3 SLERP max=0.5 | v3 SLERP max=1.0 |
|----------|-------| --------- | --------- | --------- | ---------------- | ---------------- |
| wooden countertop | 98 | 0.0715 | 0.2157 | 0.1502 | 0.1660 | 0.1470 |
| dark wooden cabinet | 50 | 0.0722 | 0.0936 | 0.0715 | 0.0680 | 0.0714 |
| wooden table | 41 | 0.0000 | 0.0061 | 0.0047 | 0.0102 | 0.0101 |
| black appliance | 36 | 0.0005 | 0.0178 | 0.0396 | 0.0156 | 0.0254 |
| bookshelf | 31 | 0.0025 | 0.0267 | 0.0162 | 0.0105 | 0.0081 |
| white tiled wall | 30 | 0.0000 | 0.1529 | 0.0911 | 0.0676 | 0.1846 |
| black cable | 26 | 0.0000 | 0.0002 | 0.0082 | 0.0020 | 0.0136 |
| stainless steel stove | 23 | 0.0011 | 0.0885 | 0.0207 | 0.0586 | 0.0704 |
| white refrigerator | 23 | 0.2533 | 0.4372 | 0.3227 | 0.2938 | 0.1855 |
| stainless steel microwave | 22 | 0.3199 | 0.2641 | 0.0832 | 0.1750 | 0.0280 |
| black blender | 19 | 0.0285 | 0.0058 | 0.0185 | 0.0050 | 0.0247 |
| black toaster | 17 | 0.0036 | 0.0177 | 0.0933 | 0.0235 | 0.0351 |
| white door | 17 | 0.0209 | 0.1441 | 0.1165 | 0.3458 | 0.1779 |
| black kettle | 16 | 0.0192 | 0.1009 | 0.1020 | 0.0245 | 0.0587 |
| dark cabinet | 15 | 0.1658 | 0.2589 | 0.2294 | 0.1642 | 0.3013 |

## Adaptive Weight Distribution

The adaptive text weight per segment is computed as `tw = (1 - cos_sim) * max_tw`.

**Observed distributions** (from preprocessing logs):

| Config | Mean tw | Min tw | Max tw | Std tw |
|--------|---------|--------|--------|--------|
| ADT max0.5 | 0.389 | 0.358 | 0.439 | 0.015 |
| ADT max1.0 | 0.779 | 0.717 | 0.877 | 0.033 |
| HDEPIC max0.5 | 0.393 | 0.343 | 0.448 | 0.022 |
| HDEPIC max1.0 | 0.786 | 0.686 | 0.896 | 0.044 |

## Analysis: Why v3 Underperforms v2

**Result**: v3 SLERP + adaptive weights performs **worse** than v2 LERP tw=0.5 across both datasets (ADT: -4.7%, HDEPIC: -21.2% for max0.5; worse for max1.0).

**Root cause — the modality gap defeats adaptive weighting**:

The adaptive weight formula `tw = (1 - cos_sim) * max_tw` was designed to give distinctive
objects (high image-text similarity) less text weight, and generic objects more. However, the
CLIP modality gap makes **all** image-text cosine similarities cluster in a narrow band
(~0.12-0.28), regardless of how distinctive the object is. This means:

1. **Nearly uniform weights**: Per-frame std is only 0.015-0.044. The mechanism barely
   differentiates between a "stainless steel refrigerator" and a "wooden countertop" because
   the gap dominates the similarity signal.

2. **Systematic undershoot**: For max0.5, the effective mean weight is ~0.39 (not 0.50).
   Since v2 showed tw=0.5 is optimal, using less text weight reduces performance.

3. **max1.0 overshoots**: Effective mean weight ~0.78 pushes past the sweet spot into
   territory similar to v2 tw=1.0 (text only), which is known to be worse.

**SLERP vs LERP**: The interpolation method (SLERP vs LERP+renorm) appears to make negligible
difference. The re-normalization step in v2's LERP already projects back to the unit sphere,
and the angular difference between SLERP geodesic and renormalized LERP is small for the
typical image-text angles (~70-80 degrees).

**Per-category exceptions**: A few categories do improve:
- "white door" (HDEPIC): v3 max0.5 = 0.346 vs v2 tw=0.5 = 0.144 (+140%)
- "dark cabinet" (HDEPIC): v3 max1.0 = 0.301 vs v2 tw=0.5 = 0.259 (+16%)

But these gains are outweighed by regressions on high-performing categories like
"white refrigerator" (0.294 vs 0.437, -33%) and "stainless steel microwave" (0.175 vs 0.264, -34%).

## Conclusion

The SLERP + adaptive per-segment weight approach does not improve over the simpler LERP
with fixed tw=0.5. The CLIP modality gap creates a near-constant offset in image-text
cosine similarities that prevents meaningful per-segment adaptation.

**Implications for future work**:
- Adaptive weights require a modality-gap-free embedding space (e.g., Qwen3-VL-Embedding)
  where cosine similarity actually reflects semantic agreement
- Alternatively, gap-reduction techniques (Fill the Gap, spectral alignment) could be applied
  as a preprocessing step before computing adaptive weights
- The fixed tw=0.5 LERP blend in v2 remains the best-performing configuration

---
*Report generated by `generate_v3_report.py`, analysis added manually*
