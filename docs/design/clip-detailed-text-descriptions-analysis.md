# Detailed vs. Short Text Descriptions for CLIP Feature Blending

**Date:** 2026-04-12
**Query:** Will enriching the render-branch descriptions (e.g. "off-white porcelain cup, blue floral rim" vs "cup") improve or hurt novel-view segmentation in our v5 CLIP CB-64 pipeline?

---

## TL;DR

Switching to detailed descriptions during training while keeping short category queries at eval time will likely produce a **small net negative** on IoU (-1 to -3 points) due to train-test text asymmetry. However, the codebook variants (scored by AP rather than IoU@0.5) may benefit from the richer text signal because detailed descriptions reduce similarity to canonical negatives and better align with image content. The biggest wins require **matching** text detail between training and eval, or using an ensemble of short + detailed descriptions.

---

## Background

Our v5 CLIP CB-64 pipeline blends CLIP features as:

```
stored_feature = normalize(0.5 * CLIP_image(crop) + 0.5 * CLIP_text(category))
```

Currently the `category` for the render branch (training views) is short: "wooden countertop", "white refrigerator", "window". The gt branch of the same VLM captioning system produces rich descriptions like "off-white porcelain cup, blue floral rim, upside-down, chipped edge" but these are not used.

At eval time, queries are the `category` strings from the `moved_050` branch (also short names). The relevancy computation is:

```
rel = softmax(10 * [cos(pixel_feat, query), cos(pixel_feat, neg₁), ..., cos(pixel_feat, neg₄)])[:, 0]
```

The question is whether enriching training-time text improves the quality of the stored 3D features.

## Key Findings

### 1. CLIP Embedding Shift: Short vs. Detailed Text

Measured on OpenCLIP ViT-B-16 (laion2b_s34b_b88k) with our actual categories:

| Short name | Detailed description | cos_sim |
|---|---|---|
| cup | off-white porcelain cup, blue floral rim, upside-down, chipped edge | 0.480 |
| knife | stainless steel knife, black handle, serrated edge, on countertop | 0.684 |
| window | rectangular window, white frame, natural light, partially open | 0.760 |
| wooden countertop | light brown wooden countertop, scratched surface, rectangular | 0.783 |
| glass jar | clear glass jar, black lid, upright, cylindrical | 0.785 |
| cutting board | wooden cutting board, rectangular, light brown, used surface | 0.802 |
| blue ceramic container | blue ceramic container, rainbow pattern, upright | 0.829 |
| white refrigerator | large white refrigerator, stainless steel handle, closed door | 0.856 |

**Pattern:** Already-descriptive category names (2-3 words with modifiers) remain close to their detailed versions (cos ~0.78-0.86). Single-word bare nouns ("cup", "knife") shift much more (cos ~0.48-0.68). Average shift across all categories: **cos = 0.747** (i.e., ~25% of max cosine distance).

### 2. Literature on Detailed Prompts for CLIP Segmentation

| Method | Venue | Finding |
|--------|-------|---------|
| **CuPL** | NeurIPS 2022 | GPT-3-generated descriptions improve zero-shot classification by +1.15pp avg, +6.5pp on fine-grained texture datasets. Ensemble averaging of multiple descriptions outperforms any single prompt. |
| **DisCLIP** | AAAI 2025 | Only *discriminative* descriptions help. 5 carefully selected descriptions outperform 1,842 random ones. Non-discriminative attributes ("upside-down", "no text") add noise. |
| **DVDet** | ICLR 2024 | Fine-grained LLM-generated part descriptors significantly improve open-vocab detection. Models trained with detailed text generalize to simple category queries at test time. |
| **AdaptCLIPZS** | CVPR 2024 | Simply feeding LLM descriptions to CLIP at inference yields negligible gain (+0.2pp). Gains require fine-tuning. Once fine-tuned, generic test prompts work fine. |
| **Query3D** | WACV 2025 | LLM-generated canonical phrases improve 3D language field IoU from 0.18 to 0.32 (+78%) in autonomous driving scenes. |
| **Long-CLIP** | ICLR 2025 | CLIP's effective context is ~20 tokens (not the nominal 77). Beyond 20 tokens, positional embeddings are inadequately trained. Our descriptions (11-16 tokens) are within the effective range. |

**Consensus:** Most open-vocab segmentation methods (LSeg, OVSeg, SAN, CAT-Seg, FC-CLIP, LangSplat, LERF) use simple "a photo of a {class}" or bare category names. Detailed descriptions help primarily in **fine-grained disambiguation** and when the **training process** can internalize the richer signal.

### 3. Train-Test Asymmetry Impact

Simulation with our pipeline's blending formula:

| Config | Mean relevancy (positive query) |
|--------|------|
| img + SHORT text (tw=0.5) — **current** | 0.908 |
| img + DETAILED text (tw=0.5) — **proposed** | 0.870 |
| SHORT text only (tw=1.0) | 0.959 |

**The asymmetry costs ~0.038 relevancy.** With the `softmax(10x)` temperature, this translates to a meaningful shift in the binary mask after thresholding at 0.5.

However, two factors mitigate this:
1. Real image features already encode visual attributes that detailed descriptions capture. The LERP(image, detailed_text) may be more internally coherent.
2. Detailed descriptions have **lower similarity to canonical negatives** ("object", "things", "stuff", "texture"): mean max-negative similarity drops from 0.668 (short) to 0.472 (detailed). This makes the softmax relevancy computation more decisive.

### 4. Critical Observation: Our Categories Already Include Modifiers

Unlike benchmarks where categories are bare nouns ("cup", "knife"), our render-branch categories are already partially descriptive:
- "wooden countertop" (not just "countertop")
- "white refrigerator" (not just "refrigerator")  
- "stainless steel toaster" (not just "toaster")
- "blue electric kettle" (not just "kettle")

These already capture the most discriminative attribute (material/color). Adding further details (surface condition, orientation, text presence) provides diminishing returns and may introduce non-visual noise.

### 5. CLIP as Bag-of-Words Cross-Modally

Recent work (2025) shows CLIP's cross-modal matching behaves like a bag-of-words model — while the text encoder preserves compositional structure internally, the cosine similarity flattens attribute-object bindings. This means detailed descriptions encode useful information *within* text space, but the cross-modal retrieval may not fully leverage compositional attributes like "blue floral rim on the cup" vs "blue cup with no rim."

## Analysis

**For the current pipeline (IoU@0.5 metric, short eval queries), detailed training descriptions will likely hurt slightly:**

1. The train-test mismatch (detailed training text → short eval query) reduces relevancy by ~4%.
2. Our category names already include key discriminative attributes.
3. State attributes in descriptions ("upside-down", "chipped edge") are viewpoint-dependent and may conflict with novel views.
4. Without model fine-tuning (AdaptCLIPZS finding), simply using richer text at feature-extraction time has marginal benefit.

**For AP/ROC-AUC (threshold-free metrics), the effect may be neutral-to-positive:**

The lower similarity to negatives and potentially better image-text coherence could improve the *ordering* of the relevancy map without improving the absolute values at threshold 0.5.

**The bigger opportunity is elsewhere:**

The unified analysis already shows the threshold choice (not the architecture or text) is the dominant confound. A threshold sweep on the current pipeline would likely yield more improvement than text enrichment.

## Recommendations

If you still want to experiment with richer text:

1. **Ensemble approach (lowest risk):** Encode both `category` and `description`, average the two text embeddings before LERP blending. This preserves the category-name signal while adding discriminative detail. CuPL validates this approach.

2. **Match train and eval text:** If you enrich training-time descriptions, also enrich eval-time queries. This eliminates the asymmetry cost entirely.

3. **Filter for discriminative attributes only (DisCLIP insight):** Remove non-visual attributes ("upside-down", "no text visible") and keep only material/color/shape attributes that are view-invariant.

4. **Lower priority than threshold calibration:** The v5 report's recommended next step (threshold sweep at ~0.58 instead of 0.50) is a 5-line change that addresses the dominant confound. Text enrichment is a second-order optimization.

## Caveats

- The relevancy simulation used synthetic image features; real image crops may behave differently because they already encode the visual attributes described in detailed text.
- The literature results are primarily on classification and detection benchmarks, not 3D language fields with codebook decoders.
- No direct ablation exists for LangSplat/LERF with detailed vs. short descriptions — this would need empirical validation.
- The gt branch descriptions were generated by a different VLM pipeline stage (refined=True) than the render branch (refined=False). Quality and consistency of descriptions would need to be validated before use.

## References

1. [Long-CLIP: Unlocking the Long-Text Capability of CLIP (ICLR 2025)](https://arxiv.org/abs/2403.15378)
2. [CuPL: What does a platypus look like? (NeurIPS 2022)](https://arxiv.org/abs/2209.03320)
3. [DisCLIP: Does VLM Classification Benefit from LLM Description Semantics? (AAAI 2025)](https://arxiv.org/abs/2412.11917)
4. [DVDet: LLMs Meet VLMs — Fine-grained Descriptors for Open Vocabulary Detection (ICLR 2024)](https://arxiv.org/abs/2402.04630)
5. [AdaptCLIPZS: Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions (CVPR 2024)](https://arxiv.org/abs/2401.02460)
6. [Query3D: LLM-Powered Open-Vocabulary 3D Scene Segmentation (WACV 2025 Workshop)](https://arxiv.org/abs/2408.03516)
7. [CLIP Behaves like a Bag-of-Words Model Cross-Modally (2025)](https://arxiv.org/abs/2502.03566)
8. [A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP on Dense Captions](https://arxiv.org/abs/2312.08578)
9. [CLIP with Quality Captions: A Strong Pretraining for Vision Tasks (2024)](https://arxiv.org/abs/2405.08911)
10. [VeCLIP: Improving CLIP Training via Visual-enriched Captions (2023)](https://arxiv.org/abs/2310.07699)
