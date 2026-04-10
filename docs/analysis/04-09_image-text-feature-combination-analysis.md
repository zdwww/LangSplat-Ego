# Analysis: Image-Text Feature Combination in 3D Language Fields

**April 2026** | Comparing our approach against the literature

---

## 1. Our Current Approach

In `preprocess_sam2_ego3dvqa.py`, we combine per-segment CLIP features as:

```python
combined = (1 - tw) * image_features + tw * text_features
combined = combined / combined.norm(dim=-1, keepdim=True)
```

Where:
- **Image features**: CLIP ViT-B-16 image encoder applied to tight-cropped, masked SAM2 segments (zero-out non-mask pixels, pad to square, resize to 224x224)
- **Text features**: CLIP ViT-B-16 text encoder applied to segment category labels (from VLM captions), with optional prefix (e.g., "a photo of a ")
- **tw (text_weight)**: Scalar blending parameter (tested at 0.0, 0.5, 1.0)

Both feature vectors are L2-normalized before blending; the result is L2-renormalized after blending.

### Our Results (Novel-View Segmentation, from eval report)

| Dataset | tw=0.0 | tw=0.5 | tw=1.0 |
|---------|--------|--------|--------|
| ADT Mean IoU | 0.064 | **0.064** | 0.054 |
| HDEPIC Mean IoU | 0.066 | **0.114** | 0.089 |
| ADT Recall | 0.259 | 0.611 | 0.616 |
| HDEPIC Recall | 0.157 | 0.477 | 0.503 |
| ADT Precision | 0.096 | 0.069 | 0.056 |
| HDEPIC Precision | 0.186 | 0.131 | 0.101 |

**Key finding**: tw=0.5 is the best overall, with HDEPIC showing +74% IoU over tw=0.0. Text blending dramatically increases recall (2-3x) at a moderate precision cost. Pure text (tw=1.0) overshoots, degrading precision more than it helps recall.

---

## 2. How Other Methods Handle Features in 3D Language Fields

### 2.1 The Standard: Image-Only CLIP Distillation

The vast majority of methods in this space use **only image-based CLIP features** during training/feature assignment, and text features appear only at query time.

| Method | Year | Venue | Feature Source | Text Used in Training? |
|--------|------|-------|----------------|----------------------|
| LangSplat | 2024 | CVPR | CLIP image encoder on SAM crops (3 hierarchy levels) | **No** |
| LERF | 2023 | ICCV | Multi-scale CLIP image patches + DINO regularization | **No** |
| LEGaussians | 2024 | CVPR | Multi-view CLIP + DINO, quantized | **No** |
| GAGS | 2024 | arXiv | CLIP image encoder on SAM crops, granularity-aware | **No** |
| Dr. Splat | 2025 | CVPR | CLIP image encoder on SAM crops, direct registration | **No** |
| Occam's LGS | 2025 | BMVC | CLIP image features, weighted multi-view aggregation | **No** |
| GOI | 2024 | ACM MM | APE (vision-language) image features, codebook-compressed | **No** |
| Gen-LangSplat | 2025 | arXiv | CLIP image encoder on SAM crops, generalized AE | **No** |
| ConceptFusion | 2023 | RSS | CLIP image features per SAM mask + full image | **No** |
| OpenScene | 2023 | CVPR | LSeg/OpenSeg per-pixel image features, multi-view fused | **No** |
| OpenMask3D | 2023 | NeurIPS | CLIP image encoder on multi-view SAM crops | **No** |
| Segment then Splat | 2025 | arXiv | CLIP image encoder on masked crops, multi-view averaged | **No** |
| Beyond Averages | 2025 | arXiv | CLIP image encoder, multi-view bag of embeddings | **No** |
| **Ours** | 2026 | -- | CLIP image + text encoder, linear blend | **Yes** |

**None of the established methods use text features during the 3D feature construction phase.** The universal paradigm is:
1. Extract 2D image-based CLIP features (from crops, patches, or pixels)
2. Distill/register/aggregate them into 3D
3. At query time, encode the text query with CLIP text encoder and compute similarity against stored 3D features

Our approach is unique in explicitly injecting text features at the training supervision stage.

### 2.2 Original LangSplat (Our Baseline)

LangSplat extracts CLIP image embeddings from SAM-segmented crops at 3 hierarchy levels (subpart/part/whole). These 512D features are compressed to 3D via a scene-specific autoencoder, then used as supervision targets for per-Gaussian language features during 3DGS training. At query time, rendered 3D features are decoded back to 512D and compared against text query embeddings via softmax relevancy.

**Key design**: Purely image-based feature supervision. No text features at all during training.

### 2.3 Dr. Splat (CVPR 2025, State-of-the-Art)

Dr. Splat bypasses rendered feature distillation entirely. It directly assigns CLIP image embeddings to 3D Gaussians via inverse volume rendering weights, aggregating per-view contributions weighted by alpha blending. Uses Product Quantization (trained once on LVIS) for compression.

**Key insight**: Direct assignment avoids the feature corruption introduced by optimizing through a rendering pipeline. Still purely image-based features.

### 2.4 Occam's LGS (BMVC 2025)

A training-free method that lifts 2D CLIP image features to 3D Gaussians via weighted multi-view aggregation derived from forward rendering. Achieves SOTA with 100x speedup.

**Key insight**: The weighted average of multi-view image features, using rendering weights, is sufficient. No compression, no training, no text features.

### 2.5 Segment then Splat (2025)

Assigns a single CLIP image embedding per object (averaged across multi-view masked crops) before reconstruction. All Gaussians in an object share the same embedding.

**Key insight**: Object-level consistency is more important than per-pixel feature variation. Averaging multi-view image features provides view invariance.

### 2.6 Beyond Averages (2025)

Instead of averaging CLIP image features, preserves the full set of per-view embeddings per object as a "bag of embeddings." At query time, text similarity is computed against all embeddings, retaining view-specific details.

**Key insight**: View-averaging loses rare but distinctive attributes. Keeping all embeddings preserves them.

---

## 3. The CLIP Modality Gap Problem

### 3.1 What Is the Modality Gap?

Recent research (Liang et al. 2022; Zhang et al. 2025; Li & Zhang 2025) has established that CLIP's image and text embeddings occupy **distinct, linearly separable regions** of the embedding space. Despite being trained via contrastive learning to align semantically similar image-text pairs, the two modalities form separate clusters on the unit hypersphere.

This gap arises from:
1. **Initialization bias** (cone effect): Separate encoders produce embeddings in narrow, non-overlapping cones
2. **Contrastive loss dynamics**: The uniformity term pushes embeddings apart within modality, while alignment only pulls matching pairs closer -- but not enough to overcome the initialization gap
3. **Architecture**: Dual-stream encoders with no shared parameters encode modality information into the embeddings

### 3.2 Implications for Our Linear Interpolation

Our blending formula `combined = (1-tw) * img + tw * txt` performs **linear interpolation (LERP)** between vectors from two separate regions of the hypersphere. This has two problems:

1. **Off-manifold interpolation**: The interpolated vector lies in the "gap" between the two modality clusters -- a sparsely populated region of the embedding space where CLIP's learned similarity structure may not hold. After L2 renormalization, the result projects back onto the hypersphere but potentially to a region that is neither a valid image embedding nor a valid text embedding.

2. **Semantic dilution**: Research on composed image retrieval (Lin et al., ECCV 2024) shows that "linear interpolation of image and text embeddings often pushes features to a suboptimal region, weakening their semantics and performing worse than using image or text alone."

### 3.3 Why Our Results Still Show Improvement

Despite the theoretical issues, tw=0.5 improves IoU on HDEPIC by 74%. This likely works because:

- **The downstream comparison is text-to-feature**: At query time, we compare the blended features against text embeddings. Adding text features to the training targets moves them closer to the text embedding region, making text queries more likely to activate.
- **Recall-precision tradeoff**: The improvement is primarily in recall (+204% on HDEPIC), not precision (-30%). Text blending makes features more responsive to text queries across the board, including false positives. This is exactly what the modality gap predicts: moving toward the text region increases text-similarity scores globally.
- **SAM2 segments have category labels**: Unlike generic scene methods, our segments come with VLM-generated category names. The text features are semantically meaningful (they name the actual object), which provides a strong prior even if the geometric blending is suboptimal.

---

## 4. Critical Assessment: Did We Do It Right?

### 4.1 What We Did Right

1. **Identified a real problem**: Image-only CLIP features are view-dependent and can fail on novel views. Our results confirm this (tw=0.0 has poor recall on HDEPIC).

2. **Leveraged available metadata**: VLM captions provide per-segment category labels that existing methods don't have. Using this information is sensible.

3. **Empirically validated**: The controlled experiment across 3 text weights x 2 datasets provides clear evidence that text blending helps recall, with tw=0.5 as the sweet spot.

4. **L2 renormalization**: Correct to renormalize after blending, since CLIP similarity operates on the unit hypersphere.

### 4.2 What Could Be Improved

#### Issue 1: Linear Interpolation in the Modality Gap

**Problem**: LERP between image and text embeddings traverses the modality gap, producing vectors in an underrepresented region of the embedding space.

**Alternative -- SLERP**: Spherical Linear Interpolation follows the geodesic on the hypersphere:
```
c = [sin((1-α)θ) / sin(θ)] * img + [sin(αθ) / sin(θ)] * txt
```
where θ = arccos(img · txt). SLERP stays on the manifold and has been shown to outperform LERP for composed image retrieval (Lin et al., ECCV 2024). However, when embeddings are already L2-normalized and the angle is small, SLERP ≈ LERP + renormalization, so the practical difference may be small for our case.

**Recommendation**: Low-effort experiment. Replace LERP with SLERP and compare. Expected impact: small to moderate.

#### Issue 2: Blending at the Wrong Stage

**Problem**: We blend features before autoencoder compression (512D → 3D). The autoencoder was trained on pure image CLIP features. Feeding it blended features that lie in a different region of the CLIP space may cause the autoencoder to produce suboptimal compressions.

**Alternative -- Retrain autoencoder on blended features**: If the autoencoder training data matches the actual features used for supervision, compression fidelity should improve.

**Alternative -- Blend after decoding at query time**: Instead of blending during preprocessing, keep pure image features for 3DGS training, but at query time, adjust the text query embedding to account for expected modality gap. This is similar to the "post-hoc calibration" approach in GR-CLIP and I0T.

**Recommendation**: Verify whether the autoencoder reconstruction quality degrades on blended vs pure image features. If it does, retraining is worthwhile.

#### Issue 3: Uniform Blending Weight

**Problem**: All segments get the same text weight regardless of whether the image feature is reliable. A clearly visible, distinctive object (refrigerator) benefits less from text features than a partially occluded or ambiguous one (white tiled wall).

**Alternative -- Adaptive per-segment weighting**: Weight the text contribution by the inverse of image feature confidence (e.g., CLIP similarity between the crop and its own category text embedding). Objects that already have strong image features need less text help; objects with weak image features benefit more.

**Alternative -- Use image-text similarity as weight**: `tw_i = 1 - cos_sim(img_i, txt_i)`. When image and text agree (high similarity), keep image features; when they disagree (image is ambiguous), lean on text.

**Recommendation**: This could explain why tw=0.5 helps some categories (generic surfaces) but hurts others (distinctive objects like "stainless steel refrigerator" which drops from 0.373 to 0.137 IoU at tw=0.5). Adaptive weighting could capture both benefits.

#### Issue 4: Not Using Multi-View Aggregation

**Problem**: We encode each segment from a single view's crop. Multi-view methods (Dr. Splat, Occam's LGS, Beyond Averages, Segment then Splat) aggregate features across views, naturally producing more view-invariant features without needing text features.

**Alternative**: Aggregate image CLIP features for the same object across multiple views (where it appears), then use that aggregated feature as supervision. This achieves view invariance through the same mechanism that text blending provides, but without leaving the image embedding manifold.

**Recommendation**: High potential impact. This is the approach most modern methods use and avoids the modality gap issue entirely. Could be combined with text blending if VLM labels are available.

#### Issue 5: No Established Method Does This

**Problem**: The absence of text-feature blending in any published method is notable. This could mean (a) it's a novel contribution no one has tried, or (b) the community has implicitly found that image-only features with text querying works well enough.

**Interpretation**: Given the modality gap research, the community's approach is theoretically sound: keep image features in the image embedding manifold for 3D, and let CLIP's contrastive alignment handle cross-modal matching at query time. Our results suggest text blending is a useful hack when image features are unreliable (egocentric views, limited training views, small objects), but it comes with the precision tradeoff documented in our evaluation.

---

## 5. Alternative Approaches Worth Exploring

### 5.1 Multi-View Image Feature Aggregation (Recommended)

Instead of text blending, aggregate image CLIP features across views where each segment's object is visible. This is what Segment then Splat, Beyond Averages, and Dr. Splat do -- and it's the cleanest solution because it stays entirely in the image embedding space.

For our pipeline, this would require:
- Tracking which SAM2 segments correspond to the same 3D object across frames
- Averaging (or keeping a bag of) their image CLIP features
- Using the aggregated feature as supervision

This directly addresses view-dependence without the modality gap issue.

### 5.2 SLERP Instead of LERP (Quick Experiment)

If keeping text blending, switch from linear to spherical interpolation:
```python
theta = torch.acos(torch.clamp((image_features * text_features).sum(dim=-1, keepdim=True), -1, 1))
sin_theta = torch.sin(theta)
combined = (torch.sin((1 - tw) * theta) / sin_theta) * image_features + \
           (torch.sin(tw * theta) / sin_theta) * text_features
```
This stays on the hypersphere and avoids traversing the modality gap.

### 5.3 Text-Conditioned Query Adjustment (No Retraining)

Keep pure image features for 3D training. At query time, instead of a raw text embedding, use a gap-adjusted query:
```python
query = text_embed - modality_gap_offset
```
where `modality_gap_offset` is estimated as the mean difference between paired image and text embeddings. This is the approach of post-hoc calibration methods (Mind the Gap, GR-CLIP, I0T).

### 5.4 Adaptive Per-Segment Text Weight

Compute per-segment text weight based on image feature quality:
```python
sim = F.cosine_similarity(image_features, text_features, dim=-1)
tw_adaptive = (1 - sim).clamp(0, 1) * max_tw  # more text for ambiguous segments
```

### 5.5 Dual-Feature Storage

Store both image and text features per segment (no blending). At query time, compute relevancy against both and take the max:
```python
rel_img = get_relevancy(decoded_feat, text_query, negatives)
rel_txt = get_relevancy(text_feat_stored, text_query, negatives)
rel = max(rel_img, rel_txt)
```
This avoids the blending problem entirely but requires storing and rendering two feature sets.

---

## 6. Verdict

**Did we do it the right way?** Partially.

**What works**: The intuition is correct. Text features provide view-invariant semantic grounding that compensates for unreliable image features in egocentric settings. The empirical results validate this.

**What's suboptimal**: The linear interpolation in CLIP's embedding space is theoretically problematic due to the well-documented modality gap. The uniform blending weight hurts distinctive objects. And the approach is unique in the literature for a reason -- most methods achieve view invariance through multi-view aggregation instead, which avoids the cross-modal blending problem entirely.

**Practical recommendation** (ranked by expected impact):

1. **Multi-view feature aggregation** -- The highest-impact improvement. Aggregate image CLIP features across views for the same 3D object. Stays on the image manifold, achieves view invariance naturally. This is what the state-of-the-art does.

2. **Adaptive per-segment text weight** -- Keep text blending but make the weight segment-dependent based on image-text agreement. Protects distinctive objects while helping ambiguous ones.

3. **SLERP instead of LERP** -- Low-effort swap that's theoretically better for hyperspherical embeddings. Expected to give modest improvement.

4. **Autoencoder consistency check** -- Verify the autoencoder handles blended features well. If not, retrain on blended features or blend after decoding.

The current approach is a reasonable first-pass that demonstrates text features have value in egocentric 3D language fields. The next iteration should incorporate multi-view aggregation as the primary mechanism for view invariance, with text features as an optional supplement for ambiguous segments.

---

## Sources

- [LangSplat: 3D Language Gaussian Splatting (CVPR 2024)](https://langsplat.github.io/)
- [LERF: Language Embedded Radiance Fields (ICCV 2023)](https://www.lerf.io/)
- [LEGaussians: Language Embedded 3D Gaussians (CVPR 2024)](https://buaavrcg.github.io/LEGaussians/)
- [GAGS: Granularity-Aware Feature Distillation (2024)](https://arxiv.org/abs/2412.13654)
- [Dr. Splat: Direct Language Embedding Registration (CVPR 2025)](https://drsplat.github.io/)
- [Occam's LGS: Efficient Language Gaussian Splatting (BMVC 2025)](https://insait-institute.github.io/OccamLGS/)
- [GOI: Find 3D Gaussians of Interest (ACM MM 2024)](https://arxiv.org/abs/2405.17596)
- [Gen-LangSplat: Generalized Feature Compression (2025)](https://arxiv.org/abs/2510.22930)
- [Segment then Splat: Unified 3D Open-Vocabulary Segmentation (2025)](https://arxiv.org/abs/2503.22204)
- [Beyond Averages: Bag of Embeddings (2025)](https://arxiv.org/abs/2509.12938)
- [ConceptFusion: Open-set Multimodal 3D Mapping (RSS 2023)](https://concept-fusion.github.io/)
- [OpenScene: 3D Scene Understanding with Open Vocabularies (CVPR 2023)](https://pengsongyou.github.io/openscene)
- [OpenMask3D: Open-Vocabulary 3D Instance Segmentation (NeurIPS 2023)](https://openmask3d.github.io/)
- [SLERP and Text-Anchoring for Zero-shot Composed Retrieval (ECCV 2024)](https://arxiv.org/abs/2405.00571)
- [Mind the Gap: Modality Gap in CLIP (NeurIPS 2022)](https://github.com/Weixin-Liang/Modality-Gap)
- [Mitigate the Gap: Cross-Modal Alignment in CLIP (ICLR 2025)](https://arxiv.org/abs/2406.17639)
- [The What and Why of Text-Image Modality Gap](https://jina.ai/news/the-what-and-why-of-text-image-modality-gap-in-clip-models/)
- [LEGS: Language-Embedded Gaussian Splats (IROS 2024)](https://autolab.berkeley.edu/assets/publications/media/2024_IROS_LEGS_CR.pdf)
- [LangSplatV2: High-dimensional Language Gaussian Splatting (2025)](https://langsplat-v2.github.io/)
- [3D Vision-Language Gaussian Splatting (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/98ed250b203d1ac6b24bbcf263e3d4a7-Paper-Conference.pdf)

---

## 7. Addendum: Qwen3-VL, SigLIP-2, and Bridging the Modality Gap

*Added April 2026*

Our VLM captions come from Qwen3-VL, which uses a **SigLIP-2 vision encoder** internally. This opens up architectural avenues for bridging the modality gap that go beyond simple interpolation tricks. This section surveys the relevant architectures and identifies concrete alternatives.

---

### 7.1 Qwen3-VL Architecture

Qwen3-VL consists of three modules:

1. **Vision Encoder**: SigLIP-2 ViT (16x16 patches), producing per-patch visual features
2. **DeepStack Fusion**: Fuses features from multiple ViT layers (early low-level + deep semantic), rather than using only the final layer
3. **MLP Projector**: 2-layer MLP that projects fused visual features into the LLM's embedding space (Qwen3 transformer decoder)

Visual tokens are inserted into the text sequence between `<|vision_start|>` and `<|vision_end|>` markers, allowing the LLM's self-attention to jointly attend over visual and text tokens.

**Key insight**: After the MLP projector, visual features live in the **same embedding space as text tokens** within the LLM. This is fundamentally different from CLIP/SigLIP-2's dual-tower architecture where image and text encoders produce features in separate (though contrastively aligned) regions.

### 7.2 SigLIP-2: Same Modality Gap as CLIP

SigLIP-2 is a dual-tower contrastive model (separate image and text encoders), just like CLIP. The key differences are:

- **Sigmoid loss** instead of softmax: Each image-text pair is evaluated independently (`"Does this caption match this image?"`) rather than relative to the batch (`"Is this caption better than the others?"`). This is more efficient and slightly more robust, but does not eliminate the modality gap.
- **Multi-task training**: Adds captioning, self-distillation, and masked patch prediction objectives alongside contrastive alignment. These improve feature quality but the dual-tower structure preserves the modality gap.
- **Theoretical confirmation**: Bangachev et al. (2025) prove that SigLIP's global minimizers produce linearly separable image and text embeddings when N > d+2 (dataset size exceeds embedding dimension), the regime all practical models operate in. The modality gap is a **mathematical property of the loss**, not a training artifact.

**Bottom line**: If we extract SigLIP-2 embeddings directly (as a standalone dual-tower encoder), we face the **exact same modality gap** as with CLIP. Linear interpolation between SigLIP-2 image and text features has the same theoretical problems described in Section 3.

### 7.3 The Real Opportunity: Unified Single-Tower Embeddings

Several recent models process both modalities through a **single architecture**, producing embeddings that are unified by construction rather than contrastively aligned post-hoc. These models largely eliminate the modality gap:

#### 7.3.1 Qwen3-VL-Embedding (Qwen, Jan 2026)

Built on the Qwen3-VL foundation model, this is a dedicated embedding model that maps text, images, videos, and mixed inputs into a **unified 2048-dimensional space** (flexible from 64 to 2048 via Matryoshka Representation Learning).

| Property | Detail |
|----------|--------|
| Architecture | Single-tower (Qwen3-VL backbone) |
| Embedding dim | 64-2048 (configurable) |
| Input types | Text, images, video, mixed |
| Training | Multi-stage: contrastive pre-training + distillation |
| MMEB-V2 score | 73.2 (2B), 77.8 (8B) |

**Critical property**: Both image and text inputs are processed through the same transformer and produce embeddings from the same [EOS] token hidden state. There is **no separate image encoder and text encoder** producing features in different manifold regions. This means:

```python
# Both produce embeddings in the SAME space — no gap
img_embed = model.process([{"image": "photo.jpg"}])   # shape: (1, 2048)
txt_embed = model.process([{"text": "a kitchen"}])     # shape: (1, 2048)
similarity = img_embed @ txt_embed.T  # meaningful cross-modal similarity
```

**Implication for our pipeline**: If we used Qwen3-VL-Embedding to produce both image features (from SAM2 crops) and text features (from category labels), linear interpolation between them would be **geometrically valid** because both vectors come from the same manifold region.

#### 7.3.2 E5-V (Jul 2024)

E5-V uses MLLMs (e.g., LLaVA) to create universal multimodal embeddings by designing prompts that force both image and text inputs through the same semantic compression:

- Image prompt: `"<image>\nSummary above image in one word:"`
- Text prompt: `"<text>\nSummary of the above sentence in one word:"`

The "in one word" instruction forces the model to produce a semantic-meaning-based embedding rather than a modality-preserving one. Visualization shows image and text embeddings become **intermingled** in the same space (organized by semantic content, not modality).

**Key finding**: Single-modality training on text pairs transfers directly to multimodal retrieval because the modality gap is already closed. Training cost is ~95% lower than standard multimodal training.

#### 7.3.3 LLM2CLIP (Microsoft, NeurIPS 2024 Workshop / AAAI 2026 Outstanding Paper)

Rather than replacing CLIP, LLM2CLIP enhances it by replacing CLIP's text encoder with a fine-tuned LLM:

1. **Caption Contrastive Fine-tuning**: Converts LLM from causal to bidirectional attention, trains with SimCSE loss on caption pairs to make LLM outputs discriminative
2. **Lightweight Adaptor**: Linear layers bridge the LLM output to CLIP's visual encoder space
3. **Frozen LLM + trainable adaptor**: Preserves LLM knowledge while aligning to vision

Results: +16.5% improvement on EVA02 retrieval. English-trained model achieves SOTA on Chinese retrieval without Chinese training data.

**Relevance**: LLM2CLIP keeps the CLIP visual encoder (compatible with our pipeline) but produces much richer text representations. It doesn't directly eliminate the modality gap but significantly improves cross-modal alignment quality.

### 7.4 Post-Hoc Gap Reduction Methods

For cases where we want to keep our current CLIP/SigLIP-2 dual-tower features but reduce the gap:

#### 7.4.1 Fill the Gap (Role et al., May 2025)

Proposes two post-processing methods applicable to **any frozen embedding model**:

**Spectral approach**:
1. Compute cross-modal similarity matrix W = XY^T
2. Form bipartite adjacency matrix and compute normalized Laplacian
3. Extract eigenvectors → new shared-space representations
- Reduces ITR/TIR from >400 to ~2-3 (near-perfect gap elimination)
- Recall@20 for cross-modal retrieval: 0% → 68-79%

**Optimal Transport approach**:
- Learns a linear transformation γ that maps both modalities into shared space
- Laplacian regularization preserves within-modality structure
- Trainable on ~5000 pairs, applies to any new embeddings

**Key advantage**: Purely post-hoc. No model retraining. Works on frozen CLIP/SigLIP embeddings.

#### 7.4.2 GR-CLIP / I0T / Mind the Gap

Various methods that compute a **modality offset vector** (mean difference between paired image and text embeddings) and subtract it:

```python
gap_offset = mean(text_embeds) - mean(image_embeds)  # estimated from calibration set
adjusted_img = image_embed + 0.5 * gap_offset  # shift image toward text manifold
# or
adjusted_txt = text_embed - 0.5 * gap_offset  # shift text toward image manifold
```

Simple but effective for retrieval tasks. I0T achieves near-zero modality gap with a single post-hoc standardization step.

### 7.5 Revised Recommendations for Our Pipeline

Given these findings, here are updated approaches ranked by expected impact and feasibility:

#### Option A: Replace CLIP with Qwen3-VL-Embedding (High impact, moderate effort)

Since our captions already come from Qwen3-VL, using Qwen3-VL-Embedding for feature extraction is architecturally coherent:

```python
# Current pipeline (problematic):
img_feat = clip.encode_image(crop)     # lives in image manifold
txt_feat = clip.encode_text(category)  # lives in text manifold
combined = (1-tw)*img_feat + tw*txt_feat  # crosses the gap

# Proposed pipeline (no gap):
img_feat = qwen3vl_embed.process({"image": crop})      # unified space
txt_feat = qwen3vl_embed.process({"text": category})   # same unified space
combined = (1-tw)*img_feat + tw*txt_feat  # valid interpolation on same manifold
```

**Pros**: Eliminates the modality gap entirely. Linear interpolation becomes geometrically valid. The model already understands the visual concepts from Qwen3-VL's training.

**Cons**: Qwen3-VL-Embedding-2B is much larger than CLIP ViT-B-16 (~2B vs ~150M params). Inference is slower. Embedding dimension is 2048 (vs 512), requiring autoencoder retraining. Query-time text encoding also needs Qwen3-VL-Embedding instead of CLIP.

**Compatibility concern**: The entire pipeline (preprocessing, autoencoder, evaluation) currently assumes 512D CLIP embeddings. Switching to 2048D Qwen3-VL embeddings requires changes throughout. The autoencoder compression ratio becomes 2048→3 instead of 512→3, which is significantly more aggressive.

#### Option B: Post-Hoc Gap Reduction on Current CLIP Features (Moderate impact, low effort)

Apply the spectral or optimal transport methods from "Fill the Gap" as a post-processing step:

```python
# After extracting CLIP features as usual:
img_feat = clip.encode_image(crop)   # 512D, image manifold
txt_feat = clip.encode_text(category)  # 512D, text manifold

# Apply learned transformation (trained once on calibration set)
img_aligned = spectral_transform(img_feat)  # shared space
txt_aligned = spectral_transform(txt_feat)  # same shared space

# Now interpolation is valid
combined = (1-tw)*img_aligned + tw*txt_aligned
```

**Pros**: No model changes. Works on frozen CLIP features. Proven to dramatically reduce the gap. Low implementation effort.

**Cons**: Requires a calibration set of paired image-text embeddings (our SAM2 crops + category labels provide this naturally). Adds a transformation step. The spectral method reduces to 60-120 effective dimensions, which may or may not hurt downstream quality.

#### Option C: Use Qwen3-VL's Internal Features Directly (High impact, high effort)

Instead of using a standalone embedding model, extract features from Qwen3-VL's **intermediate layers** (after DeepStack fusion + MLP projector). At this point, visual features already live in the LLM's token space:

```python
# Extract visual tokens from Qwen3-VL's internal representation
visual_tokens = qwen3vl.forward(image, return_visual_tokens=True)
# These are already in the LLM's embedding space
# Text tokens from the same model are in the same space
```

**Pros**: Richest features (DeepStack fuses multiple ViT layers). Same space as text by construction. Leverages Qwen3-VL's full understanding.

**Cons**: Requires modifying Qwen3-VL inference code to extract intermediate features. Hidden dimension is very large (e.g., 3584 for Qwen3-VL-2B). Most complex to integrate.

#### Option D: Keep CLIP but Use SLERP + Adaptive Weights (Low impact, minimal effort)

The simplest improvement to the current pipeline:

```python
# SLERP instead of LERP
theta = torch.acos((img_feat * txt_feat).sum(dim=-1, keepdim=True).clamp(-1, 1))
sin_theta = torch.sin(theta).clamp(min=1e-6)
combined = (torch.sin((1-tw)*theta)/sin_theta)*img_feat + (torch.sin(tw*theta)/sin_theta)*txt_feat

# Adaptive per-segment weight
sim = F.cosine_similarity(img_feat, txt_feat, dim=-1)
tw_adaptive = (1 - sim).clamp(0, 1) * max_tw
```

**Pros**: Minimal code change. Theoretically better. Adaptive weights protect distinctive objects.

**Cons**: Still crosses the modality gap (SLERP follows the geodesic but the gap region is still underrepresented). Marginal improvement expected.

### 7.6 Summary: Architecture-Aware Path Forward

| Approach | Gap Reduction | Effort | Compatibility | Recommended? |
|----------|--------------|--------|---------------|-------------|
| **A. Qwen3-VL-Embedding** | Eliminates | Moderate | Requires pipeline changes (2048D, new AE) | Yes, for next major iteration |
| **B. Post-hoc spectral/OT** | Dramatic | Low | Drop-in on current CLIP features | Yes, try first |
| **C. Qwen3-VL internal** | Eliminates | High | Major pipeline overhaul | Future work |
| **D. SLERP + adaptive** | Marginal | Minimal | Drop-in | Yes, immediate |

**Recommended path**: Start with **D** (immediate, minimal risk), then try **B** (low effort, potentially high reward). If results justify a larger change, pursue **A** in the next pipeline version. **C** is the theoretically ideal approach but requires the most engineering.

The fundamental insight is that the modality gap is not a bug we need to work around -- it's an architectural property of dual-tower contrastive models (CLIP, SigLIP, SigLIP-2). The cleanest solutions either (1) avoid crossing the gap entirely (multi-view aggregation, Section 5.1), (2) use models that don't have the gap (unified single-tower embeddings like Qwen3-VL-Embedding, E5-V), or (3) explicitly correct for it post-hoc (Fill the Gap, I0T).

---

### 7.7 Additional Sources

- [Qwen3-VL-Embedding-2B (Hugging Face)](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Qwen3-VL-Embedding Paper (Jan 2026)](https://arxiv.org/abs/2601.04720)
- [Qwen3-VL Architecture (DeepWiki)](https://deepwiki.com/QwenLM/Qwen3-VL/4.2-model-architecture)
- [SigLIP 2: Multilingual Vision-Language Encoders (Feb 2025)](https://arxiv.org/abs/2502.14786)
- [SigLIP 2 Blog (Hugging Face)](https://huggingface.co/blog/siglip2)
- [Global Minimizers of Sigmoid Contrastive Loss (2025)](https://arxiv.org/abs/2509.18552)
- [Fill the Gap: Quantifying and Reducing the Modality Gap (May 2025)](https://arxiv.org/abs/2505.03703)
- [E5-V: Universal Embeddings with MLLMs (Jul 2024)](https://arxiv.org/abs/2407.12580)
- [LLM2CLIP: LLM-Enhanced CLIP (AAAI 2026 Outstanding Paper)](https://arxiv.org/abs/2411.04997)
- [GaussianVLM: Language-Aligned Gaussian Splats (IEEE RA-L 2025)](https://arxiv.org/abs/2507.00886)
- [LIFT-GS: Cross-Scene Render-Supervised Distillation (2025)](https://arxiv.org/abs/2502.20389)
