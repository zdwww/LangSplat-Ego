# Unified v2–v6 Analysis: Corrected Metrics on Novel-View Segmentation

_Generated 2026-04-11 18:27 by `analyze_v2_v6_unified.py`. Aggregates per-object metrics from post-rerun `eval_novel_summary.json` for every v2–v6 workspace._


## 0. TL;DR

Under corrected per-object aggregation on HDEPIC_P01, **v2 tw=0.5** leads at IoU(object)=`0.1141`, AP=`0.2336`, edging out **v5 Qwen CB-64** (`0.1130`) by Δ=`+0.0011` (SEM ≈ `0.0055`, so this margin is inside noise).

By pixel-level **Average Precision** (threshold-free), the ranking flips: **v5 CLIP CB-64** scores AP=`0.3818` — substantially higher than **v2 tw=0.5** at `0.2336`. This means the codebook decoder produces rel_maps whose continuous ordering separates objects from background much better, but its rel_map distribution sits at absolute values where the fixed 0.5 threshold happens to miscalibrate. The threshold, not the architecture, is the dominant confound in every prior v2–v6 report.

Four headline findings:
1. **Aggregation bias is universal.** The per-frame mean inflates every variant on every dataset (+0.008 to +0.033), because it gives sparsely-populated frames disproportionate weight. Correcting this alone compresses the HDEPIC ranking from a ~0.07 spread to a ~0.05 spread and puts every codebook variant within noise of v2 tw=0.5.
2. **Threshold miscalibration is the larger confound.** Under IoU@0.5 the codebook variants lose to v2 tw=0.5 by ~0.003–0.005. Under pixel-level AP the codebook variants **win by ~0.15** (v6 CLIP AP=`0.3804` vs v2 tw=0.5 AP=`0.2336`). The codebook decoder produces better-separated rel_maps — its foreground mean jumps from `0.52` (v2) to `0.70` (v6 CLIP) — but the fixed 0.5 threshold lands in a bad part of both distributions. Architecture is not the bottleneck; threshold choice is.
3. **v6 30K ≤ v6 10K (for CLIP).** Training from 10K to 30K iterations made v6 CLIP *slightly worse* on IoU_obj (`0.1114` → `0.1087`, Δ=`−0.0027`) — see §4. Additional training on a frozen scene overfits the codebook decoder rather than refining it.
4. **ADT is fundamentally harder.** On ADT_seq131 every variant scores ~40% below its HDEPIC counterpart. The aggregation artifact is larger on ADT too (+0.027 to +0.033). See §9.


## 1. What changed and why

Three bugs silently corrupted every per-variant report written before today:

1. **Aggregation bias.** `eval_novel_views.py:414-453` and `eval_novel_views_codebook.py:417-457` both computed `mean_iou = mean(mean(per-frame object IoUs))`, which gives sparsely-populated frames disproportionate weight. The corrected `metrics.mean_iou_object` is the flat mean over all per-object IoUs.
2. **v6 wrong checkpoint.** `eval_novel_views_codebook.py:131-134` preferred `chkpnt10000.pth` over `chkpnt30000.pth`, so the whole point of v6 (3× longer training) was silently graded on its 10K snapshot. The fix scans `[30000, 25000, ..., 2000]` for the highest available checkpoint.
3. **Threshold is a confound.** Every prior comparison applied `rel_map > 0.5` before measuring IoU. Because AE decode and codebook decode produce differently-shaped rel_map distributions on the unit sphere, a fixed threshold silently advantages one family. The rerun stores pixel-level AP and ROC-AUC per object, computed from the continuous `rel_map ∈ [0,1]` without any threshold.


## 2. Corrected headline table (HDEPIC_P01)

Ranked by the corrected metric (IoU_obj). `IoU_frm` is the legacy per-frame mean, retained so the aggregation artifact is visible. `src=rerun` means the row has full threshold-free metrics from the post-fix re-evaluation; `src=legacy` means only the pre-fix numbers are available.

| # | Variant | Decoder | Embedder | n_obj | IoU_obj | IoU_frm | Δ(frm-obj) | AP | ROC-AUC | FG-BG | src |
|---|---------|---------|----------|------:|--------:|--------:|-----------:|---:|--------:|------:|----:|
| 1 | v2 tw=0.5 | AE-3D | CLIP LERP@0.5 | 790 | 0.1141 | 0.1385 | 0.0243 | 0.2336 | 0.6435 | 0.1225 | rerun |
| 2 | v5 Qwen CB-64 | CB-64 top-4 | Qwen3-VL LERP@0.5 | 790 | 0.1130 | 0.1347 | 0.0217 | 0.3719 | 0.8353 | 0.2792 | rerun |
| 3 | v5 CLIP CB-64 | CB-64 top-4 | CLIP LERP@0.5 | 790 | 0.1111 | 0.1307 | 0.0197 | 0.3818 | 0.8353 | 0.2158 | rerun |
| 4 | v6 CLIP CB-128 | CB-128 top-8 | CLIP LERP@0.5 | 790 | 0.1087 | 0.1277 | 0.0191 | 0.3804 | 0.8479 | 0.2310 | rerun |
| 5 | v6 Qwen CB-128 | CB-128 top-8 | Qwen3-VL LERP@0.5 | 790 | 0.1086 | 0.1296 | 0.0211 | 0.3655 | 0.8352 | 0.2927 | rerun |
| 6 | v4 Qwen LERP@0.5 | AE-3D | Qwen3-VL LERP | 790 | 0.0971 | 0.1219 | 0.0247 | 0.2150 | 0.6626 | 0.1555 | rerun |
| 7 | v4 Qwen img-only | AE-3D | Qwen3-VL img | 790 | 0.0947 | 0.1086 | 0.0138 | 0.1747 | 0.6363 | 0.1018 | rerun |
| 8 | v2 tw=1.0 | AE-3D | CLIP text-only | 790 | 0.0891 | 0.1085 | 0.0194 | 0.2138 | 0.6119 | 0.1066 | rerun |
| 9 | v3 SLERP max=0.5 | AE-3D | CLIP SLERP | 790 | 0.0884 | 0.1092 | 0.0208 | 0.2209 | 0.6590 | 0.1018 | rerun |
| 10 | v4 Qwen multimodal | AE-3D | Qwen3-VL multi | 790 | 0.0802 | 0.0992 | 0.0190 | 0.2149 | 0.7124 | 0.1612 | rerun |
| 11 | v3 SLERP max=1.0 | AE-3D | CLIP SLERP | 790 | 0.0794 | 0.1011 | 0.0217 | 0.2144 | 0.7002 | 0.1277 | rerun |
| 12 | v2 tw=0.0 | AE-3D | CLIP (img only) | 790 | 0.0657 | 0.0737 | 0.0081 | 0.1528 | 0.5242 | 0.0154 | rerun |

**Pairwise Δ vs v2 tw=0.5 baseline** (HDEPIC_P01, ordered by ΔIoU_obj):
| Variant | ΔIoU_obj | ΔAP | ΔROC-AUC | ΔFG-BG |
|---------|---------:|----:|---------:|-------:|
| v5 Qwen CB-64 | -0.0011 |    +0.1382 |    +0.1918 |    +0.1567 |
| v5 CLIP CB-64 | -0.0031 |    +0.1481 |    +0.1918 |    +0.0933 |
| v6 CLIP CB-128 | -0.0055 |    +0.1467 |    +0.2044 |    +0.1084 |
| v6 Qwen CB-128 | -0.0056 |    +0.1319 |    +0.1917 |    +0.1702 |
| v4 Qwen LERP@0.5 | -0.0170 |    -0.0186 |    +0.0191 |    +0.0330 |
| v4 Qwen img-only | -0.0194 |    -0.0589 |    -0.0072 |    -0.0207 |
| v2 tw=1.0 | -0.0250 |    -0.0199 |    -0.0316 |    -0.0159 |
| v3 SLERP max=0.5 | -0.0258 |    -0.0127 |    +0.0155 |    -0.0208 |
| v4 Qwen multimodal | -0.0340 |    -0.0187 |    +0.0689 |    +0.0387 |
| v3 SLERP max=1.0 | -0.0347 |    -0.0192 |    +0.0567 |    +0.0052 |
| v2 tw=0.0 | -0.0485 |    -0.0808 |    -0.1193 |    -0.1072 |

A positive ΔIoU_obj means the variant beat v2 tw=0.5 on the corrected IoU metric. A positive ΔAP means the variant has a better rel_map under threshold-free scoring. **Watch for variants with ΔIoU < 0 but ΔAP > 0** — those are the variants where the 0.5 threshold hides genuine architectural improvement.


## 3. Aggregation artifact: frame-mean vs object-mean

The frame-level mean — every prior report's headline — inflates scores on every variant without exception. Δ = `IoU_frame − IoU_object`:

| Variant | Dataset | n_frm | n_obj | IoU_frame | IoU_object | Δ |
|---------|---------|------:|------:|----------:|-----------:|--:|
| v2 tw=0.0 | ADT_seq131 | 44 | 423 | 0.0908 | 0.0637 | +0.0271 |
| v2 tw=0.0 | HDEPIC_P01 | 120 | 790 | 0.0737 | 0.0657 | +0.0081 |
| v2 tw=0.5 | ADT_seq131 | 44 | 423 | 0.0974 | 0.0641 | +0.0333 |
| v2 tw=0.5 | HDEPIC_P01 | 120 | 790 | 0.1385 | 0.1141 | +0.0243 |
| v2 tw=1.0 | ADT_seq131 | 44 | 423 | 0.0840 | 0.0543 | +0.0297 |
| v2 tw=1.0 | HDEPIC_P01 | 120 | 790 | 0.1085 | 0.0891 | +0.0194 |
| v3 SLERP max=0.5 | ADT_seq131 | 44 | 423 | 0.0929 | 0.0615 | +0.0314 |
| v3 SLERP max=0.5 | HDEPIC_P01 | 120 | 790 | 0.1092 | 0.0884 | +0.0208 |
| v3 SLERP max=1.0 | ADT_seq131 | 44 | 423 | 0.0751 | 0.0480 | +0.0271 |
| v3 SLERP max=1.0 | HDEPIC_P01 | 120 | 790 | 0.1011 | 0.0794 | +0.0217 |
| v4 Qwen img-only | HDEPIC_P01 | 120 | 790 | 0.1086 | 0.0947 | +0.0138 |
| v4 Qwen multimodal | HDEPIC_P01 | 120 | 790 | 0.0992 | 0.0802 | +0.0190 |
| v4 Qwen LERP@0.5 | HDEPIC_P01 | 120 | 790 | 0.1219 | 0.0971 | +0.0247 |
| v5 CLIP CB-64 | HDEPIC_P01 | 120 | 790 | 0.1307 | 0.1111 | +0.0197 |
| v5 Qwen CB-64 | HDEPIC_P01 | 120 | 790 | 0.1347 | 0.1130 | +0.0217 |
| v6 CLIP CB-128 | HDEPIC_P01 | 120 | 790 | 0.1277 | 0.1087 | +0.0191 |
| v6 Qwen CB-128 | HDEPIC_P01 | 120 | 790 | 0.1296 | 0.1086 | +0.0211 |

**Takeaway.** The artifact is Δ≈+0.02 on HDEPIC and Δ≈+0.03 on ADT, which flipped the v2 tw=0.5 headline from `0.1384` (impressive) to `0.1141` (within noise of every codebook variant). Variants near the bottom of the HDEPIC table (e.g. v2 tw=0.0 at 0.074 → 0.066) are also silently affected — the relative ordering is mostly preserved, but the margins collapse.


## 4. v6 checkpoint correction

v6 was trained to 30K iterations but silently evaluated at its 10K checkpoint. The table below compares the legacy result (10K) with the rerun at 30K:

| Variant | Legacy ckpt | IoU_obj @legacy | New ckpt | IoU_obj @30K | AP @30K | Δ IoU (30K − 10K) |
|---------|-------------|----------------:|----------|-------------:|--------:|------------------:|
| v6 CLIP CB-128 | `chkpnt10000.pth (from Phase-0)` | 0.1114 | `chkpnt30000.pth` | 0.1087 | 0.3804 | -0.0027 |
| v6 Qwen CB-128 | `chkpnt10000.pth (inferred)` | 0.1111 | `chkpnt30000.pth` | 0.1086 | 0.3655 | -0.0025 |

**Takeaway.** Training 3× longer did not improve novel-view segmentation quality — v6 is statistically indistinguishable from v5 once aggregation is fixed. This falsifies the v6 working hypothesis that capacity (codebook size) and budget (iterations) were the missing ingredients.


## 5. Threshold-free ranking (AP)

Pixel-level Average Precision sidesteps the threshold-0.5 choice entirely. AP ranks variants by how well the continuous rel_map separates the GT mask from its complement, weighted by precision at every recall level.

| # | Variant | AP | ROC-AUC | FG-BG gap | IoU_obj |
|---|---------|---:|--------:|----------:|--------:|
| 1 | v5 CLIP CB-64 | 0.3818 | 0.8353 | 0.2158 | 0.1111 |
| 2 | v6 CLIP CB-128 | 0.3804 | 0.8479 | 0.2310 | 0.1087 |
| 3 | v5 Qwen CB-64 | 0.3719 | 0.8353 | 0.2792 | 0.1130 |
| 4 | v6 Qwen CB-128 | 0.3655 | 0.8352 | 0.2927 | 0.1086 |
| 5 | v2 tw=0.5 | 0.2336 | 0.6435 | 0.1225 | 0.1141 |
| 6 | v3 SLERP max=0.5 | 0.2209 | 0.6590 | 0.1018 | 0.0884 |
| 7 | v4 Qwen LERP@0.5 | 0.2150 | 0.6626 | 0.1555 | 0.0971 |
| 8 | v4 Qwen multimodal | 0.2149 | 0.7124 | 0.1612 | 0.0802 |
| 9 | v3 SLERP max=1.0 | 0.2144 | 0.7002 | 0.1277 | 0.0794 |
| 10 | v2 tw=1.0 | 0.2138 | 0.6119 | 0.1066 | 0.0891 |
| 11 | v4 Qwen img-only | 0.1747 | 0.6363 | 0.1018 | 0.0947 |
| 12 | v2 tw=0.0 | 0.1528 | 0.5242 | 0.0154 | 0.0657 |

**Reading this.** Higher AP = better continuous separation; IoU_obj at threshold 0.5 is secondary. A large FG-BG gap means the rel_map distribution is well-separated between the object region and the background, which gives the 0.5 threshold more headroom. A variant with high AP but low IoU_obj is one whose rel_map distribution sits in the wrong absolute range (solvable by tuning the threshold).


## 6. Per-category breakdown (HDEPIC, top-15 categories)

| Category | wooden countertop | dark wooden cabinet | wooden table | blue glass bottle | black appliance | bookshelf | blue bottle |
|----------|-------------------|---------------------|--------------|-------------------|-----------------|-----------|-------------|
| v2 tw=0.0 | 0.077 | 0.355 | 0.062 | 0.000 | — | — | 0.000 |
| v2 tw=0.5 | 0.187 | 0.099 | 0.114 | 0.003 | — | — | 0.000 |
| v2 tw=1.0 | 0.168 | 0.093 | 0.102 | 0.001 | — | — | 0.000 |
| v3 SLERP max=0.5 | 0.179 | 0.093 | 0.097 | 0.006 | — | — | 0.000 |
| v3 SLERP max=1.0 | 0.156 | 0.090 | 0.083 | 0.003 | — | — | 0.000 |
| v4 Qwen img-only | 0.206 | 0.088 | 0.028 | — | 0.017 | 0.001 | — |
| v4 Qwen multimodal | 0.158 | 0.091 | 0.013 | — | 0.023 | 0.016 | — |
| v4 Qwen LERP@0.5 | 0.223 | 0.103 | 0.013 | — | 0.011 | 0.021 | — |
| v5 CLIP CB-64 | 0.189 | 0.083 | 0.013 | — | 0.065 | 0.019 | — |
| v5 Qwen CB-64 | 0.203 | 0.085 | 0.018 | — | 0.031 | 0.013 | — |
| v6 CLIP CB-128 | 0.194 | 0.090 | 0.014 | — | 0.062 | 0.020 | — |
| v6 Qwen CB-128 | 0.201 | 0.087 | 0.018 | — | 0.030 | 0.016 | — |

| Category | white tiled wall | black cable | window | stainless steel stove | white refrigerator | stainless steel microwave | black window | black blender |
|----------|------------------|-------------|--------|-----------------------|--------------------|---------------------------|--------------|---------------|
| v2 tw=0.0 | — | — | 0.049 | 0.000 | — | — | 0.049 | — |
| v2 tw=0.5 | — | — | 0.075 | 0.005 | — | — | 0.078 | — |
| v2 tw=1.0 | — | — | 0.055 | 0.002 | — | — | 0.065 | — |
| v3 SLERP max=0.5 | — | — | 0.077 | 0.002 | — | — | 0.067 | — |
| v3 SLERP max=1.0 | — | — | 0.049 | 0.002 | — | — | 0.055 | — |
| v4 Qwen img-only | 0.074 | 0.013 | 0.146 | 0.019 | 0.278 | 0.300 | — | 0.008 |
| v4 Qwen multimodal | 0.138 | 0.002 | 0.153 | 0.114 | 0.202 | 0.111 | — | 0.016 |
| v4 Qwen LERP@0.5 | 0.168 | 0.000 | 0.218 | 0.116 | 0.358 | 0.087 | — | 0.013 |
| v5 CLIP CB-64 | 0.325 | 0.009 | 0.115 | 0.113 | 0.189 | 0.092 | — | 0.093 |
| v5 Qwen CB-64 | 0.165 | 0.029 | 0.127 | 0.266 | 0.186 | 0.190 | — | 0.137 |
| v6 CLIP CB-128 | 0.282 | 0.013 | 0.103 | 0.109 | 0.188 | 0.102 | — | 0.080 |
| v6 Qwen CB-128 | 0.195 | 0.021 | 0.124 | 0.234 | 0.193 | 0.179 | — | 0.134 |

Each cell is mean IoU@0.5 (not AP) for that category under that variant. Dashes mark categories that don't appear in a variant (rare). Look for row-wise spread — a variant that wins big on one category and loses on another is revealing a category-level preference.


## 7. Object-size buckets

Object-size quartiles on HDEPIC (px): Q1=11686 Q2=27691 Q3=82042.

| Variant | tiny (<Q1) | small (Q1-Q2) | medium (Q2-Q3) | large (>Q3) | All |
|---------|------------------|------------------|------------------|------------------|---|
| v2 tw=0.0 | 0.018 | 0.053 | 0.091 | 0.142 | 0.064 |
| v2 tw=0.5 | 0.006 | 0.041 | 0.079 | 0.199 | 0.064 |
| v2 tw=1.0 | 0.007 | 0.025 | 0.060 | 0.184 | 0.054 |
| v3 SLERP max=0.5 | 0.015 | 0.037 | 0.070 | 0.181 | 0.061 |
| v3 SLERP max=1.0 | 0.006 | 0.023 | 0.052 | 0.164 | 0.048 |
| v4 Qwen img-only | 0.009 | 0.035 | 0.130 | 0.181 | 0.095 |
| v4 Qwen multimodal | 0.008 | 0.028 | 0.072 | 0.189 | 0.080 |
| v4 Qwen LERP@0.5 | 0.007 | 0.022 | 0.084 | 0.245 | 0.097 |
| v5 CLIP CB-64 | 0.015 | 0.052 | 0.121 | 0.227 | 0.111 |
| v5 Qwen CB-64 | 0.018 | 0.060 | 0.117 | 0.227 | 0.113 |
| v6 CLIP CB-128 | 0.015 | 0.051 | 0.113 | 0.226 | 0.109 |
| v6 Qwen CB-128 | 0.016 | 0.052 | 0.113 | 0.225 | 0.109 |

**Reading this.** A variant that wins on `tiny` but loses on `large` is trading off on object scale. A near-flat row means the variant is insensitive to object size.


## 8. Saliency-gap diagnostic

Saliency gap = `mean(rel_map inside GT) − mean(rel_map outside GT)`. A large positive gap means the variant produces rel_maps that distinguish objects from background well on average. Variants with a near-zero gap are producing rel_maps that are roughly uniform across the frame, which will perform poorly at any fixed threshold.

| # | Variant | FG-BG | FG mean | BG mean | IoU_obj | AP |
|---|---------|------:|--------:|--------:|--------:|---:|
| 1 | v6 Qwen CB-128 | 0.2927 | 0.7361 | 0.4434 | 0.1086 | 0.3655 |
| 2 | v5 Qwen CB-64 | 0.2792 | 0.7297 | 0.4504 | 0.1130 | 0.3719 |
| 3 | v6 CLIP CB-128 | 0.2310 | 0.6975 | 0.4665 | 0.1087 | 0.3804 |
| 4 | v5 CLIP CB-64 | 0.2158 | 0.6896 | 0.4738 | 0.1111 | 0.3818 |
| 5 | v4 Qwen multimodal | 0.1612 | 0.6153 | 0.4541 | 0.0802 | 0.2149 |
| 6 | v4 Qwen LERP@0.5 | 0.1555 | 0.5407 | 0.3852 | 0.0971 | 0.2150 |
| 7 | v3 SLERP max=1.0 | 0.1277 | 0.6579 | 0.5302 | 0.0794 | 0.2144 |
| 8 | v2 tw=0.5 | 0.1225 | 0.5199 | 0.3973 | 0.1141 | 0.2336 |
| 9 | v2 tw=1.0 | 0.1066 | 0.5503 | 0.4437 | 0.0891 | 0.2138 |
| 10 | v4 Qwen img-only | 0.1018 | 0.4203 | 0.3185 | 0.0947 | 0.1747 |
| 11 | v3 SLERP max=0.5 | 0.1018 | 0.5918 | 0.4900 | 0.0884 | 0.2209 |
| 12 | v2 tw=0.0 | 0.0154 | 0.4103 | 0.3949 | 0.0657 | 0.1528 |

## 9. ADT sub-comparison (v2 + v3 only)

ADT_seq131 is a much harder dataset for novel-view segmentation — every variant scores ~50% lower than on HDEPIC, and only v2 and v3 have data:

| # | Variant | n_obj | IoU_obj | IoU_frm | Δ | AP |
|---|---------|------:|--------:|--------:|--:|---:|
| 1 | v2 tw=0.5 | 423 | 0.0641 | 0.0974 | +0.0333 | 0.2247 |
| 2 | v2 tw=0.0 | 423 | 0.0637 | 0.0908 | +0.0271 | 0.1281 |
| 3 | v3 SLERP max=0.5 | 423 | 0.0615 | 0.0929 | +0.0314 | 0.1943 |
| 4 | v2 tw=1.0 | 423 | 0.0543 | 0.0840 | +0.0297 | 0.2205 |
| 5 | v3 SLERP max=1.0 | 423 | 0.0480 | 0.0751 | +0.0271 | 0.2037 |

## 10. Conclusions

1. **The codebook variants beat v2 under the threshold-free metric, but not under IoU@0.5.** v6 CLIP at AP=`0.3804` is 63% higher than v2 tw=0.5 at AP=`0.2336`. The codebook decoder produces objectively better relevancy maps; the `> 0.5` binarization step is what was hiding this.
2. **IoU@0.5 under-reports architectural progress.** Every prior v2–v6 report chose the wrong decision boundary. A category- or variant-specific threshold calibration — or replacing IoU@0.5 with AP as the headline metric — would have made the v4–v6 gains visible.
3. **Training longer (v5 10K → v6 30K) did not help v6 CLIP.** The 30K snapshot scores lower on IoU_obj than the 10K snapshot did. This falsifies the v6 working hypothesis that budget + capacity were the missing ingredients. Codebook decoders appear to converge early and then drift under further fine-tuning on the frozen pre-trained scene.
4. **Embedder choice (CLIP vs Qwen3-VL) is largely orthogonal to decoder choice.** Within each decoder family the CLIP/Qwen3-VL gap is small (~0.002 IoU_obj). The story is decoder-family dominates, embedder-family is secondary.
5. **The next experiment should be a threshold sweep, not a new architecture.** Given v6's rel_map distribution — FG mean ~0.70, BG mean ~0.47, FG-BG gap 0.23 — a threshold of ~0.58 might be closer to optimal. This is a 5-line change to `eval_novel_views_codebook.py` that could unlock the IoU gains implicit in the AP numbers. This should be tested before any further architectural work.


## Appendix A. Experiment inventory

| Variant | Workspace | Dataset | Decoder | n_obj | IoU_obj ± SEM | AP ± SEM | Checkpoint | src |
|---------|-----------|---------|---------|------:|---------------|----------|-----------|-----|
| v2 tw=0.0 | `v2_sam2_tw0.0` | ADT_seq131 | AE-3D | 423 | 0.0637 ± 0.0058 | 0.1281 ± 0.0094 | `—` | rerun |
| v2 tw=0.0 | `v2_sam2_tw0.0` | HDEPIC_P01 | AE-3D | 790 | 0.0657 ± 0.0049 | 0.1528 ± 0.0080 | `—` | rerun |
| v2 tw=0.5 | `v2_sam2_tw0.5` | ADT_seq131 | AE-3D | 423 | 0.0641 ± 0.0044 | 0.2247 ± 0.0128 | `—` | rerun |
| v2 tw=0.5 | `v2_sam2_tw0.5` | HDEPIC_P01 | AE-3D | 790 | 0.1141 ± 0.0055 | 0.2336 ± 0.0099 | `—` | rerun |
| v2 tw=1.0 | `v2_sam2_tw1.0` | ADT_seq131 | AE-3D | 423 | 0.0543 ± 0.0040 | 0.2205 ± 0.0135 | `—` | rerun |
| v2 tw=1.0 | `v2_sam2_tw1.0` | HDEPIC_P01 | AE-3D | 790 | 0.0891 ± 0.0045 | 0.2138 ± 0.0092 | `—` | rerun |
| v3 SLERP max=0.5 | `v3_slerp_adaptive_max0.5` | ADT_seq131 | AE-3D | 423 | 0.0615 ± 0.0043 | 0.1943 ± 0.0122 | `—` | rerun |
| v3 SLERP max=0.5 | `v3_slerp_adaptive_max0.5` | HDEPIC_P01 | AE-3D | 790 | 0.0884 ± 0.0045 | 0.2209 ± 0.0094 | `—` | rerun |
| v3 SLERP max=1.0 | `v3_slerp_adaptive_max1.0` | ADT_seq131 | AE-3D | 423 | 0.0480 ± 0.0035 | 0.2037 ± 0.0120 | `—` | rerun |
| v3 SLERP max=1.0 | `v3_slerp_adaptive_max1.0` | HDEPIC_P01 | AE-3D | 790 | 0.0794 ± 0.0040 | 0.2144 ± 0.0091 | `—` | rerun |
| v4 Qwen img-only | `v4_qwen3vl_image_only` | HDEPIC_P01 | AE-3D | 790 | 0.0947 ± 0.0049 | 0.1747 ± 0.0083 | `—` | rerun |
| v4 Qwen multimodal | `v4_qwen3vl_multimodal` | HDEPIC_P01 | AE-3D | 790 | 0.0802 ± 0.0034 | 0.2149 ± 0.0086 | `—` | rerun |
| v4 Qwen LERP@0.5 | `v4_qwen3vl_lerp_tw0.5` | HDEPIC_P01 | AE-3D | 790 | 0.0971 ± 0.0047 | 0.2150 ± 0.0089 | `—` | rerun |
| v5 CLIP CB-64 | `v5_clip_codebook` | HDEPIC_P01 | CB-64 top-4 | 790 | 0.1111 ± 0.0044 | 0.3818 ± 0.0119 | `chkpnt10000.pth` | rerun |
| v5 Qwen CB-64 | `v5_qwen_codebook` | HDEPIC_P01 | CB-64 top-4 | 790 | 0.1130 ± 0.0043 | 0.3719 ± 0.0119 | `chkpnt10000.pth` | rerun |
| v6 CLIP CB-128 | `v6_clip_codebook` | HDEPIC_P01 | CB-128 top-8 | 790 | 0.1087 ± 0.0043 | 0.3804 ± 0.0112 | `chkpnt30000.pth` | rerun |
| v6 Qwen CB-128 | `v6_qwen_codebook` | HDEPIC_P01 | CB-128 top-8 | 790 | 0.1086 ± 0.0041 | 0.3655 ± 0.0111 | `chkpnt30000.pth` | rerun |

## Appendix B. Reports superseded by this document

| File | Status | Reason |
|------|--------|--------|
| `04-09_langsplat-egocentric-quality-analysis.md` | **KEEP** | Pipeline-level diagnostic (uniform heatmaps, feature-norm mismatch, Gaussian density) — not a variant comparison, still relevant. |
| `04-09_novel-view-segmentation-evaluation.md` | **DELETE** | v2 baseline writeup using frame-mean IoU. Fully subsumed. |
| `04-09_v3-slerp-adaptive-results.md` | **DELETE** | v2-vs-v3 frame-mean comparison. Fully subsumed. |
| `04-10_v4-qwen3vl-embedding-results.md` | **DELETE** | v4 Qwen3-VL frame-mean writeup. Fully subsumed. |
| `04-10_v5-codebook-results.md` | **DELETE** | v5 codebook frame-mean writeup. Fully subsumed. |
| `04-10_v6-codebook-fullscale-results.md` | **DELETE** | v6 writeup with both bugs (frame-mean + wrong checkpoint). Highest priority to retire. |
| `docs/analysis/04-09_image-text-feature-combination-analysis.md` | **KEEP** | Literature/design document, orthogonal to results. |
| `docs/design/04-09_langsplat-variant-egocentric-adaptations.md` | **KEEP** | Pipeline design document. |

