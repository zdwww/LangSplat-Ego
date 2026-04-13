#!/usr/bin/env python3
"""
Generate unified v2-v6 experiment visualization.

For each object with per-variant vis images, creates a composite showing:
- Header: input RGB with bbox, GT mask, object metadata
- Per-variant rows: relevancy heatmap + error overlay, with variant label and metrics

Output: /mnt/raptor/daiwei/LangSplat-workspace/unified_vis/{dataset}/
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WS = '/mnt/raptor/daiwei/LangSplat-workspace'

# ── Variant definitions ──────────────────────────────────────────────────────

DATASETS = {
    'HDEPIC_P01': {
        'variants': [
            ('v2 tw=0.0 (CLIP img)',   'v2_sam2_tw0.0/HDEPIC_P01'),
            ('v2 tw=0.5 (CLIP LERP)',  'v2_sam2_tw0.5/HDEPIC_P01'),
            ('v2 tw=1.0 (CLIP txt)',   'v2_sam2_tw1.0/HDEPIC_P01'),
            ('v3 SLERP max=0.5',       'v3_slerp_adaptive_max0.5/HDEPIC_P01'),
            ('v3 SLERP max=1.0',       'v3_slerp_adaptive_max1.0/HDEPIC_P01'),
            ('v4 Qwen img-only',       'v4_qwen3vl_image_only/HDEPIC_P01'),
            ('v4 Qwen multimodal',     'v4_qwen3vl_multimodal/HDEPIC_P01'),
            ('v4 Qwen LERP@0.5',      'v4_qwen3vl_lerp_tw0.5/HDEPIC_P01'),
            ('v5 CLIP CB-64',         'v5_clip_codebook/HDEPIC_P01'),
            ('v5 Qwen CB-64',         'v5_qwen_codebook/HDEPIC_P01'),
            ('v6 CLIP CB-128',        'v6_clip_codebook/HDEPIC_P01'),
            ('v6 Qwen CB-128',        'v6_qwen_codebook/HDEPIC_P01'),
        ],
        'gt_masks_dir': f'{WS}/v2_sam2_shared/HDEPIC_novel_masks',
        'rgb_dir': '/mnt/raptor/daiwei/Ego3DVQA-data/HD-EPIC/P01/'
                   'P01-20240202-110250/vlm-data/moved_050/rgb',
    },
    'ADT_seq131': {
        'variants': [
            ('v2 tw=0.0 (CLIP img)',   'v2_sam2_tw0.0/ADT_seq131'),
            ('v2 tw=0.5 (CLIP LERP)',  'v2_sam2_tw0.5/ADT_seq131'),
            ('v2 tw=1.0 (CLIP txt)',   'v2_sam2_tw1.0/ADT_seq131'),
            ('v3 SLERP max=0.5',       'v3_slerp_adaptive_max0.5/ADT_seq131'),
            ('v3 SLERP max=1.0',       'v3_slerp_adaptive_max1.0/ADT_seq131'),
        ],
        'gt_masks_dir': f'{WS}/v2_sam2_shared/ADT_novel_masks',
        'rgb_dir': '/mnt/raptor/daiwei/Ego3DVQA-data/ADT/'
                   'Apartment_release_clean_seq131_M1292/vlm-data/moved_050/rgb',
    },
}

# ── Layout constants ─────────────────────────────────────────────────────────

LABEL_W = 300        # width of text label column
PANEL_W = 500        # width of each image panel (heatmap / overlay)
ROW_H = 420          # height per variant row
HEADER_H = 480       # height of header row
PAD = 8              # padding between elements
TOTAL_W = LABEL_W + PANEL_W * 2 + PAD * 4

# Variant-family background colors (light tints for grouping)
FAMILY_COLORS = {
    'v2': (240, 248, 255),   # alice blue
    'v3': (255, 248, 240),   # linen
    'v4': (240, 255, 240),   # honeydew
    'v5': (255, 240, 245),   # lavender blush
    'v6': (245, 245, 220),   # beige
}

BEST_COLOR = (0, 180, 0)    # green for best IoU
WORST_COLOR = (180, 0, 0)   # red for worst IoU


def get_family(label):
    """Extract variant family (v2/v3/v4/v5/v6) from label."""
    for fam in ('v6', 'v5', 'v4', 'v3', 'v2'):
        if label.startswith(fam):
            return fam
    return 'v2'


def load_font(size):
    """Load a TrueType font with fallback."""
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def crop_panel(img, panel_idx, n_panels=5):
    """Crop the i-th panel from a horizontal multi-panel matplotlib image.

    Panels are approximately equal width; the colorbar on panel 1
    makes it slightly narrower, but the equal-division crop is close enough.
    """
    w = img.width
    pw = w / n_panels
    x0 = int(round(pw * panel_idx))
    x1 = int(round(pw * (panel_idx + 1)))
    return img.crop((x0, 0, x1, img.height))


def resize_to_fit(img, max_w, max_h):
    """Resize image to fit within max_w × max_h, preserving aspect ratio."""
    ratio = min(max_w / img.width, max_h / img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)


def paste_centered(canvas, img, x, y, w, h):
    """Paste img centered within the box (x, y, x+w, y+h) on canvas."""
    ox = x + (w - img.width) // 2
    oy = y + (h - img.height) // 2
    if img.mode == 'RGBA':
        canvas.paste(img, (ox, oy), img)
    else:
        canvas.paste(img, (ox, oy))


def draw_text_block(draw, lines, x, y, font, color=(30, 30, 30), line_spacing=4):
    """Draw multiple lines of text, return total height used."""
    total_h = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lh = bbox[3] - bbox[1]
        draw.text((x, y + total_h), line, fill=color, font=font)
        total_h += lh + line_spacing
    return total_h


def load_obj_metrics(summary_data, frame_id, obj_idx):
    """Get per-object metrics from a loaded summary dict."""
    frame = summary_data.get('frames', {}).get(frame_id, {})
    for obj in frame.get('objects', []):
        if obj['obj_idx'] == obj_idx:
            return obj
    return None


def generate_dataset_vis(dataset_name, cfg):
    """Generate unified visualizations for one dataset."""
    variants = cfg['variants']
    gt_masks_dir = cfg['gt_masks_dir']
    rgb_dir = cfg['rgb_dir']
    n_variants = len(variants)

    out_dir = os.path.join(WS, 'unified_vis', dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load summary JSONs for all variants
    summaries = {}
    for label, ws_rel in variants:
        json_path = os.path.join(WS, ws_rel, 'eval_novel_results',
                                 'eval_novel_summary.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                summaries[label] = json.load(f)

    # Load GT segments.json for bbox/category info
    seg_json = os.path.join(gt_masks_dir, 'segments.json')
    segments = {}
    if os.path.exists(seg_json):
        with open(seg_json) as f:
            segments = json.load(f)

    # Get vis file list from the first variant (intersection across all)
    vis_sets = []
    for label, ws_rel in variants:
        vd = os.path.join(WS, ws_rel, 'eval_novel_results', 'vis')
        if os.path.isdir(vd):
            vis_sets.append(set(os.listdir(vd)))
    common_vis = sorted(set.intersection(*vis_sets)) if vis_sets else []
    common_vis = [f for f in common_vis if f.endswith('.png')]

    print(f"\n{'='*60}")
    print(f"  {dataset_name}: {len(common_vis)} objects, {n_variants} variants")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Fonts
    font_title = load_font(22)
    font_label = load_font(18)
    font_metrics = load_font(15)
    font_small = load_font(13)

    total_h = HEADER_H + n_variants * ROW_H

    for vi, vis_file in enumerate(common_vis):
        # Parse filename: {frame_id}_{obj_idx:03d}_{category}.png
        stem = vis_file[:-4]
        parts = stem.split('_', 2)
        frame_id = parts[0]
        obj_idx = int(parts[1])
        category = parts[2].replace('_', ' ')

        # Collect per-variant metrics and find best/worst IoU
        var_metrics = []
        for label, ws_rel in variants:
            m = load_obj_metrics(summaries.get(label, {}), frame_id, obj_idx)
            var_metrics.append(m)
        ious = [m['iou'] for m in var_metrics if m and 'iou' in m]
        best_iou = max(ious) if ious else 0
        worst_iou = min(ious) if ious else 0

        # ── Create canvas ─────────────────────────────────────────────
        canvas = Image.new('RGB', (TOTAL_W, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # ── Header row ────────────────────────────────────────────────
        # Draw header background
        draw.rectangle([0, 0, TOTAL_W, HEADER_H], fill=(245, 245, 250))

        # Load input RGB
        rgb_path = os.path.join(rgb_dir, f'{frame_id}.jpg')
        header_panel_w = (TOTAL_W - PAD * 3) // 2
        header_panel_h = HEADER_H - 60  # leave room for title text

        if os.path.exists(rgb_path):
            rgb_img = Image.open(rgb_path).convert('RGB')
            # Draw bbox on RGB
            seg_frame = segments.get(frame_id, {})
            for seg in seg_frame.get('segments', []):
                if seg['obj_idx'] == obj_idx:
                    bbox = seg['bbox_abs']
                    # Scale bbox to the resized image dimensions
                    # Draw on original, then resize
                    draw_rgb = ImageDraw.Draw(rgb_img)
                    x1, y1, x2, y2 = bbox
                    draw_rgb.rectangle([x1, y1, x2, y2], outline='lime', width=4)
                    # Add category text near bbox
                    draw_rgb.text((x1 + 4, max(y1 - 22, 2)), category,
                                  fill='lime', font=font_label)
                    break
            rgb_thumb = resize_to_fit(rgb_img, header_panel_w, header_panel_h)
            paste_centered(canvas, rgb_thumb, PAD, 50,
                           header_panel_w, header_panel_h)

        # Load GT mask
        gt_mask_path = os.path.join(gt_masks_dir, 'masks',
                                     f'{frame_id}_{obj_idx:03d}.png')
        if os.path.exists(gt_mask_path):
            gt_img = Image.open(gt_mask_path).convert('L')
            # Make it visible: white object on dark background
            gt_vis = Image.merge('RGB', [gt_img, gt_img, gt_img])
            gt_thumb = resize_to_fit(gt_vis, header_panel_w, header_panel_h)
            paste_centered(canvas, gt_thumb,
                           PAD * 2 + header_panel_w, 50,
                           header_panel_w, header_panel_h)

        # Header title text
        title = f'Frame {frame_id}  |  obj #{obj_idx}  |  "{category}"'
        draw.text((PAD, 8), title, fill=(20, 20, 80), font=font_title)
        # Column labels
        draw.text((PAD + header_panel_w // 2 - 60, HEADER_H - 24),
                   'Input RGB + bbox', fill=(80, 80, 80), font=font_small)
        draw.text((PAD * 2 + header_panel_w + header_panel_w // 2 - 40,
                    HEADER_H - 24),
                   'GT mask', fill=(80, 80, 80), font=font_small)

        # ── Variant rows ──────────────────────────────────────────────
        for ri, ((label, ws_rel), metrics) in enumerate(
                zip(variants, var_metrics)):
            y0 = HEADER_H + ri * ROW_H
            family = get_family(label)
            bg = FAMILY_COLORS.get(family, (255, 255, 255))

            # Row background (alternating shade within family)
            draw.rectangle([0, y0, TOTAL_W, y0 + ROW_H], fill=bg)

            # Thin separator line
            draw.line([0, y0, TOTAL_W, y0], fill=(200, 200, 200), width=1)

            # ── Label column ──────────────────────────────────────────
            lx, ly = PAD, y0 + PAD

            # Variant name
            draw.text((lx, ly), label, fill=(20, 20, 60), font=font_label)
            ly += 28

            if metrics:
                iou = metrics.get('iou', 0)
                ap = metrics.get('ap', 0)
                prec = metrics.get('precision', 0)
                rec = metrics.get('recall', 0)

                # Color-code IoU: green for best, red for worst
                if best_iou > worst_iou:
                    if abs(iou - best_iou) < 1e-6:
                        iou_color = BEST_COLOR
                    elif abs(iou - worst_iou) < 1e-6:
                        iou_color = WORST_COLOR
                    else:
                        iou_color = (60, 60, 60)
                else:
                    iou_color = (60, 60, 60)

                draw.text((lx, ly), f'IoU  {iou:.4f}',
                           fill=iou_color, font=font_metrics)
                ly += 22
                draw.text((lx, ly), f'AP   {ap:.4f}',
                           fill=(60, 60, 60), font=font_metrics)
                ly += 22
                draw.text((lx, ly), f'P    {prec:.3f}',
                           fill=(60, 60, 60), font=font_metrics)
                ly += 20
                draw.text((lx, ly), f'R    {rec:.3f}',
                           fill=(60, 60, 60), font=font_metrics)
                ly += 22

                # FG/BG rel means if available
                fg = metrics.get('rel_mean_fg')
                bg_val = metrics.get('rel_mean_bg')
                if fg is not None and bg_val is not None:
                    draw.text((lx, ly), f'FG   {fg:.3f}',
                               fill=(100, 100, 100), font=font_small)
                    ly += 18
                    draw.text((lx, ly), f'BG   {bg_val:.3f}',
                               fill=(100, 100, 100), font=font_small)
            else:
                draw.text((lx, ly), '(no data)', fill=(150, 150, 150),
                           font=font_metrics)

            # ── Image panels ──────────────────────────────────────────
            vis_path = os.path.join(WS, ws_rel, 'eval_novel_results',
                                     'vis', vis_file)
            if os.path.exists(vis_path):
                vis_img = Image.open(vis_path).convert('RGB')

                # Panel 1: relevancy heatmap (index 1 of 5)
                heatmap = crop_panel(vis_img, 1, 5)
                hm_thumb = resize_to_fit(heatmap, PANEL_W - PAD,
                                          ROW_H - PAD * 2)
                paste_centered(canvas, hm_thumb,
                               LABEL_W + PAD, y0 + PAD,
                               PANEL_W, ROW_H - PAD * 2)

                # Panel 4: error overlay (index 4 of 5)
                overlay = crop_panel(vis_img, 4, 5)
                ov_thumb = resize_to_fit(overlay, PANEL_W - PAD,
                                          ROW_H - PAD * 2)
                paste_centered(canvas, ov_thumb,
                               LABEL_W + PANEL_W + PAD * 2, y0 + PAD,
                               PANEL_W, ROW_H - PAD * 2)

        # ── Column headers for panels ─────────────────────────────────
        # Draw on top of variant rows at the very top of the grid
        col_label_y = HEADER_H + 2
        draw.text((LABEL_W + PANEL_W // 2 - 60, col_label_y),
                   'Relevancy Heatmap', fill=(100, 100, 100), font=font_small)
        draw.text((LABEL_W + PANEL_W + PAD + PANEL_W // 2 - 70, col_label_y),
                   'Error Overlay (TP/FP/FN)', fill=(100, 100, 100),
                   font=font_small)

        # ── Save ──────────────────────────────────────────────────────
        out_path = os.path.join(out_dir, vis_file)
        canvas.save(out_path, quality=92)

        if (vi + 1) % 10 == 0 or vi == 0:
            print(f'  [{vi+1:3d}/{len(common_vis)}] {vis_file}')

    print(f'  Done: {len(common_vis)} images saved to {out_dir}')


def main():
    for ds_name, cfg in DATASETS.items():
        generate_dataset_vis(ds_name, cfg)

    # Print summary
    print(f"\n{'='*60}")
    print("  Unified visualizations complete.")
    print(f"  HDEPIC: {WS}/unified_vis/HDEPIC_P01/")
    print(f"  ADT:    {WS}/unified_vis/ADT_seq131/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
