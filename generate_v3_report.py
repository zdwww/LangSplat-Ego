"""Generate comparison report for LangSplat v3 (SLERP + adaptive) vs v2 baselines.

Collects eval_novel_summary.json from v2 and v3 configs, produces a markdown
comparison table with per-category breakdown.
"""
import os
import json
import argparse
from datetime import datetime


def load_novel_summary(workspace):
    """Load eval_novel_summary.json from a workspace."""
    path = os.path.join(workspace, 'eval_novel_results', 'eval_novel_summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ws_base', type=str, default='/mnt/raptor/daiwei/LangSplat-workspace')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    ws_base = args.ws_base
    output = args.output or os.path.join(
        '/home/daiwei/Ego3DVQA-GS/LangSplat/docs/experiments',
        '04-09_v3-slerp-adaptive-results.md')
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Define configs: (label, workspace_subdir)
    configs = [
        ('v2 tw=0.0 (image only)', 'v2_sam2_tw0.0'),
        ('v2 tw=0.5 (LERP blend)', 'v2_sam2_tw0.5'),
        ('v2 tw=1.0 (text only)', 'v2_sam2_tw1.0'),
        ('v3 SLERP max=0.5', 'v3_slerp_adaptive_max0.5'),
        ('v3 SLERP max=1.0', 'v3_slerp_adaptive_max1.0'),
    ]
    datasets = [
        ('ADT', 'ADT_seq131'),
        ('HD-EPIC', 'HDEPIC_P01'),
    ]

    lines = []
    lines.append('# LangSplat v3: SLERP + Adaptive Per-Segment Weights')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append('## Method')
    lines.append('**v2 baseline**: Linear interpolation (LERP) with fixed text_weight across all segments.')
    lines.append('**v3 (this experiment)**: Spherical linear interpolation (SLERP) with adaptive per-segment')
    lines.append('text weight based on image-text cosine similarity:\n')
    lines.append('```')
    lines.append('sim = cosine_similarity(image_feat, text_feat)')
    lines.append('tw_adaptive = (1 - sim) * max_tw   # high sim → low text weight')
    lines.append('combined = SLERP(image_feat, text_feat, tw_adaptive)')
    lines.append('```\n')

    # Collect all results
    all_results = {}
    for ds_label, ds_dir in datasets:
        all_results[ds_label] = {}
        for config_label, config_dir in configs:
            ws = os.path.join(ws_base, config_dir, ds_dir)
            summary = load_novel_summary(ws)
            all_results[ds_label][config_label] = summary

    # Find best v2 per dataset
    v2_labels = [c[0] for c in configs[:3]]

    lines.append('## Novel-View Segmentation Results (Mean IoU)\n')
    # Summary table
    lines.append('| Config | ADT IoU | ADT delta | HDEPIC IoU | HDEPIC delta |')
    lines.append('|--------|---------|-----------|------------|--------------|')

    best_v2 = {}
    for ds_label, _ in datasets:
        best = 0
        for vl in v2_labels:
            s = all_results[ds_label].get(vl)
            if s and s['metrics']['mean_iou'] > best:
                best = s['metrics']['mean_iou']
        best_v2[ds_label] = best

    for config_label, _ in configs:
        row = f'| {config_label} |'
        for ds_label, _ in datasets:
            s = all_results[ds_label].get(config_label)
            if s:
                iou = s['metrics']['mean_iou']
                if config_label in v2_labels:
                    row += f' {iou:.4f} | — |'
                else:
                    bv2 = best_v2[ds_label]
                    if bv2 > 0:
                        delta = (iou - bv2) / bv2 * 100
                        sign = '+' if delta >= 0 else ''
                        row += f' {iou:.4f} | {sign}{delta:.1f}% |'
                    else:
                        row += f' {iou:.4f} | N/A |'
            else:
                row += ' N/A | N/A |'
        lines.append(row)

    # Detailed per-dataset tables
    for ds_label, ds_dir in datasets:
        lines.append(f'\n## {ds_label} — Detailed Metrics\n')
        lines.append('| Config | Mean IoU | Median IoU | Std IoU |')
        lines.append('|--------|----------|------------|---------|')

        for config_label, _ in configs:
            s = all_results[ds_label].get(config_label)
            if s:
                m = s['metrics']
                lines.append(f'| {config_label} | {m["mean_iou"]:.4f} | '
                             f'{m["median_iou"]:.4f} | {m["std_iou"]:.4f} |')
            else:
                lines.append(f'| {config_label} | N/A | N/A | N/A |')

        # Per-category comparison (top 10 categories by count)
        lines.append(f'\n### {ds_label} — Per-Category IoU (top 15 by frequency)\n')

        # Get categories from v2 tw=0.5 (reference)
        ref = all_results[ds_label].get('v2 tw=0.5 (LERP blend)')
        if ref and 'per_category' in ref['metrics']:
            cats = ref['metrics']['per_category']
            sorted_cats = sorted(cats.items(), key=lambda x: x[1]['count'], reverse=True)[:15]

            header = '| Category | Count |'
            sep = '|----------|-------|'
            for cl, _ in configs:
                short = cl.split('(')[0].strip() if '(' in cl else cl
                header += f' {short} |'
                sep += f' {"-"*len(short)} |'
            lines.append(header)
            lines.append(sep)

            for cat, info in sorted_cats:
                row = f'| {cat} | {info["count"]} |'
                for config_label, _ in configs:
                    s = all_results[ds_label].get(config_label)
                    if s and cat in s['metrics'].get('per_category', {}):
                        ciou = s['metrics']['per_category'][cat]['mean_iou']
                        row += f' {ciou:.4f} |'
                    else:
                        row += ' N/A |'
                lines.append(row)

    # Adaptive weight statistics (from logs if available)
    lines.append('\n## Adaptive Weight Distribution\n')
    lines.append('The adaptive text weight per segment is computed as `tw = (1 - cos_sim) * max_tw`.')
    lines.append('Check `v3_logs/stage2_*.log` for per-frame distribution statistics.\n')

    lines.append('---')
    lines.append('*Report generated by `generate_v3_report.py`*')

    report = '\n'.join(lines) + '\n'
    with open(output, 'w') as f:
        f.write(report)
    print(f'Report saved to {output}')
    print(report)


if __name__ == '__main__':
    main()
