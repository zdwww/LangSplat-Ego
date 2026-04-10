"""Generate comparison report for LangSplat v4 (Qwen3-VL-Embedding) vs v2 baselines.

Collects eval_novel_summary.json from v2 and v4 configs, produces a markdown
comparison table with per-category breakdown. HD-EPIC only.
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
        '04-10_v4-qwen3vl-embedding-results.md')
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Define configs: (label, workspace_subdir)
    configs = [
        ('v2 tw=0.0 (CLIP image only)', 'v2_sam2_tw0.0'),
        ('v2 tw=0.5 (CLIP LERP blend)', 'v2_sam2_tw0.5'),
        ('v4 image_only (Qwen3-VL)', 'v4_qwen3vl_image_only'),
        ('v4 multimodal (Qwen3-VL)', 'v4_qwen3vl_multimodal'),
        ('v4 lerp tw=0.5 (Qwen3-VL)', 'v4_qwen3vl_lerp_tw0.5'),
    ]
    ds_dir = 'HDEPIC_P01'

    lines = []
    lines.append('# LangSplat v4: Qwen3-VL-Embedding (Unified Multimodal)')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append('## Method')
    lines.append('**v2 baseline**: OpenCLIP ViT-B-16, LERP blend of separate image + text embeddings.')
    lines.append('**v4 (this experiment)**: Qwen3-VL-Embedding-2B, single-tower unified embeddings.')
    lines.append('- `image_only`: SAM2 crop encoded as image through Qwen3-VL-Embedding')
    lines.append('- `multimodal`: SAM2 crop + category text encoded together (unified fusion)')
    lines.append('- 512D via Matryoshka truncation (from 2048D native)\n')

    # Collect results
    all_results = {}
    for config_label, config_dir in configs:
        ws = os.path.join(ws_base, config_dir, ds_dir)
        all_results[config_label] = load_novel_summary(ws)

    # Find best v2
    v2_labels = [c[0] for c in configs[:2]]
    best_v2 = 0
    for vl in v2_labels:
        s = all_results.get(vl)
        if s and s['metrics']['mean_iou'] > best_v2:
            best_v2 = s['metrics']['mean_iou']

    # Summary table
    lines.append('## Novel-View Segmentation Results (HD-EPIC, Mean IoU)\n')
    lines.append('| Config | Mean IoU | Delta vs best v2 |')
    lines.append('|--------|----------|-------------------|')

    for config_label, _ in configs:
        s = all_results.get(config_label)
        if s:
            iou = s['metrics']['mean_iou']
            if config_label in v2_labels:
                lines.append(f'| {config_label} | {iou:.4f} | — |')
            else:
                if best_v2 > 0:
                    delta = (iou - best_v2) / best_v2 * 100
                    sign = '+' if delta >= 0 else ''
                    lines.append(f'| {config_label} | {iou:.4f} | {sign}{delta:.1f}% |')
                else:
                    lines.append(f'| {config_label} | {iou:.4f} | N/A |')
        else:
            lines.append(f'| {config_label} | N/A | N/A |')

    # Detailed metrics
    lines.append('\n## HD-EPIC — Detailed Metrics\n')
    lines.append('| Config | Mean IoU | Median IoU | Std IoU |')
    lines.append('|--------|----------|------------|---------|')

    for config_label, _ in configs:
        s = all_results.get(config_label)
        if s:
            m = s['metrics']
            lines.append(f'| {config_label} | {m["mean_iou"]:.4f} | '
                         f'{m["median_iou"]:.4f} | {m["std_iou"]:.4f} |')
        else:
            lines.append(f'| {config_label} | N/A | N/A | N/A |')

    # Per-category comparison (top 15)
    lines.append('\n### HD-EPIC — Per-Category IoU (top 15 by frequency)\n')

    ref = all_results.get('v2 tw=0.5 (CLIP LERP blend)')
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
                s = all_results.get(config_label)
                if s and cat in s['metrics'].get('per_category', {}):
                    ciou = s['metrics']['per_category'][cat]['mean_iou']
                    row += f' {ciou:.4f} |'
                else:
                    row += ' N/A |'
            lines.append(row)

    lines.append('\n---')
    lines.append('*Report generated by `generate_v4_report.py`*')

    report = '\n'.join(lines) + '\n'
    with open(output, 'w') as f:
        f.write(report)
    print(f'Report saved to {output}')
    print(report)


if __name__ == '__main__':
    main()
