"""Generate v6 comparison report: full-scale codebook vs v2/v4/v5 baselines."""
import os
import json
import argparse
from datetime import datetime


def load_novel_summary(workspace):
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
        '04-10_v6-codebook-fullscale-results.md')
    os.makedirs(os.path.dirname(output), exist_ok=True)

    configs = [
        ('v2 tw=0.5 (CLIP AE, 30K iters, res 1)', 'v2_sam2_tw0.5'),
        ('v4 lerp tw=0.5 (Qwen3-VL AE, 30K iters, res 1)', 'v4_qwen3vl_lerp_tw0.5'),
        ('v5 CLIP codebook (64/top4, 10K iters, res 2)', 'v5_clip_codebook'),
        ('v5 Qwen3-VL codebook (64/top4, 10K iters, res 2)', 'v5_qwen_codebook'),
        ('v6 CLIP codebook (128/top8, 30K iters, res 2)', 'v6_clip_codebook'),
        ('v6 Qwen3-VL codebook (128/top8, 30K iters, res 2)', 'v6_qwen_codebook'),
    ]
    ds_dir = 'HDEPIC_P01'

    lines = []
    lines.append('# LangSplat v6: Full-Scale Codebook Experiment')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append('## Method')
    lines.append('**v6 changes from v5**:')
    lines.append('- 30K iterations instead of 10K (matches v2 training budget)')
    lines.append('- Codebook size 128 instead of 64 (expanded multi-modal capacity)')
    lines.append('- topk 8 instead of 4 (finer sparse representation)')
    lines.append('- Resolution kept at 2 (half-res). Attempted res=1 but OOMs at 24GB VRAM.')
    lines.append('')
    lines.append('**Motivation**: v5 analysis showed codebook approach loses on large surfaces')
    lines.append('due to (1) insufficient training (10K vs 30K) and (2) top-4 sparsity limiting')
    lines.append('multi-modal representation on textured/multi-mode surfaces.\n')

    all_results = {}
    for config_label, config_dir in configs:
        ws = os.path.join(ws_base, config_dir, ds_dir)
        all_results[config_label] = load_novel_summary(ws)

    v2_label = 'v2 tw=0.5 (CLIP AE, 30K iters, res 1)'
    best_v2 = 0
    s = all_results.get(v2_label)
    if s:
        best_v2 = s['metrics']['mean_iou']

    lines.append('## Novel-View Segmentation Results (HD-EPIC, Mean IoU)\n')
    lines.append('| Config | Mean IoU | Delta vs v2 |')
    lines.append('|--------|----------|-------------|')
    for config_label, _ in configs:
        s = all_results.get(config_label)
        if s:
            iou = s['metrics']['mean_iou']
            if config_label == v2_label:
                lines.append(f'| {config_label} | {iou:.4f} | -- |')
            else:
                if best_v2 > 0:
                    delta = (iou - best_v2) / best_v2 * 100
                    sign = '+' if delta >= 0 else ''
                    lines.append(f'| {config_label} | {iou:.4f} | {sign}{delta:.1f}% |')
                else:
                    lines.append(f'| {config_label} | {iou:.4f} | N/A |')
        else:
            lines.append(f'| {config_label} | N/A | N/A |')

    lines.append('\n## HD-EPIC -- Detailed Metrics\n')
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
    lines.append('\n### HD-EPIC -- Per-Category IoU (top 15 by frequency)\n')
    ref = all_results.get(v2_label)
    if ref and 'per_category' in ref['metrics']:
        cats = ref['metrics']['per_category']
        sorted_cats = sorted(cats.items(), key=lambda x: x[1]['count'], reverse=True)[:15]

        # Short labels for header
        short_labels = ['v2 CLIP AE', 'v4 Qwen AE', 'v5 CLIP cb', 'v5 Qwen cb', 'v6 CLIP cb', 'v6 Qwen cb']
        header = '| Category | Count |' + ''.join(f' {sl} |' for sl in short_labels)
        sep = '|----------|-------|' + ''.join(f' {"-"*len(sl)} |' for sl in short_labels)
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
    lines.append('*Report generated by `generate_v6_report.py`*')

    report = '\n'.join(lines) + '\n'
    with open(output, 'w') as f:
        f.write(report)
    print(f'Report saved to {output}')
    print(report)


if __name__ == '__main__':
    main()
