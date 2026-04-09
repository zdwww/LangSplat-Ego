"""Generate comparison report for LangSplat v2 experiments.

Collects eval_summary.json from all v2 configs and v1 baseline,
produces a markdown comparison table.
"""
import os
import json
import argparse
from datetime import datetime


def load_eval_summary(workspace):
    """Load eval_summary.json from a workspace."""
    path = os.path.join(workspace, 'eval_results', 'eval_summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ws_base', type=str, default='/mnt/raptor/daiwei/LangSplat-workspace',
                        help='Base workspace directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path (default: ws_base/v2_report.md)')
    args = parser.parse_args()

    ws_base = args.ws_base
    output = args.output or os.path.join(ws_base, 'v2_report.md')

    # Define all experiment configs
    configs = [
        ('v1_og (baseline)', 'v1_og'),
        ('v2 tw=0.0 (image only)', 'v2_sam2_tw0.0'),
        ('v2 tw=0.5 (blended)', 'v2_sam2_tw0.5'),
        ('v2 tw=1.0 (text only)', 'v2_sam2_tw1.0'),
    ]
    datasets = [
        ('ADT', 'ADT_seq131'),
        ('HD-EPIC', 'HDEPIC_P01'),
    ]

    lines = []
    lines.append(f"# LangSplat v2 Experiment Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Changes from v1")
    lines.append("- **SAM2** box-prompted segmentation (from VLM caption bboxes) replaces SAM automatic masks")
    lines.append("- **Text Feature Blending**: CLIP image + text features at configurable weights")
    lines.append("- **Frame selection**: render-mode frames from captions.json (not covisibility+blur)")
    lines.append("- **Full resolution**: no 2x downscale")
    lines.append("- **Single feature level**: SAM2 produces 1 level (replicated 4x)\n")

    for ds_label, ds_dir in datasets:
        lines.append(f"## {ds_label}\n")

        # Collect all prompts across configs
        all_prompts = set()
        results = {}
        for config_label, config_dir in configs:
            ws = os.path.join(ws_base, config_dir, ds_dir)
            summary = load_eval_summary(ws)
            results[config_label] = summary
            if summary and 'prompts' in summary:
                all_prompts.update(summary['prompts'].keys())

        if not all_prompts:
            lines.append("*No evaluation results found.*\n")
            continue

        prompts = sorted(all_prompts)

        # Header
        header = f"| {'Query':<25} |"
        separator = f"| {'-'*25} |"
        for config_label, _ in configs:
            short = config_label.split('(')[0].strip()
            header += f" {short:<18} |"
            separator += f" {'-'*18} |"
        lines.append(header)
        lines.append(separator)

        # Rows
        for prompt in prompts:
            row = f"| {prompt:<25} |"
            for config_label, _ in configs:
                summary = results.get(config_label)
                if summary and 'prompts' in summary and prompt in summary['prompts']:
                    ps = summary['prompts'][prompt]
                    bl = ps.get('best_level', '?')
                    mean = ps.get('best_mean', 0)
                    row += f" L{bl} {mean:.4f}       |"
                else:
                    row += f" {'N/A':^18} |"
            lines.append(row)

        # Summary row (average)
        lines.append(separator)
        row = f"| {'**Average**':<25} |"
        for config_label, _ in configs:
            summary = results.get(config_label)
            if summary and 'prompts' in summary:
                means = []
                for p in prompts:
                    if p in summary['prompts']:
                        means.append(summary['prompts'][p].get('best_mean', 0))
                if means:
                    avg = sum(means) / len(means)
                    row += f" **{avg:.4f}**          |"
                else:
                    row += f" {'N/A':^18} |"
            else:
                row += f" {'N/A':^18} |"
        lines.append(row)
        lines.append("")

    # Frame count summary
    lines.append("## Frame Counts\n")
    lines.append("| Config | ADT | HD-EPIC |")
    lines.append("| ------ | --- | ------- |")
    for config_label, config_dir in configs:
        row = f"| {config_label} |"
        for _, ds_dir in datasets:
            ws = os.path.join(ws_base, config_dir, ds_dir)
            sel_path = os.path.join(ws, 'selected_frames.json')
            if os.path.exists(sel_path):
                with open(sel_path) as f:
                    sel = json.load(f)
                n = len(sel.get('selected_frames', []))
                row += f" {n} |"
            else:
                img_dir = os.path.join(ws, 'images')
                if os.path.exists(img_dir):
                    n = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
                    row += f" {n} |"
                else:
                    row += " N/A |"
        lines.append(row)

    report = "\n".join(lines) + "\n"

    with open(output, 'w') as f:
        f.write(report)
    print(f"Report saved to {output}")
    print(report)


if __name__ == '__main__':
    main()
