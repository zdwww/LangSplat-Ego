"""Visualize preprocessing results: images, masks, and segmentation maps."""
import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util


def colorize_segmap(seg_map):
    h, w = seg_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = seg_map >= 0
    if not valid.any():
        return rgb
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(int(seg_map[valid].max()) + 1, 3), dtype=np.uint8)
    for uid in np.unique(seg_map[valid]):
        rgb[seg_map == uid] = colors[uid]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=8)
    args = parser.parse_args()

    lf_dir = os.path.join(args.workspace, 'language_features')
    img_dir = os.path.join(args.workspace, 'images')
    diag_dir = os.path.join(args.workspace, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    seg_files = sorted(f for f in os.listdir(lf_dir) if f.endswith('_s.npy'))
    n_total = len(seg_files)
    print(f"Total feature files: {n_total}")

    # Load masks
    obj_masks = hand_masks = None
    for name, label in [('masks_rle.json', 'object'), ('hand_masks_rle.json', 'hand')]:
        path = os.path.join(args.workspace, name)
        if os.path.exists(path):
            with open(path) as f:
                if label == 'object': obj_masks = json.load(f)
                else: hand_masks = json.load(f)

    # Coverage stats
    level_names = {0: 'default', 1: 's (small)', 2: 'm (medium)', 3: 'l (large)'}
    level_coverages = {i: [] for i in range(4)}
    for sf in seg_files:
        seg = np.load(os.path.join(lf_dir, sf))
        for lvl in range(4):
            level_coverages[lvl].append(float((seg[lvl] != -1).sum() / seg[lvl].size))

    stats = {'total_frames': n_total}
    for lvl in range(4):
        c = level_coverages[lvl]
        stats[f'level_{lvl}'] = {
            'name': level_names[lvl],
            'mean': float(np.mean(c)), 'min': float(np.min(c)), 'max': float(np.max(c)),
        }
        print(f"  Level {lvl} ({level_names[lvl]}): mean={np.mean(c):.1%}")

    with open(os.path.join(diag_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Montage
    indices = np.linspace(0, n_total - 1, args.num_samples, dtype=int)
    samples = [seg_files[i] for i in indices]

    fig, axes = plt.subplots(len(samples), 6, figsize=(30, 5 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for row, sf in enumerate(samples):
        stem = sf.replace('_s.npy', '')
        seg = np.load(os.path.join(lf_dir, sf))

        # Image
        img_name = None
        for ext in ['.jpg', '.jpeg', '.png']:
            if os.path.exists(os.path.join(img_dir, stem + ext)):
                img_name = stem + ext
                break
        img = np.array(Image.open(os.path.join(img_dir, img_name))) if img_name else np.zeros((seg.shape[1], seg.shape[2], 3), dtype=np.uint8)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'{stem}\n{img.shape[1]}x{img.shape[0]}', fontsize=8)
        axes[row, 0].axis('off')

        # Mask overlay
        overlay = img.copy().astype(float)
        seg_h, seg_w = seg.shape[1], seg.shape[2]
        combined = np.ones((seg_h, seg_w), dtype=np.uint8)
        if img_name:
            for md in [obj_masks, hand_masks]:
                if md and img_name in md.get('frames', {}):
                    m = mask_util.decode(md['frames'][img_name])
                    if m.shape != (seg_h, seg_w):
                        m = np.array(Image.fromarray(m).resize((seg_w, seg_h), Image.NEAREST))
                    combined &= m
        excluded = combined == 0
        overlay[excluded, 0] = np.clip(overlay[excluded, 0] * 0.5 + 127, 0, 255)
        overlay[excluded, 1] *= 0.3
        overlay[excluded, 2] *= 0.3
        axes[row, 1].imshow(overlay.astype(np.uint8))
        axes[row, 1].set_title(f'Mask: {excluded.sum()/excluded.size*100:.1f}% masked', fontsize=8)
        axes[row, 1].axis('off')

        # Seg maps
        for col, lvl in enumerate(range(4)):
            rgb = colorize_segmap(seg[lvl])
            cov = (seg[lvl] != -1).sum() / seg[lvl].size * 100
            n_seg = len(np.unique(seg[lvl][seg[lvl] >= 0])) if (seg[lvl] >= 0).any() else 0
            axes[row, col + 2].imshow(rgb)
            axes[row, col + 2].set_title(f'L{lvl} ({level_names[lvl]})\n{n_seg} segs, {cov:.0f}%', fontsize=8)
            axes[row, col + 2].axis('off')

    plt.tight_layout()
    out = os.path.join(diag_dir, 'preprocessing_samples.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved montage to {out}")


if __name__ == '__main__':
    main()
