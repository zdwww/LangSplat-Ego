"""Post-process segmentation maps to mask out dynamic objects and hands."""
import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    args = parser.parse_args()

    lf_dir = os.path.join(args.workspace, 'language_features')
    obj_masks = hand_masks = None

    for name, label in [('masks_rle.json', 'object'), ('hand_masks_rle.json', 'hand')]:
        path = os.path.join(args.workspace, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if label == 'object':
                obj_masks = data
            else:
                hand_masks = data
            print(f"Loaded {label} masks: {len(data.get('frames', {}))} frames")

    if not obj_masks and not hand_masks:
        print("No mask files found.")
        return

    seg_files = sorted(f for f in os.listdir(lf_dir) if f.endswith('_s.npy'))
    stats = {'total': 0, 'masked': 0, 'pixels_masked': 0, 'pixels_total': 0}

    for sf in tqdm(seg_files, desc="Post-processing"):
        stem = sf.replace('_s.npy', '')
        image_name = None
        for ext in ['.jpg', '.jpeg', '.png']:
            c = stem + ext
            if (obj_masks and c in obj_masks.get('frames', {})) or \
               (hand_masks and c in hand_masks.get('frames', {})):
                image_name = c
                break

        stats['total'] += 1
        if not image_name:
            continue

        seg_map = np.load(os.path.join(lf_dir, sf))
        _, seg_h, seg_w = seg_map.shape
        combined = np.ones((seg_h, seg_w), dtype=np.uint8)

        for masks_data in [obj_masks, hand_masks]:
            if masks_data and image_name in masks_data.get('frames', {}):
                m = mask_util.decode(masks_data['frames'][image_name])
                if m.shape[0] != seg_h or m.shape[1] != seg_w:
                    m = np.array(Image.fromarray(m).resize((seg_w, seg_h), Image.NEAREST))
                combined &= m

        exclude = combined == 0
        if exclude.sum() > 0:
            for lvl in range(seg_map.shape[0]):
                seg_map[lvl][exclude] = -1
            np.save(os.path.join(lf_dir, sf), seg_map)
            stats['masked'] += 1
            stats['pixels_masked'] += int(exclude.sum())
        stats['pixels_total'] += seg_h * seg_w

    pct = stats['pixels_masked'] / max(stats['pixels_total'], 1) * 100
    print(f"Done: {stats['masked']}/{stats['total']} maps modified, {pct:.1f}% pixels masked")


if __name__ == '__main__':
    main()
