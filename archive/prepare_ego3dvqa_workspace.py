"""Prepare Ego3DVQA workspace for LangSplat pipeline.

Reads selected_frames.json, downscales images by 2x,
symlinks COLMAP data, and copies mask files to workspace.
"""
import os
import json
import argparse
import shutil
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--selected_frames', type=str, default=None,
                        help='Path to selected_frames.json (default: data_root/selected_frames.json)')
    args = parser.parse_args()

    data_root = args.data_root
    workspace = args.workspace
    sel_path = args.selected_frames or os.path.join(data_root, 'selected_frames.json')

    os.makedirs(os.path.join(workspace, 'images'), exist_ok=True)
    os.makedirs(os.path.join(workspace, 'sparse'), exist_ok=True)

    # 1. Read selected frames
    with open(sel_path) as f:
        sel_data = json.load(f)
    frames = sel_data['selected_frames']
    print(f"Selected frames: {len(frames)}")

    # 2. Downscale and copy
    src_dir = os.path.join(data_root, 'images')
    dst_dir = os.path.join(workspace, 'images')
    skipped = 0
    for name in tqdm(frames, desc="Downscaling images"):
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            skipped += 1
            continue
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            print(f"WARNING: {name} not found")
            continue
        img = Image.open(src)
        w, h = img.size
        img.resize((w // 2, h // 2), Image.LANCZOS).save(dst, quality=95)

    if skipped:
        print(f"Skipped {skipped} existing images")

    # 3. Symlink COLMAP
    sparse_src = os.path.join(data_root, 'sparse', '0')
    sparse_dst = os.path.join(workspace, 'sparse', '0')
    if os.path.islink(sparse_dst):
        os.unlink(sparse_dst)
    if not os.path.exists(sparse_dst):
        os.symlink(sparse_src, sparse_dst)
        print(f"Symlinked sparse/0")

    # 4. Copy masks
    for mf in ['masks_rle.json', 'hand_masks_rle.json']:
        src = os.path.join(data_root, mf)
        dst = os.path.join(workspace, mf)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {mf}")

    actual = len(os.listdir(dst_dir))
    sample = Image.open(os.path.join(dst_dir, os.listdir(dst_dir)[0]))
    print(f"\nWorkspace ready: {actual} images at {sample.size}")


if __name__ == '__main__':
    main()
