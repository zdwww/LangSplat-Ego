"""Prepare Ego3DVQA workspace for LangSplat v2 pipeline.

Reads captions.json render-mode frames, symlinks images at full resolution,
symlinks COLMAP data, and copies mask files to workspace.
"""
import os
import json
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--captions_json', type=str, required=True,
                        help='Path to vlm-captions/captions.json')
    args = parser.parse_args()

    data_root = args.data_root
    workspace = args.workspace

    os.makedirs(os.path.join(workspace, 'images'), exist_ok=True)

    # 1. Read captions.json, extract render-mode frame IDs
    with open(args.captions_json) as f:
        captions = json.load(f)

    render_frame_ids = sorted(
        fid for fid, modes in captions.items() if "render" in modes
    )
    # Map to image filenames
    frames = [f"camera-rgb_{fid}.jpg" for fid in render_frame_ids]
    print(f"Render-mode frames: {len(frames)}")

    # 2. Symlink images at full resolution (no downscaling)
    src_dir = os.path.join(data_root, 'images')
    dst_dir = os.path.join(workspace, 'images')
    linked = 0
    missing = 0
    for name in frames:
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            print(f"WARNING: {name} not found in {src_dir}")
            missing += 1
            continue
        os.symlink(src, dst)
        linked += 1

    if missing:
        print(f"Missing {missing} images")
    existing = len(os.listdir(dst_dir))
    print(f"Symlinked {linked} new images ({existing} total in workspace)")

    # 3. Symlink COLMAP sparse/0
    sparse_src = os.path.join(data_root, 'sparse', '0')
    sparse_parent = os.path.join(workspace, 'sparse')
    os.makedirs(sparse_parent, exist_ok=True)
    sparse_dst = os.path.join(sparse_parent, '0')
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

    # 5. Write selected_frames.json for downstream compatibility
    sel_path = os.path.join(workspace, 'selected_frames.json')
    sel_data = {
        'source': 'render-mode from captions.json',
        'captions_json': args.captions_json,
        'total_render_frames': len(render_frame_ids),
        'selected_frames': frames,
    }
    with open(sel_path, 'w') as f:
        json.dump(sel_data, f, indent=2)
    print(f"Wrote selected_frames.json ({len(frames)} frames)")

    print(f"\nWorkspace ready: {workspace}")


if __name__ == '__main__':
    main()
