"""Select frames for LangSplat using covisibility + blur filtering.

Uses the same covisibility algorithm as the Ego3DVQA VLM pipeline:
1. Load trained 3DGS model + COLMAP cameras
2. Compute Gaussian visibility matrix
3. Greedy set cover + IoU diversity sampling
4. Reject blurry frames via Laplacian variance

Outputs selected_frames.json to the data directory.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
from typing import List
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    qvec2rotmat,
)
from utils.graphics_utils import focal2fov
from utils.covisibility import compute_visibility_matrix, select_frames_coverage_diversity
from plyfile import PlyData


# Lightweight CameraInfo — no image loading
CamInfo = namedtuple('CamInfo', ['R', 'T', 'FovX', 'FovY', 'image_name'])


def load_cameras_lightweight(data_root):
    """Load camera parameters from COLMAP without opening images."""
    sparse_dir = os.path.join(data_root, 'sparse', '0')

    # Try binary first, fall back to text
    if os.path.exists(os.path.join(sparse_dir, 'images.bin')):
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, 'images.bin'))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, 'cameras.bin'))
    else:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, 'images.txt'))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, 'cameras.txt'))

    cam_infos = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            FovX = focal2fov(intr.params[0], intr.width)
            FovY = focal2fov(intr.params[0], intr.height)
        elif intr.model == "PINHOLE":
            FovX = focal2fov(intr.params[0], intr.width)
            FovY = focal2fov(intr.params[1], intr.height)
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        image_name = os.path.basename(extr.name)
        cam_infos.append(CamInfo(R=R, T=T, FovX=FovX, FovY=FovY, image_name=image_name))

    # Sort by image name for deterministic ordering
    cam_infos.sort(key=lambda c: c.image_name)
    return cam_infos


def load_gaussian_xyz(model_path):
    """Load Gaussian positions from PLY file."""
    ply_path = os.path.join(model_path, 'point_cloud.ply')
    if not os.path.exists(ply_path):
        # Try iteration subdirs
        for it in [45000, 30000, 7000]:
            candidate = os.path.join(model_path, 'point_cloud', f'iteration_{it}', 'point_cloud.ply')
            if os.path.exists(candidate):
                ply_path = candidate
                break

    plydata = PlyData.read(ply_path)
    xyz = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z'],
    ], axis=1)
    return torch.from_numpy(xyz).float()


def compute_blur_scores(image_dir, frame_names):
    """Compute Laplacian variance (sharpness) for each frame."""
    scores = {}
    for name in frame_names:
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            scores[name] = 0.0
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            scores[name] = 0.0
            continue
        # Downscale for speed
        h, w = img.shape
        if max(h, w) > 600:
            scale = 600 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        scores[name] = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Ego3DVQA dataset (with sparse/ and gs-output/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: data_root/selected_frames.json)')
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--diversity_threshold', type=float, default=0.5)
    parser.add_argument('--subsample', type=int, default=4)
    parser.add_argument('--blur_percentile', type=float, default=15.0,
                        help='Reject frames below this percentile of blur scores')
    args = parser.parse_args()

    data_root = args.data_root
    output_path = args.output or os.path.join(data_root, 'selected_frames.json')

    # 1. Load cameras (lightweight, no image loading)
    print("Loading cameras from COLMAP...")
    all_cameras = load_cameras_lightweight(data_root)
    print(f"  {len(all_cameras)} cameras loaded")

    # 2. Exclude open-close frames
    oc_path = os.path.join(data_root, 'open_close_frames.json')
    excluded = set()
    if os.path.exists(oc_path):
        with open(oc_path) as f:
            oc_data = json.load(f)
        if isinstance(oc_data, dict):
            excluded = set(oc_data.get("affected_frames", []))
        else:
            excluded = set(oc_data)
        if excluded:
            before = len(all_cameras)
            all_cameras = [c for c in all_cameras if c.image_name not in excluded]
            print(f"  Excluded {before - len(all_cameras)} open-close frames -> {len(all_cameras)}")

    # 3. Load Gaussian positions
    print("Loading Gaussian positions...")
    gs_path = os.path.join(data_root, 'gs-output')
    gaussian_xyz = load_gaussian_xyz(gs_path)
    print(f"  {gaussian_xyz.shape[0]} Gaussians")

    # 4. Compute visibility matrix
    print(f"Computing visibility matrix (subsample={args.subsample})...")
    visibility = compute_visibility_matrix(
        all_cameras, gaussian_xyz,
        subsample=args.subsample, batch_size=32,
    )

    # 5. Run covisibility selection
    print(f"Running coverage+diversity selection (threshold={args.diversity_threshold}, max={args.max_frames})...")
    selected_indices, stats = select_frames_coverage_diversity(
        visibility,
        diversity_threshold=args.diversity_threshold,
        max_frames=args.max_frames,
    )
    selected_indices = sorted(selected_indices)
    selected_names = [all_cameras[i].image_name for i in selected_indices]
    print(f"  Selected {len(selected_names)} frames "
          f"(phase1={stats['phase1_frames']}, phase2={stats['phase2_frames']}, "
          f"{stats['coverage_pct']:.1f}% coverage)")

    # 6. Blur filtering
    print("Computing blur scores...")
    image_dir = os.path.join(data_root, 'images')
    blur_scores = compute_blur_scores(image_dir, selected_names)
    scores_array = np.array([blur_scores[n] for n in selected_names])
    blur_threshold = np.percentile(scores_array, args.blur_percentile)
    print(f"  Blur scores: min={scores_array.min():.1f}, max={scores_array.max():.1f}, "
          f"mean={scores_array.mean():.1f}, threshold(p{args.blur_percentile:.0f})={blur_threshold:.1f}")

    # Remove blurry frames
    filtered_names = [n for n in selected_names if blur_scores[n] >= blur_threshold]
    n_removed = len(selected_names) - len(filtered_names)
    print(f"  Removed {n_removed} blurry frames -> {len(filtered_names)} final frames")

    # 7. Save results
    result = {
        'selected_frames': filtered_names,
        'stats': {
            **stats,
            'blur_threshold': float(blur_threshold),
            'blur_percentile': args.blur_percentile,
            'frames_before_blur': len(selected_names),
            'frames_after_blur': len(filtered_names),
            'frames_removed_blur': n_removed,
        },
        'params': {
            'max_frames': args.max_frames,
            'diversity_threshold': args.diversity_threshold,
            'subsample': args.subsample,
            'blur_percentile': args.blur_percentile,
        }
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(filtered_names)} frames to {output_path}")

    del visibility, gaussian_xyz
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
