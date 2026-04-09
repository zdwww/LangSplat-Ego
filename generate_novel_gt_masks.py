"""Generate SAM2 ground-truth masks for moved_050 novel views.

Reads captions.json, extracts moved_050 frames and their detections,
runs SAM2ImagePredictor with box prompts, saves binary mask PNGs and segments.json.

Adapted from generate_sam2_masks.py (which processes "render" mode from training images).

Intended to run in the da3 conda env (torch 2.5.1) with SAM2 loaded via sys.path.
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate SAM2 GT masks for moved_050 novel views")
    parser.add_argument("--captions_json", type=str, required=True,
                        help="Path to vlm-captions/captions.json")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Path to moved_050/rgb/ directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for masks/ and segments.json")
    parser.add_argument("--sam2_repo", type=str, default="/home/daiwei/ego3dvqa-local/sam2",
                        help="Path to SAM2 repository (added to sys.path)")
    parser.add_argument("--sam2_ckpt", type=str,
                        default="/mnt/raptor/daiwei/sam-ckpts/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM2 hydra config name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for SAM2 inference")
    args = parser.parse_args()

    # Add SAM2 repo to path for imports
    sys.path.insert(0, args.sam2_repo)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Load captions and filter for moved_050 frames
    print(f"Loading captions from {args.captions_json}")
    with open(args.captions_json) as f:
        captions = json.load(f)

    moved_frames = {}
    for frame_id, modes in captions.items():
        if "moved_050" in modes:
            moved_frames[frame_id] = modes["moved_050"]
    print(f"Found {len(moved_frames)} moved_050 frames")

    # Create output directories
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Load SAM2
    print(f"Loading SAM2 from {args.sam2_ckpt}...")
    predictor = SAM2ImagePredictor(build_sam2(args.sam2_cfg, args.sam2_ckpt, device=args.device))
    print("SAM2 loaded")

    segments_data = {}
    total_masks = 0

    for frame_id in tqdm(sorted(moved_frames.keys()), desc="Generating SAM2 masks"):
        frame = moved_frames[frame_id]
        detections = frame.get("detections", [])
        if not detections:
            continue

        # Load moved_050 RGB image
        image_path = os.path.join(args.rgb_dir, f"{frame_id}.jpg")
        if not os.path.exists(image_path):
            print(f"WARNING: {image_path} not found, skipping")
            continue

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"WARNING: Could not read {image_path}, skipping")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Set image for SAM2 predictor
        predictor.set_image(image_rgb)

        segments = []
        for obj_idx, det in enumerate(detections):
            category = det.get("category", "unknown")
            bbox = det.get("bbox_abs")
            if bbox is None:
                print(f"WARNING: No bbox_abs for {frame_id} det {obj_idx}, skipping")
                continue

            # SAM2 box prompt: [x1, y1, x2, y2]
            box = np.array(bbox, dtype=np.float32)

            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True,
                )

            # Take the mask with highest score
            best_idx = scores.argmax()
            best_mask = masks[best_idx]  # [H, W] boolean
            best_score = float(scores[best_idx])

            # Save as binary PNG (0/255)
            mask_filename = f"{frame_id}_{obj_idx:03d}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, (best_mask.astype(np.uint8) * 255))

            segments.append({
                "obj_idx": obj_idx,
                "category": category,
                "description": det.get("description", ""),
                "mask_file": f"masks/{mask_filename}",
                "bbox_abs": bbox,
                "sam2_score": best_score,
                "mask_pixels": int(best_mask.sum()),
            })
            total_masks += 1

        segments_data[frame_id] = {
            "image_file": f"{frame_id}.jpg",
            "image_size": list(image_rgb.shape[:2][::-1]),  # [W, H]
            "num_detections": len(segments),
            "segments": segments,
        }

    # Save segments.json
    segments_path = os.path.join(args.output_dir, "segments.json")
    with open(segments_path, "w") as f:
        json.dump(segments_data, f, indent=2)

    print(f"\nDone! {len(segments_data)} frames, {total_masks} masks")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
