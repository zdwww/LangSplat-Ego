"""
preprocess_qwen3vl_ego3dvqa.py — Qwen3-VL-Embedding encoding for SAM2 segments.

Replaces OpenCLIP in preprocess_sam2_ego3dvqa.py with Qwen3-VL-Embedding-2B.
Supports two encoding modes:
  - image_only: encode SAM2 crops as images
  - multimodal: encode SAM2 crops + category text together (unified embedding)

Outputs LangSplat-compatible _s.npy and _f.npy (512D via Matryoshka truncation).

Requires: qwen3vl conda env (torch 2.4+cu118, transformers 4.57+)
  PYTHONPATH must include Qwen3-VL-Embedding/src
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Import Qwen3-VL-Embedding
sys.path.insert(0, os.environ.get(
    "QWEN_SRC", "/home/daiwei/Ego3DVQA-GS/Qwen3-VL-Embedding/src"))
from models.qwen3_vl_embedding import Qwen3VLEmbedder


# ============================================================
# Utility functions (same as preprocess_sam2_ego3dvqa.py)
# ============================================================

def pad_img(img):
    """Pad image to square with black borders."""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def save_montage(tiles, categories, save_path, tile_size=224, max_cols=8):
    """Save a grid montage of input tiles with category labels."""
    M = len(tiles)
    if M == 0:
        return
    n_cols = min(M, max_cols)
    n_rows = (M + n_cols - 1) // n_cols

    label_h = 20
    cell_h = tile_size + label_h
    montage = np.zeros((n_rows * cell_h, n_cols * tile_size, 3), dtype=np.uint8)

    for idx in range(M):
        r, c = divmod(idx, n_cols)
        y_off = r * cell_h
        x_off = c * tile_size
        montage[y_off:y_off + tile_size, x_off:x_off + tile_size] = tiles[idx]
        label = categories[idx] if idx < len(categories) else ""
        if len(label) > 25:
            label = label[:22] + "..."
        cv2.putText(montage, label, (x_off + 2, y_off + tile_size + 14),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))


# ============================================================
# Main preprocessing
# ============================================================

def preprocess_qwen3vl(args):
    # Load segments.json
    print(f"Loading segments from {args.segments_json}")
    with open(args.segments_json) as f:
        segments_data = json.load(f)
    print(f"Found {len(segments_data)} frames")

    # Load Qwen3-VL-Embedding model
    print(f"Loading {args.model_name}...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=args.model_name,
        dtype=torch.bfloat16,
    )
    print(f"Model loaded (embed_dim={args.embed_dim}, mode={args.encode_mode})")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    # Base directory for mask paths
    sam2_base = os.path.dirname(args.segments_json)

    frame_keys = sorted(segments_data.keys())
    skipped = 0
    for frame_key in tqdm(frame_keys, desc="Processing frames"):
        frame = segments_data[frame_key]
        image_filename = frame["original_image"]
        image_name = image_filename.split(".")[0]
        segments = frame.get("segments", [])
        M = len(segments)

        # Resume support: skip already-processed frames
        s_path = os.path.join(args.output_dir, image_name + "_s.npy")
        f_path = os.path.join(args.output_dir, image_name + "_f.npy")
        if os.path.exists(s_path) and os.path.exists(f_path):
            skipped += 1
            continue

        # --- Load RGB image ---
        image_path = os.path.join(args.images_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"WARNING: Could not load {image_path}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        # --- Process each segment ---
        seg_tiles = []
        pil_crops = []
        categories = []
        seg_map = np.full((H, W), -1, dtype=np.int32)

        for seg_info in segments:
            obj_idx = seg_info["obj_idx"]
            category = seg_info.get("category", "unknown")
            categories.append(category)

            # Load SAM2 binary mask
            mask_path = os.path.join(sam2_base, seg_info["mask_file"])
            obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if obj_mask is None:
                print(f"WARNING: Could not load mask {mask_path}")
                seg_tiles.append(np.zeros((224, 224, 3), dtype=np.uint8))
                pil_crops.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
                continue

            effective_mask = obj_mask > 0
            seg_map[effective_mask] = obj_idx

            # Tight crop, zero-out non-mask pixels
            ys, xs = np.where(effective_mask)
            if len(ys) == 0:
                seg_tiles.append(np.zeros((224, 224, 3), dtype=np.uint8))
                pil_crops.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
                continue

            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1

            crop = image[y1:y2, x1:x2].copy()
            mask_crop = effective_mask[y1:y2, x1:x2]
            crop[~mask_crop] = 0

            # Pad to square, resize to 224x224
            padded = pad_img(crop)
            tile = cv2.resize(padded, (224, 224))
            seg_tiles.append(tile)
            pil_crops.append(Image.fromarray(tile))

        # --- Encode with Qwen3-VL-Embedding ---
        if M > 0 and len(pil_crops) > 0:
            all_features = []
            for i in range(0, M, args.batch_size):
                batch_crops = pil_crops[i:i + args.batch_size]
                batch_cats = categories[i:i + args.batch_size]

                # Build inputs based on encode mode
                inputs = []
                for crop, cat in zip(batch_crops, batch_cats):
                    if args.encode_mode == "multimodal":
                        inputs.append({"image": crop, "text": cat})
                    else:  # image_only
                        inputs.append({"image": crop})

                with torch.no_grad():
                    emb = embedder.process(inputs)  # [batch, 2048]
                    # Matryoshka truncation
                    emb = emb[:, :args.embed_dim]
                    emb = F.normalize(emb.float(), dim=-1)
                    all_features.append(emb.cpu())

            clip_features = torch.cat(all_features, dim=0).numpy()  # [M, embed_dim]
        else:
            clip_features = np.zeros((0, args.embed_dim), dtype=np.float32)

        # --- Build 4-channel seg_map with offset indexing ---
        if M > 0:
            seg_map_4 = np.stack([seg_map.copy() for _ in range(4)], axis=0)
            for k in range(4):
                valid = seg_map_4[k] != -1
                seg_map_4[k][valid] += k * M
            feature_array = np.tile(clip_features, (4, 1))  # [4*M, embed_dim]
        else:
            seg_map_4 = np.full((4, H, W), -1, dtype=np.int32)
            feature_array = np.zeros((4, args.embed_dim), dtype=np.float32)

        # --- Save ---
        np.save(s_path, seg_map_4)
        np.save(f_path, feature_array)

        # --- Save visualization montage ---
        if args.viz_dir and M > 0:
            viz_path = os.path.join(args.viz_dir, image_name + "_montage.jpg")
            save_montage(seg_tiles, categories, viz_path)

    print(f"\nDone! Processed {len(frame_keys) - skipped} frames ({skipped} skipped/resumed)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Embedding encoding for SAM2 segments")
    parser.add_argument("--segments_json", type=str, required=True,
                        help="Path to segments.json from generate_sam2_masks.py")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to full-resolution RGB images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for language_features/ (_f.npy and _s.npy)")
    parser.add_argument("--viz_dir", type=str, default=None,
                        help="Directory for input montage visualizations")
    parser.add_argument("--encode_mode", type=str, default="multimodal",
                        choices=["image_only", "multimodal"],
                        help="image_only: crops only; multimodal: crops + category text together")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-Embedding-2B",
                        help="HuggingFace model name or local path")
    parser.add_argument("--embed_dim", type=int, default=512,
                        help="Matryoshka truncation dimension (default: 512)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Segments per Qwen3-VL forward pass")

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    preprocess_qwen3vl(args)
