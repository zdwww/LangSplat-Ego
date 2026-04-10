"""
preprocess_sam2_ego3dvqa.py — CLIP encoding + text feature blending for SAM2 segments.

Adapted from /home/daiwei/LangSplat-variants/LangSplat/preprocess_sam2.py.
Reads pre-computed SAM2 segments (from generate_sam2_masks.py), encodes with OpenCLIP,
optionally blends image + text features, outputs LangSplat-compatible _s.npy and _f.npy.

Runs in langsplat_v2 env.
"""

import os
import json
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from dataclasses import dataclass, field
from typing import Tuple, Type


# ============================================================
# OpenCLIP model (from variant's preprocess_sam2.py)
# ============================================================

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

    def encode_text(self, text_list):
        tokens = self.tokenizer(text_list).to("cuda")
        with torch.no_grad():
            return self.model.encode_text(tokens)


# ============================================================
# Utility functions
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
    """Save a grid montage of CLIP input tiles with category labels."""
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


def slerp(v0, v1, t):
    """Spherical linear interpolation between unit vectors.
    v0, v1: [N, D] L2-normalized tensors
    t: [N, 1] or scalar interpolation weights in [0, 1]
    Returns: [N, D] L2-normalized interpolated vectors
    """
    dot = (v0 * v1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    small = (theta.abs() < 1e-4).squeeze(-1)
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    result = w0 * v0 + w1 * v1
    if small.any():
        lerp_result = (1.0 - t) * v0 + t * v1
        lerp_result = lerp_result / lerp_result.norm(dim=-1, keepdim=True)
        result[small] = lerp_result[small]
    return result / result.norm(dim=-1, keepdim=True)


# ============================================================
# Main preprocessing
# ============================================================

def preprocess_sam2(args):
    # Load segments.json
    print(f"Loading segments from {args.segments_json}")
    with open(args.segments_json) as f:
        segments_data = json.load(f)
    print(f"Found {len(segments_data)} frames")

    # Load OpenCLIP model
    print("Loading OpenCLIP ViT-B-16...")
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    print("OpenCLIP loaded")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    # Base directory for mask paths (segments.json mask_file entries are relative)
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

        # --- Load static mask (optional) ---
        static_mask = None
        if args.static_masks_dir:
            static_mask_path = os.path.join(args.static_masks_dir, image_name + ".png")
            if os.path.exists(static_mask_path):
                static_mask = cv2.imread(static_mask_path, cv2.IMREAD_GRAYSCALE)

        # --- Process each segment ---
        seg_tiles = []
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
                continue

            # Combine with static mask (AND operation)
            if static_mask is not None:
                effective_mask = (obj_mask > 0) & (static_mask > 0)
            else:
                effective_mask = obj_mask > 0

            # Assign pixels in seg_map (later objects overwrite earlier on overlap)
            seg_map[effective_mask] = obj_idx

            # Tight crop from original image, zero-out non-mask pixels
            ys, xs = np.where(effective_mask)
            if len(ys) == 0:
                seg_tiles.append(np.zeros((224, 224, 3), dtype=np.uint8))
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

        # --- Encode with OpenCLIP ---
        if M > 0 and len(seg_tiles) > 0:
            text_weight = args.text_weight
            blend_mode = args.blend_mode
            need_image = blend_mode == "slerp_adaptive" or text_weight < 1.0
            need_text = blend_mode == "slerp_adaptive" or text_weight > 0.0

            # Image features
            if need_image:
                seg_imgs = np.stack(seg_tiles, axis=0)
                seg_tensor = torch.from_numpy(seg_imgs.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
                seg_tensor = seg_tensor.to("cuda")

                all_features = []
                for i in range(0, M, args.batch_size):
                    batch = seg_tensor[i:i + args.batch_size]
                    with torch.no_grad():
                        feat = clip_model.encode_image(batch)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    all_features.append(feat.detach().cpu().float())
                image_features = torch.cat(all_features, dim=0)  # [M, 512]

            # Text features
            if need_text:
                prefix = args.text_prompt_prefix
                text_inputs = [prefix + cat for cat in categories]
                text_feat = clip_model.encode_text(text_inputs)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                text_features = text_feat.detach().cpu().float()  # [M, 512]

            # Combine
            if blend_mode == "slerp_adaptive":
                max_tw = args.max_text_weight if args.max_text_weight is not None else text_weight
                sim = F.cosine_similarity(image_features, text_features, dim=-1)
                tw_per_seg = ((1.0 - sim).clamp(0, 1) * max_tw).unsqueeze(-1)  # [M, 1]
                combined = slerp(image_features, text_features, tw_per_seg)
                clip_features = combined.numpy()
                tw_np = tw_per_seg.squeeze(-1).numpy()
                if M > 1:
                    print(f"  Adaptive tw: mean={tw_np.mean():.3f} min={tw_np.min():.3f} "
                          f"max={tw_np.max():.3f} std={tw_np.std():.3f}")
                else:
                    print(f"  Adaptive tw: {tw_np.item():.3f}")
            else:
                # Original LERP (v2 backward compat)
                if text_weight <= 0.0:
                    clip_features = image_features.numpy()
                elif text_weight >= 1.0:
                    clip_features = text_features.numpy()
                else:
                    combined = (1.0 - text_weight) * image_features + text_weight * text_features
                    combined = combined / combined.norm(dim=-1, keepdim=True)
                    clip_features = combined.numpy()
        else:
            clip_features = np.zeros((0, 512), dtype=np.float32)

        # --- Build 4-channel seg_map with offset indexing ---
        if M > 0:
            seg_map_4 = np.stack([seg_map.copy() for _ in range(4)], axis=0)  # [4, H, W]
            for k in range(4):
                valid = seg_map_4[k] != -1
                seg_map_4[k][valid] += k * M
            feature_array = np.tile(clip_features, (4, 1))  # [4*M, 512]
        else:
            seg_map_4 = np.full((4, H, W), -1, dtype=np.int32)
            feature_array = np.zeros((4, 512), dtype=np.float32)

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
    parser = argparse.ArgumentParser(description="CLIP encoding + text blending for SAM2 segments")
    parser.add_argument("--segments_json", type=str, required=True,
                        help="Path to segments.json from generate_sam2_masks.py")
    parser.add_argument("--sam2_masks_dir", type=str, default=None,
                        help="(unused, kept for compat) SAM2 masks dir")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to full-resolution RGB images")
    parser.add_argument("--static_masks_dir", type=str, default=None,
                        help="Path to static masks (0=dynamic, 255=keep)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for language_features/ (_f.npy and _s.npy)")
    parser.add_argument("--viz_dir", type=str, default=None,
                        help="Directory for CLIP-input montage visualizations")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for CLIP encoding")
    parser.add_argument("--text_weight", type=float, default=0.0,
                        help="Weight for text CLIP features (0=image only, 1=text only)")
    parser.add_argument("--text_prompt_prefix", type=str, default="",
                        help="Prefix for category names (e.g., 'a photo of a ')")
    parser.add_argument("--blend_mode", type=str, default="lerp",
                        choices=["lerp", "slerp_adaptive"],
                        help="Feature blending mode: lerp (v2) or slerp_adaptive (v3)")
    parser.add_argument("--max_text_weight", type=float, default=None,
                        help="Max text weight for slerp_adaptive mode (default: text_weight)")

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    preprocess_sam2(args)
