"""Novel-view segmentation evaluation for LangSplat on Ego3DVQA datasets.

Renders language features from moved_050 camera poses, computes per-object
segmentation via CLIP text queries, and evaluates against SAM2 ground-truth masks.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autoencoder'))

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from autoencoder.model import Autoencoder

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip required")


# ---------------------------------------------------------------------------
# Reused from eval_ego3dvqa.py
# ---------------------------------------------------------------------------

def load_autoencoder(ckpt_path):
    model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model


def get_relevancy(feat_512, text_embed, neg_embeds):
    phrases = torch.cat([text_embed, neg_embeds], dim=0)
    p = phrases.to(feat_512.dtype)
    output = torch.mm(feat_512, p.T)
    pos_vals = output[:, :1]
    neg_vals = output[:, 1:]
    sims = torch.stack((pos_vals.repeat(1, neg_vals.shape[1]), neg_vals), dim=-1)
    softmax = torch.softmax(10 * sims, dim=-1)
    best_id = softmax[..., 0].argmin(dim=1)
    result = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], neg_vals.shape[1], 2))
    return result[:, 0, 0]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gaussian_model(workspace):
    """Load GaussianModel from checkpoint without Scene."""
    ckpt_path = os.path.join(workspace, 'output_1', 'chkpnt30000.pth')
    print(f"Loading checkpoint: {ckpt_path}")
    model_params, first_iter = torch.load(ckpt_path)

    gaussians = GaussianModel(sh_degree=3)
    # restore with mode='test' skips training_setup
    opt = SimpleNamespace(include_feature=True)
    gaussians.restore(model_params, opt, mode='test')
    print(f"  Loaded {gaussians.get_xyz.shape[0]} Gaussians, language_feature shape: {gaussians.get_language_feature.shape}")
    return gaussians


# ---------------------------------------------------------------------------
# Camera handling
# ---------------------------------------------------------------------------

def load_moved_cameras(metadata_json):
    """Parse moved_050/metadata.json -> list of camera parameter dicts."""
    with open(metadata_json) as f:
        metadata = json.load(f)

    cameras = []
    for frame in metadata['frames']:
        cam = frame['camera_moved']
        cameras.append({
            'frame_id': frame['filename'],
            'frame_index': frame['frame_index'],
            'R': np.array(cam['R'], dtype=np.float64),
            'T': np.array(cam['T'], dtype=np.float64),
            'FovX': cam['FovX'],
            'FovY': cam['FovY'],
            'width': cam['width'],
            'height': cam['height'],
        })
    return cameras


def create_camera(cam_params, device="cuda"):
    """Create a Camera object from moved camera parameters."""
    H, W = cam_params['height'], cam_params['width']
    dummy_image = torch.zeros(3, H, W)
    return Camera(
        colmap_id=0,
        R=cam_params['R'],
        T=cam_params['T'],
        FoVx=cam_params['FovX'],
        FoVy=cam_params['FovY'],
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=cam_params['frame_id'],
        uid=0,
        data_device=device,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred_mask, gt_mask):
    """Compute IoU and related metrics between two binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(intersection) / (float(union) + 1e-8)
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    precision = float(intersection) / (float(pred_pixels) + 1e-8)
    recall = float(intersection) / (float(gt_pixels) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_pixels': pred_pixels,
        'gt_pixels': gt_pixels,
        'intersection': int(intersection),
        'union': int(union),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_visualization(rgb, rel_map, pred_mask, gt_mask, category, metrics, bbox, save_path):
    """Generate a 5-panel visualization figure."""
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # Panel 1: RGB with bbox
    axes[0].imshow(rgb)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title(f'"{category}"')
    axes[0].axis('off')

    # Panel 2: Relevancy heatmap (fixed [0, 1])
    im = axes[1].imshow(rel_map, cmap='turbo', vmin=0, vmax=1)
    axes[1].set_title(f'Relevancy max={rel_map.max():.3f}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Panel 3: Predicted mask
    axes[2].imshow(pred_mask.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Pred ({pred_mask.sum()} px)')
    axes[2].axis('off')

    # Panel 4: GT mask
    axes[3].imshow(gt_mask.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f'GT ({gt_mask.sum()} px)')
    axes[3].axis('off')

    # Panel 5: TP/FP/FN overlay on RGB
    overlay = rgb.copy().astype(np.float32) / 255.0
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, ~gt_mask)
    fn = np.logical_and(~pred_mask, gt_mask)
    alpha = 0.5
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha  # green
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha  # red
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha  # blue
    axes[4].imshow(np.clip(overlay, 0, 1))
    axes[4].set_title(f'IoU={metrics["iou"]:.3f} P={metrics["precision"]:.2f} R={metrics["recall"]:.2f}')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--metadata_json', type=str, required=True,
                        help="Path to moved_050/metadata.json")
    parser.add_argument('--captions_json', type=str, required=True,
                        help="Path to vlm-captions/captions.json")
    parser.add_argument('--gt_masks_dir', type=str, required=True,
                        help="Directory with masks/ and segments.json from generate_novel_gt_masks.py")
    parser.add_argument('--moved_rgb_dir', type=str, required=True,
                        help="Path to moved_050/rgb/ for visualization")
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_vis_frames', type=int, default=10)
    args = parser.parse_args()

    output_dir = os.path.join(args.workspace, 'eval_novel_results')
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- Load models ----
    print("Loading Gaussian model...")
    gaussians = load_gaussian_model(args.workspace)

    print("Loading autoencoder...")
    ae = load_autoencoder(args.ae_ckpt)

    print("Loading CLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16')
    clip_model = clip_model.cuda().eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    # Negative embeddings (same as eval_ego3dvqa.py)
    negatives = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        neg_embeds = clip_model.encode_text(
            torch.cat([tokenizer(n) for n in negatives]).cuda()
        ).float()
        neg_embeds = F.normalize(neg_embeds, dim=-1)

    # Rendering params
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    opt = SimpleNamespace(include_feature=True)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # ---- Load data ----
    print("Loading camera poses...")
    camera_list = load_moved_cameras(args.metadata_json)
    print(f"  {len(camera_list)} moved_050 cameras")

    print("Loading captions...")
    with open(args.captions_json) as f:
        captions = json.load(f)

    print("Loading GT segments...")
    with open(os.path.join(args.gt_masks_dir, 'segments.json')) as f:
        gt_segments = json.load(f)

    # ---- Pre-compute unique text embeddings ----
    all_categories = set()
    for cam_params in camera_list:
        fid = cam_params['frame_id']
        if fid in captions and 'moved_050' in captions[fid]:
            for det in captions[fid]['moved_050'].get('detections', []):
                all_categories.add(det['category'])

    print(f"Encoding {len(all_categories)} unique category text queries...")
    text_embeds = {}
    with torch.no_grad():
        for cat in sorted(all_categories):
            e = clip_model.encode_text(tokenizer(cat).cuda()).float()
            text_embeds[cat] = F.normalize(e, dim=-1)

    # Select frames for visualization
    vis_indices = set(np.linspace(0, len(camera_list) - 1,
                                  min(args.num_vis_frames, len(camera_list)),
                                  dtype=int).tolist())

    # ---- Evaluate ----
    all_results = {}
    all_ious = []

    for cam_idx, cam_params in enumerate(tqdm(camera_list, desc="Evaluating novel views")):
        frame_id = cam_params['frame_id']

        # Skip frames without moved_050 detections or GT masks
        if frame_id not in captions or 'moved_050' not in captions[frame_id]:
            continue
        detections = captions[frame_id]['moved_050'].get('detections', [])
        if not detections or frame_id not in gt_segments:
            continue

        gt_frame = gt_segments[frame_id]

        # 1. Create camera and render
        camera = create_camera(cam_params)
        with torch.no_grad():
            output = render(camera, gaussians, pipe, bg, opt)
            rgb_tensor = output["render"]           # (3, H, W)
            lang_feat_3d = output["language_feature_image"]  # (3, H, W)

        H, W = lang_feat_3d.shape[1:]

        # 2. Decode 3D -> 512D
        flat = lang_feat_3d.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        with torch.no_grad():
            feat_512 = F.normalize(ae.decode(flat), dim=-1)

        # Load RGB for visualization
        rgb_path = os.path.join(args.moved_rgb_dir, f"{frame_id}.jpg")
        if os.path.exists(rgb_path):
            rgb_img = np.array(Image.open(rgb_path))
            # Resize to match rendered resolution if needed
            if rgb_img.shape[:2] != (H, W):
                rgb_img = np.array(Image.open(rgb_path).resize((W, H)))
        else:
            rgb_img = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # 3. Per-detection evaluation
        frame_results = []
        for obj_idx, det in enumerate(detections):
            category = det['category']

            if category not in text_embeds:
                continue

            # Find matching GT segment
            gt_seg = None
            for seg in gt_frame['segments']:
                if seg['obj_idx'] == obj_idx:
                    gt_seg = seg
                    break
            if gt_seg is None:
                continue

            # Compute relevancy mask
            with torch.no_grad():
                rel = get_relevancy(feat_512, text_embeds[category], neg_embeds)
            rel_map = rel.cpu().numpy().reshape(H, W)
            pred_mask = rel_map > args.threshold

            # Load GT mask
            gt_mask_path = os.path.join(args.gt_masks_dir, gt_seg['mask_file'])
            gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is None:
                print(f"WARNING: Could not read {gt_mask_path}, skipping")
                continue
            # Resize GT mask to rendered resolution if needed
            if gt_mask_raw.shape[:2] != (H, W):
                gt_mask_raw = cv2.resize(gt_mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_mask = gt_mask_raw > 127

            # Compute metrics
            metrics = compute_iou(pred_mask, gt_mask)

            frame_results.append({
                'obj_idx': obj_idx,
                'category': category,
                'bbox_abs': det.get('bbox_abs'),
                'iou': metrics['iou'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'relevancy_max': float(rel_map.max()),
                'relevancy_mean': float(rel_map.mean()),
            })

            # Generate visualization for selected frames
            if cam_idx in vis_indices:
                safe_cat = category.replace(' ', '_').replace('/', '_')[:30]
                vis_path = os.path.join(vis_dir, f"{frame_id}_{obj_idx:03d}_{safe_cat}.png")
                create_visualization(
                    rgb_img, rel_map, pred_mask, gt_mask,
                    category, metrics, det.get('bbox_abs'), vis_path
                )

        if frame_results:
            frame_iou = np.mean([r['iou'] for r in frame_results])
            all_ious.append(frame_iou)
            all_results[frame_id] = {
                'num_objects': len(frame_results),
                'mean_iou': float(frame_iou),
                'objects': frame_results,
            }

        # Free GPU memory
        del output, rgb_tensor, lang_feat_3d, flat, feat_512
        torch.cuda.empty_cache()

    # ---- Aggregate metrics ----
    # Per-category aggregation
    cat_ious = {}
    for fid, fr in all_results.items():
        for obj in fr['objects']:
            cat = obj['category']
            if cat not in cat_ious:
                cat_ious[cat] = []
            cat_ious[cat].append(obj['iou'])

    per_category = {}
    for cat, ious in sorted(cat_ious.items()):
        per_category[cat] = {
            'mean_iou': float(np.mean(ious)),
            'count': len(ious),
        }

    summary = {
        'experiment': {
            'workspace': args.workspace,
            'ae_ckpt': args.ae_ckpt,
            'threshold': args.threshold,
            'num_frames': len(all_results),
            'total_objects': sum(r['num_objects'] for r in all_results.values()),
        },
        'metrics': {
            'mean_iou': float(np.mean(all_ious)) if all_ious else 0.0,
            'median_iou': float(np.median(all_ious)) if all_ious else 0.0,
            'std_iou': float(np.std(all_ious)) if all_ious else 0.0,
            'per_category': per_category,
        },
        'frames': all_results,
    }

    summary_path = os.path.join(output_dir, 'eval_novel_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print(f"Novel View Evaluation: {args.workspace}")
    print(f"Threshold: {args.threshold}")
    print(f"Frames: {summary['experiment']['num_frames']}, "
          f"Objects: {summary['experiment']['total_objects']}")
    print("-" * 70)
    print(f"{'Mean IoU':<15} {'Median IoU':<15} {'Std IoU':<15}")
    print(f"{summary['metrics']['mean_iou']:<15.4f} "
          f"{summary['metrics']['median_iou']:<15.4f} "
          f"{summary['metrics']['std_iou']:<15.4f}")
    print("-" * 70)
    print(f"{'Category':<30} {'Mean IoU':<12} {'Count':<8}")
    print("-" * 70)
    for cat, stats in sorted(per_category.items(), key=lambda x: -x[1]['mean_iou']):
        print(f"{cat:<30} {stats['mean_iou']:<12.4f} {stats['count']:<8}")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
