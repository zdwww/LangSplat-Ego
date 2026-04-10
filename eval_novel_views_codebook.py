"""Novel-view segmentation evaluation using LangSplatV2 codebook decode.

Same eval protocol as eval_novel_views.py, but replaces autoencoder decode
with codebook-based feature recovery:
  Render 64-ch weight map → codebook^T @ weights → 512D → relevancy → IoU

Requires LangSplatV2 repo (GaussianModel + CUDA rasterizer) via --langsplatv2_dir.
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
import matplotlib.patches as patches

try:
    import open_clip
except ImportError:
    open_clip = None


# ---------------------------------------------------------------------------
# Relevancy + metrics + visualization (identical to eval_novel_views.py)
# ---------------------------------------------------------------------------

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


def compute_iou(pred_mask, gt_mask):
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
        'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1,
        'pred_pixels': pred_pixels, 'gt_pixels': gt_pixels,
        'intersection': int(intersection), 'union': int(union),
    }


def create_visualization(rgb, rel_map, pred_mask, gt_mask, category, metrics, bbox, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    axes[0].imshow(rgb)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title(f'"{category}"')
    axes[0].axis('off')

    im = axes[1].imshow(rel_map, cmap='turbo', vmin=0, vmax=1)
    axes[1].set_title(f'Relevancy max={rel_map.max():.3f}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(pred_mask.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Pred ({pred_mask.sum()} px)')
    axes[2].axis('off')

    axes[3].imshow(gt_mask.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f'GT ({gt_mask.sum()} px)')
    axes[3].axis('off')

    overlay = rgb.copy().astype(np.float32) / 255.0
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, ~gt_mask)
    fn = np.logical_and(~pred_mask, gt_mask)
    alpha = 0.5
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha
    axes[4].imshow(np.clip(overlay, 0, 1))
    axes[4].set_title(f'IoU={metrics["iou"]:.3f} P={metrics["precision"]:.2f} R={metrics["recall"]:.2f}')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# LangSplatV2 model loading
# ---------------------------------------------------------------------------

def setup_langsplatv2_imports(langsplatv2_dir):
    """Add LangSplatV2 to sys.path so its modules (GaussianModel, render, etc.)
    take priority over our LangSplat modules."""
    # Prioritize LangSplatV2 modules
    sys.path.insert(0, langsplatv2_dir)
    # Ensure the efficient rasterizer is found
    rast_build = os.path.join(
        langsplatv2_dir,
        'submodules/efficient-langsplat-rasterization/build/lib.linux-x86_64-cpython-39')
    if os.path.isdir(rast_build):
        sys.path.insert(0, rast_build)


def load_codebook_model(workspace, langsplatv2_dir):
    """Load LangSplatV2 GaussianModel with codebook from checkpoint."""
    from scene.gaussian_model import GaussianModel
    from arguments import OptimizationParams
    from argparse import ArgumentParser

    # Find checkpoint (10000 iterations is the default for LangSplatV2)
    ckpt_path = os.path.join(workspace, 'output_1', 'chkpnt10000.pth')
    if not os.path.exists(ckpt_path):
        # Fall back to 30000
        ckpt_path = os.path.join(workspace, 'output_1', 'chkpnt30000.pth')
    print(f"Loading codebook checkpoint: {ckpt_path}")
    model_params, first_iter = torch.load(ckpt_path)

    gaussians = GaussianModel(sh_degree=3)

    # Create training_args for restore (needed for mode='test' path)
    opt = SimpleNamespace(
        include_feature=True,
        quick_render=False,
        vq_layer_num=1,
        codebook_size=64,
    )
    gaussians.restore(model_params, opt, mode='test')

    n_gauss = gaussians.get_xyz.shape[0]
    cb_shape = gaussians._language_feature_codebooks.shape
    print(f"  Loaded {n_gauss} Gaussians, codebook shape: {cb_shape}")
    return gaussians


# ---------------------------------------------------------------------------
# Camera handling (same as eval_novel_views.py)
# ---------------------------------------------------------------------------

def load_moved_cameras(metadata_json):
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
    from scene.cameras import Camera
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--langsplatv2_dir', type=str,
                        default='/home/daiwei/LangSplat-variants/LangSplatV2',
                        help="Path to LangSplatV2 repo (for GaussianModel and rasterizer)")
    parser.add_argument('--metadata_json', type=str, required=True)
    parser.add_argument('--captions_json', type=str, required=True)
    parser.add_argument('--gt_masks_dir', type=str, required=True)
    parser.add_argument('--moved_rgb_dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_vis_frames', type=int, default=10)
    parser.add_argument('--encoder_type', type=str, default='clip',
                        choices=['clip', 'qwen3vl', 'precomputed'])
    parser.add_argument('--qwen3vl_model', type=str,
                        default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('--precomputed_text_embeds', type=str, default=None,
                        help="Path to .npz file with pre-computed text embeddings "
                             "(for --encoder_type precomputed)")
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--topk', type=int, default=4,
                        help="Top-k for sparse codebook weights during rendering")
    args = parser.parse_args()

    output_dir = os.path.join(args.workspace, 'eval_novel_results')
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- Setup LangSplatV2 imports ----
    setup_langsplatv2_imports(args.langsplatv2_dir)
    from gaussian_renderer import render

    # ---- Load codebook model ----
    print("Loading codebook Gaussian model...")
    gaussians = load_codebook_model(args.workspace, args.langsplatv2_dir)

    # ---- Load text encoder ----
    negatives = ["object", "things", "stuff", "texture"]

    if args.encoder_type == "precomputed":
        # Load pre-computed text embeddings from .npz
        assert args.precomputed_text_embeds, "--precomputed_text_embeds required"
        print(f"Loading precomputed text embeddings: {args.precomputed_text_embeds}")
        npz = np.load(args.precomputed_text_embeds, allow_pickle=True)
        precomp_categories = list(npz['categories'])
        precomp_cat_embeds = torch.from_numpy(npz['category_embeds']).float().cuda()
        precomp_negatives = list(npz['negatives'])
        precomp_neg_embeds = torch.from_numpy(npz['negative_embeds']).float().cuda()
        assert precomp_negatives == negatives, \
            f"Negatives mismatch: {precomp_negatives} vs {negatives}"
        neg_embeds = F.normalize(precomp_neg_embeds, dim=-1)
        _precomp_map = {c: precomp_cat_embeds[i:i+1] for i, c in enumerate(precomp_categories)}

        def encode_text(text):
            if text in _precomp_map:
                return F.normalize(_precomp_map[text], dim=-1)
            raise KeyError(f"Category '{text}' not in precomputed embeddings")

    elif args.encoder_type == "qwen3vl":
        qwen_src = os.environ.get("QWEN_SRC",
            "/home/daiwei/Ego3DVQA-GS/Qwen3-VL-Embedding/src")
        sys.path.insert(0, qwen_src)
        from models.qwen3_vl_embedding import Qwen3VLEmbedder
        print(f"Loading {args.qwen3vl_model}...")
        qwen_embedder = Qwen3VLEmbedder(
            model_name_or_path=args.qwen3vl_model, dtype=torch.bfloat16)
        dim = args.embed_dim

        def encode_text(text):
            emb = qwen_embedder.process([{"text": text}])
            emb = emb[:, :dim]
            return F.normalize(emb.float(), dim=-1)

        with torch.no_grad():
            neg_embeds = torch.cat([encode_text(n) for n in negatives], dim=0)
    else:
        if open_clip is None:
            raise ImportError("open_clip required for --encoder_type clip")
        print("Loading CLIP...")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16')
        clip_model = clip_model.cuda().eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-16')

        def encode_text(text):
            with torch.no_grad():
                e = clip_model.encode_text(tokenizer(text).cuda()).float()
            return F.normalize(e, dim=-1)

        with torch.no_grad():
            neg_embeds = torch.cat([encode_text(n) for n in negatives], dim=0)

    # Rendering params (LangSplatV2 render expects opt with include_feature and quick_render)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    opt = SimpleNamespace(include_feature=True, quick_render=False, topk=args.topk)
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
            text_embeds[cat] = encode_text(cat)

    vis_indices = set(np.linspace(0, len(camera_list) - 1,
                                  min(args.num_vis_frames, len(camera_list)),
                                  dtype=int).tolist())

    # ---- Evaluate ----
    all_results = {}
    all_ious = []

    for cam_idx, cam_params in enumerate(tqdm(camera_list, desc="Evaluating novel views")):
        frame_id = cam_params['frame_id']

        if frame_id not in captions or 'moved_050' not in captions[frame_id]:
            continue
        detections = captions[frame_id]['moved_050'].get('detections', [])
        if not detections or frame_id not in gt_segments:
            continue

        gt_frame = gt_segments[frame_id]

        # 1. Create camera and render codebook weight map
        camera = create_camera(cam_params)
        with torch.no_grad():
            output = render(camera, gaussians, pipe, bg, opt)
            rgb_tensor = output["render"]                        # (3, H, W)
            weight_map = output["language_feature_weight_map"]   # (64, H, W)

        H, W = weight_map.shape[1:]

        # 2. Decode codebook weights → 512D features
        with torch.no_grad():
            feat_512_map = gaussians.compute_final_feature_map(weight_map)  # (512, H, W)
            flat = feat_512_map.permute(1, 2, 0).reshape(-1, 512)           # (H*W, 512)
            feat_512 = F.normalize(flat, dim=-1)

        # Load RGB for visualization
        rgb_path = os.path.join(args.moved_rgb_dir, f"{frame_id}.jpg")
        if os.path.exists(rgb_path):
            rgb_img = np.array(Image.open(rgb_path))
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

            gt_seg = None
            for seg in gt_frame['segments']:
                if seg['obj_idx'] == obj_idx:
                    gt_seg = seg
                    break
            if gt_seg is None:
                continue

            with torch.no_grad():
                rel = get_relevancy(feat_512, text_embeds[category], neg_embeds)
            rel_map = rel.cpu().numpy().reshape(H, W)
            pred_mask = rel_map > args.threshold

            gt_mask_path = os.path.join(args.gt_masks_dir, gt_seg['mask_file'])
            gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is None:
                print(f"WARNING: Could not read {gt_mask_path}, skipping")
                continue
            if gt_mask_raw.shape[:2] != (H, W):
                gt_mask_raw = cv2.resize(gt_mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_mask = gt_mask_raw > 127

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

        del output, rgb_tensor, weight_map, feat_512_map, flat, feat_512
        torch.cuda.empty_cache()

    # ---- Aggregate metrics ----
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
            'decode_method': 'codebook',
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
    print(f"Novel View Evaluation (Codebook): {args.workspace}")
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
