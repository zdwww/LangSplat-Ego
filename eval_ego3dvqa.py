"""Qualitative evaluation for LangSplat on Ego3DVQA datasets."""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autoencoder'))

from autoencoder.model import Autoencoder

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip required")


def load_autoencoder(ckpt_path):
    model = Autoencoder([256,128,64,32,3], [16,32,64,128,256,256,512]).cuda()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--prompts', nargs='+', type=str, required=True)
    parser.add_argument('--num_vis_frames', type=int, default=10)
    args = parser.parse_args()

    eval_dir = os.path.join(args.workspace, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)

    print("Loading autoencoder...")
    ae = load_autoencoder(args.ae_ckpt)
    print("Loading CLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16')
    clip_model = clip_model.cuda().eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    negatives = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        neg_embeds = clip_model.encode_text(torch.cat([tokenizer(n) for n in negatives]).cuda()).float()
        neg_embeds = F.normalize(neg_embeds, dim=-1)

    text_embeds = {}
    with torch.no_grad():
        for p in args.prompts:
            e = clip_model.encode_text(tokenizer(p).cuda()).float()
            text_embeds[p] = F.normalize(e, dim=-1)

    levels = [1, 2, 3]
    level_renders = {}
    for lv in levels:
        rd = os.path.join(args.workspace, f'output_{lv}', 'train', 'ours_None', 'renders_npy')
        if os.path.exists(rd):
            files = sorted(f for f in os.listdir(rd) if f.endswith('.npy'))
            level_renders[lv] = (rd, files)
            print(f"Level {lv}: {len(files)} renders")

    if not level_renders:
        print("ERROR: No rendered features found!")
        return

    img_dir = os.path.join(args.workspace, 'images')
    img_files = sorted(os.listdir(img_dir))
    ref_lv = list(level_renders.keys())[0]
    n_renders = len(level_renders[ref_lv][1])
    vis_idx = np.linspace(0, n_renders - 1, min(args.num_vis_frames, n_renders), dtype=int)

    summary = {'prompts': {}}

    for prompt in args.prompts:
        print(f"\nQuery: '{prompt}'")
        pdir = os.path.join(eval_dir, prompt.replace(' ', '_'))
        os.makedirs(pdir, exist_ok=True)
        pstats = {'level_scores': {}, 'best_level': None, 'best_mean': 0}

        for lv in sorted(level_renders.keys()):
            rd, rfiles = level_renders[lv]
            scores = []
            for idx in range(len(rfiles)):
                f3d = np.load(os.path.join(rd, rfiles[idx]))
                flat = torch.from_numpy(f3d.reshape(-1, 3)).float().cuda()
                with torch.no_grad():
                    f512 = F.normalize(ae.decode(flat), dim=-1)
                    rel = get_relevancy(f512, text_embeds[prompt], neg_embeds)
                scores.append(float(rel.max().cpu()))
            mean_s = np.mean(scores)
            pstats['level_scores'][str(lv)] = {'mean': float(mean_s), 'max': float(np.max(scores))}
            if mean_s > pstats['best_mean']:
                pstats['best_mean'] = float(mean_s)
                pstats['best_level'] = lv

        bl = pstats['best_level']
        print(f"  Best level: {bl} (mean={pstats['best_mean']:.4f})")

        rd, rfiles = level_renders[bl]
        for vi in vis_idx:
            f3d = np.load(os.path.join(rd, rfiles[vi]))
            H, W = f3d.shape[:2]
            flat = torch.from_numpy(f3d.reshape(-1, 3)).float().cuda()
            with torch.no_grad():
                f512 = F.normalize(ae.decode(flat), dim=-1)
                rel = get_relevancy(f512, text_embeds[prompt], neg_embeds)
            rel_map = rel.cpu().numpy().reshape(H, W)

            img = np.array(Image.open(os.path.join(img_dir, img_files[vi])).resize((W, H))) if vi < len(img_files) else np.zeros((H, W, 3), dtype=np.uint8)

            # Normalize relevancy to [0,1] using percentile-based range for contrast
            vlo, vhi = np.percentile(rel_map, 2), np.percentile(rel_map, 98)
            if vhi - vlo < 1e-6:
                vhi = vlo + 1e-6
            rel_norm = np.clip((rel_map - vlo) / (vhi - vlo), 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            axes[0].imshow(img); axes[0].set_title('Input'); axes[0].axis('off')
            # Raw relevancy with adaptive range
            im1 = axes[1].imshow(rel_map, cmap='turbo', vmin=vlo, vmax=vhi)
            axes[1].set_title(f'"{prompt}" [{vlo:.3f}, {vhi:.3f}] max={rel_map.max():.3f}')
            axes[1].axis('off'); plt.colorbar(im1, ax=axes[1], fraction=0.046)
            # Percentile-normalized heatmap
            axes[2].imshow(rel_norm, cmap='turbo', vmin=0, vmax=1)
            axes[2].set_title(f'Normalized (2-98 pctl)'); axes[2].axis('off')
            # Overlay: use normalized map for alpha, threshold at 0.5 for selective overlay
            hm = cm.turbo(rel_norm)[:, :, :3]
            alpha = np.clip((rel_norm - 0.5) * 3, 0, 0.8)  # Only highlight top ~50% of normalized range
            overlay = img / 255.0 * (1 - alpha[..., None]) + hm * alpha[..., None]
            axes[3].imshow(np.clip(overlay, 0, 1)); axes[3].set_title(f'Overlay (L{bl})'); axes[3].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(pdir, f'frame_{vi:05d}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        summary['prompts'][prompt] = pstats

    with open(os.path.join(eval_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Query':<25} {'Best Lvl':<10} {'Mean':<10} {'Max':<10}")
    print("-" * 60)
    for p, s in summary['prompts'].items():
        bl = s['best_level']
        print(f"{p:<25} {bl:<10} {s['best_mean']:<10.4f} {s['level_scores'][str(bl)]['max']:<10.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
