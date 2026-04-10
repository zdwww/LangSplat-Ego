"""Pre-compute Qwen3-VL text embeddings for eval.

Runs in qwen3vl env, produces a .npz file that can be loaded by
eval_novel_views_codebook.py in langsplat_v2 env (avoiding rasterizer
torch version conflicts).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get(
    "QWEN_SRC", "/home/daiwei/Ego3DVQA-GS/Qwen3-VL-Embedding/src"))
from models.qwen3_vl_embedding import Qwen3VLEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_json', type=str, required=True)
    parser.add_argument('--metadata_json', type=str, required=True)
    parser.add_argument('--output', type=str, required=True,
                        help=".npz file to save")
    parser.add_argument('--model_name', type=str,
                        default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('--embed_dim', type=int, default=512)
    args = parser.parse_args()

    # Collect all unique categories from moved_050 detections
    with open(args.captions_json) as f:
        captions = json.load(f)
    with open(args.metadata_json) as f:
        metadata = json.load(f)

    all_categories = set()
    for frame in metadata['frames']:
        fid = frame['filename']
        if fid in captions and 'moved_050' in captions[fid]:
            for det in captions[fid]['moved_050'].get('detections', []):
                all_categories.add(det['category'])

    categories = sorted(all_categories)
    negatives = ["object", "things", "stuff", "texture"]

    print(f"Loading {args.model_name}...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=args.model_name, dtype=torch.bfloat16)
    dim = args.embed_dim

    def encode(text):
        with torch.no_grad():
            emb = embedder.process([{"text": text}])
            emb = emb[:, :dim]
            return F.normalize(emb.float(), dim=-1).cpu().numpy()

    print(f"Encoding {len(categories)} categories + {len(negatives)} negatives...")
    cat_embeds = np.stack([encode(c)[0] for c in categories])   # [N_cats, dim]
    neg_embeds = np.stack([encode(n)[0] for n in negatives])    # [N_negs, dim]

    np.savez(
        args.output,
        categories=np.array(categories, dtype=object),
        category_embeds=cat_embeds,
        negatives=np.array(negatives, dtype=object),
        negative_embeds=neg_embeds,
        embed_dim=dim,
    )
    print(f"Saved {args.output}")
    print(f"  {len(categories)} category embeddings: {cat_embeds.shape}")
    print(f"  {len(negatives)} negative embeddings: {neg_embeds.shape}")


if __name__ == '__main__':
    main()
