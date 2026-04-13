"""Pre-compute CLIP ViT-B-16 text embeddings so v2/v3 novel-view evals can run
in environments without open_clip (e.g. qwen3vl) while still using CLIP for
text queries. Produces a .npz compatible with
`eval_novel_views.py --encoder_type precomputed`.

This is the CLIP analogue of precompute_qwen3vl_text_embeds.py — same .npz
schema (`categories`, `category_embeds`, `negatives`, `negative_embeds`,
`embed_dim`).
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import open_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_json', type=str, required=True)
    parser.add_argument('--metadata_json', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default='ViT-B-16')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--embed_dim', type=int, default=512)
    args = parser.parse_args()

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

    print(f"Loading {args.clip_model} ({args.clip_pretrained})...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, precision='fp16')
    clip_model = clip_model.cuda().eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    dim = args.embed_dim

    def encode(text):
        with torch.no_grad():
            e = clip_model.encode_text(tokenizer(text).cuda()).float()
        e = e[:, :dim]
        return F.normalize(e, dim=-1).cpu().numpy()

    print(f"Encoding {len(categories)} categories + {len(negatives)} negatives...")
    cat_embeds = np.stack([encode(c)[0] for c in categories])
    neg_embeds = np.stack([encode(n)[0] for n in negatives])

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
