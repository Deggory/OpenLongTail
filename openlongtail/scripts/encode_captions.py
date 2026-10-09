"""UMT5 caption embeddings of a latent cache: <cache>/text_cache/null.pt (empty prompt) and per_uuid/<uuid>.pt.

Captions are read from <captions>/per_uuid/<uuid>.txt (the layout build_captions writes) or, for dataset scenes,
from <captions>/<uuid>/vlm_caption.txt.

    python -m openlongtail.scripts.encode_captions --captions CAPTIONS --output-root CACHE/text_cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openlongtail.models.wan import UMT5Encoder


def read_captions(root: Path) -> dict[str, str]:
    if (root / 'per_uuid').is_dir():
        return {p.stem: p.read_text().strip() for p in sorted((root / 'per_uuid').glob('*.txt'))}
    return {d.name: (d / 'vlm_caption.txt').read_text().strip() for d in sorted(root.iterdir()) if (d / 'vlm_caption.txt').exists()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--captions', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    a = ap.parse_args()
    encoder = UMT5Encoder('cuda')
    (a.output_root / 'per_uuid').mkdir(parents=True, exist_ok=True)
    text_emb, text_mask = encoder.encode('')
    torch.save({'text_emb': text_emb, 'text_mask': text_mask}, a.output_root / 'null.pt')
    captions = read_captions(a.captions)
    for uuid, caption in captions.items():
        text_emb, text_mask = encoder.encode(caption)
        torch.save({'text_emb': text_emb, 'text_mask': text_mask, 'prompt': caption}, a.output_root / 'per_uuid' / f'{uuid}.pt')
    print(f'encoded {len(captions)} captions -> {a.output_root}', flush=True)


if __name__ == '__main__':
    main()
