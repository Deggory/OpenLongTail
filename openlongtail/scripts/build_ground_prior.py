"""Stage-2 training data: per clip of a latent cache (with clip_<id>_p4.pt poses)

    clip_<id>_gprior.pt    ground prior of the three rear cameras from the ground-truth front / cross-left / cross-right
                           {'prior_latents': (3,16,11,60,104) bf16, 'prior_visibility': (3,1,11,60,104) fp16}
    clip_<id>_rearlane.pt  lane-marking mask of the ground-truth rear views {'lane': (3,1,11,60,104) bool}

Rear cameras are stored in the order rear-left, rear-right, rear-tele.

    python -m openlongtail.scripts.build_ground_prior --latent-cache-root CACHE [--num-shards 8 --shard-index 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from openlongtail.config import VAE_PATH
from openlongtail.data.ground_prior import KEY_RGB, Segmenter, ground_prior
from openlongtail.data.latent_cache import read_index, sidecar
from openlongtail.models.wan import load_vae, vae_decode_batch, vae_encode_batch


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.no_grad()
def build_clip(clip_path: Path, vae, seg: Segmenter, device: torch.device, decode_batch: int, seg_batch: int) -> None:
    payload = torch.load(clip_path, map_location='cpu', weights_only=True)
    K, E, z = payload['K'].float().to(device), payload['E'].float().to(device), payload['z_all']
    T_front = torch.load(sidecar(clip_path, 'p4'), map_location='cpu', weights_only=True)['T_anchor_front'].float()
    if not sidecar(clip_path, 'gprior').exists():
        src = torch.stack(vae_decode_batch(vae, [z[0], z[1], z[2]], device, batch=decode_batch))
        rgbs, vis, coverage = ground_prior(src, K, E, T_front, seg, seg_batch=seg_batch)
        save_atomic({'prior_latents': vae_encode_batch(vae, rgbs, batch=decode_batch).to('cpu', torch.bfloat16),
                     'prior_visibility': torch.stack([v.to('cpu', torch.float16) for v in vis]), 'coverage': coverage}, sidecar(clip_path, 'gprior'))
    if not sidecar(clip_path, 'rearlane').exists():
        rear = torch.stack(vae_decode_batch(vae, [z[3], z[4], z[5]], device, batch=decode_batch))[:, KEY_RGB]
        cls = seg(rear.flatten(0, 1), batch=seg_batch).view(3, len(KEY_RGB), *rear.shape[3:])
        lane = F.adaptive_max_pool2d(torch.isin(cls, seg.lane_ids).float().flatten(0, 1)[:, None], (60, 104)).view(3, len(KEY_RGB), 1, 60, 104)
        save_atomic({'lane': lane.permute(0, 2, 1, 3, 4).bool().cpu()}, sidecar(clip_path, 'rearlane'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latent-cache-root', type=Path, required=True)
    ap.add_argument('--index-name', type=str, default='index.jsonl', help='clip list in the cache root (default: every per_uuid/*/clip_<n>.pt)')
    ap.add_argument('--vae-path', type=Path, default=VAE_PATH)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--decode-batch', type=int, default=3, help='videos per VAE call')
    ap.add_argument('--seg-batch', type=int, default=12, help='frames per segmenter call')
    ap.add_argument('--failed-list', type=Path, default=None, help='append clips that fail here and continue (default: stop)')
    a = ap.parse_args()
    device = torch.device('cuda')
    vae = load_vae(device, a.vae_path)
    seg = Segmenter(device)
    root = a.latent_cache_root
    clips = [root / e['path'] for e in read_index(root, a.index_name)][a.shard_index::a.num_shards]
    done, t0 = 0, time.time()
    for n, clip in enumerate(clips, 1):
        if (sidecar(clip, 'gprior').exists() and sidecar(clip, 'rearlane').exists()) or not sidecar(clip, 'p4').exists():
            continue
        try:
            build_clip(clip, vae, seg, device, a.decode_batch, a.seg_batch)
        except Exception as exc:                                    # one broken clip must not stop a large run
            if a.failed_list is None:
                raise
            print(f'[shard {a.shard_index}] FAILED {clip}: {type(exc).__name__}: {exc}', flush=True)
            with a.failed_list.open('a') as h:
                h.write(json.dumps({'path': str(clip), 'error': f'{type(exc).__name__}: {exc}'[:300]}) + '\n')
            torch.cuda.empty_cache()
            continue
        done += 1
        if done % 25 == 0:
            print(f'[shard {a.shard_index}] {n}/{len(clips)}: built {done} ({(time.time() - t0) / done:.1f} s/clip)', flush=True)
    print(f'done: shard {a.shard_index} built {done} clips', flush=True)


if __name__ == '__main__':
    main()
