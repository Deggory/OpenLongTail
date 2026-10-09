"""Stage-1 training data: the warp sidecar of every clip of a latent cache.

    clip_<id>_warp.pt      the front warped into the cross cameras at the same time
                           {'warped_target_latents': (2,16,11,60,104), 'warped_target_visibility': (2,1,11,60,104),
                            'target_view_ids': [1, 2], ...}

Source data per scene under --data-root (file patterns --front-video / --front-depth, with {uuid} and {sensor} filled in):
    the full front video, and its per-frame metric depth {'depth_sequence': (N,1,h,w), 'source_indices': (N,) video
    frame of each depth map}, e.g. from DepthCrafter.
The clip's frame indices, calibration and output size come from the latent cache.

    python -m openlongtail.scripts.build_warp_sidecars --latent-cache-root CACHE --data-root DATA [--num-shards 8 --shard-index 0]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from openlongtail.config import CROSS_VIEWS, FRONT, SENSOR_ORDER, VAE_PATH
from openlongtail.data.latent_cache import sidecar
from openlongtail.data.warp import encode_warp, visibility_to_latent, warp_same_time
from openlongtail.models.wan import load_vae

FRONT_VIDEO = '{uuid}/{sensor}.mp4'
FRONT_DEPTH = '{uuid}/front_depth.pt'


def load_scene(data_root: Path, uuid: str, video_pattern: str, depth_pattern: str) -> tuple[np.ndarray, torch.Tensor, dict[int, int]]:
    import imageio.v3 as iio
    video = iio.imread(data_root / video_pattern.format(sensor=SENSOR_ORDER[FRONT], uuid=uuid), plugin='pyav')
    depth = torch.load(data_root / depth_pattern.format(sensor=SENSOR_ORDER[FRONT], uuid=uuid), map_location='cpu', weights_only=True)
    return video, depth['depth_sequence'], {int(s): i for i, s in enumerate(depth['source_indices'])}


@torch.no_grad()
def build_clip(clip_path: Path, video: np.ndarray, depth_seq: torch.Tensor, depth_pos: dict[int, int], vae, device: torch.device,
               splat_radius: int) -> dict:
    cache = torch.load(clip_path, map_location='cpu', weights_only=True)
    K, E = cache['K'].float().to(device), cache['E'].float().to(device)
    H, W = cache['output_size']
    frames = cache['frame_indices'][SENSOR_ORDER[FRONT]].numpy()
    front = np.stack([np.array(Image.fromarray(video[int(i)]).resize((W, H), Image.BILINEAR)) for i in frames])
    front = torch.from_numpy(front).permute(0, 3, 1, 2).contiguous().to(device)
    depth = torch.stack([depth_seq[depth_pos[int(i)]] for i in frames]).float()
    depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False).squeeze(1).to(device)
    latents, visibility, coverage = [], [], []
    for view in CROSS_VIEWS:
        rgb, vis = warp_same_time(front, depth, K, E, view, splat_radius)
        latents.append(encode_warp(vae, rgb, device))
        visibility.append(visibility_to_latent(vis.cpu()))
        coverage.append(float(vis.float().mean().cpu()))
    return {'warped_target_latents': torch.stack(latents).contiguous(), 'warped_target_visibility': torch.stack(visibility).contiguous(),
            'visible_fraction': torch.tensor(coverage), 'splat_radius': splat_radius, 'target_view_ids': list(CROSS_VIEWS)}


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latent-cache-root', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--front-video', default=FRONT_VIDEO, help='front video of a scene, relative to --data-root')
    ap.add_argument('--front-depth', default=FRONT_DEPTH, help='front depth of a scene, relative to --data-root')
    ap.add_argument('--vae-path', type=Path, default=VAE_PATH)
    ap.add_argument('--splat-radius', type=int, default=1)
    ap.add_argument('--num-shards', type=int, default=1, help='scenes are split round-robin over shards')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    device = torch.device('cuda')
    vae = load_vae(device, a.vae_path)
    scenes = sorted(p for p in (a.latent_cache_root / 'per_uuid').iterdir() if p.is_dir())[a.shard_index::a.num_shards]
    done, t_start = 0, time.time()
    for scene in scenes:
        clips = [p for p in sorted(scene.glob('clip_*.pt')) if p.stem[5:].isdigit()]
        clips = [p for p in clips if a.overwrite or not sidecar(p, 'warp').exists()]
        if not clips:
            continue
        try:
            video, depth_seq, depth_pos = load_scene(a.data_root, scene.name, a.front_video, a.front_depth)
        except (FileNotFoundError, KeyError) as exc:
            print(f'[skip] {scene.name}: {exc}', flush=True)
            continue
        for clip in clips:
            t0 = time.time()
            warp = build_clip(clip, video, depth_seq, depth_pos, vae, device, a.splat_radius)
            save_atomic(warp, sidecar(clip, 'warp'))
            done += 1
            visibility = [round(100 * v, 1) for v in warp['visible_fraction'].tolist()]
            print(f'[ok] {scene.name}/{clip.stem} {time.time() - t0:.1f}s  visibility {visibility}', flush=True)
    print(f'done: {done} clips in {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
