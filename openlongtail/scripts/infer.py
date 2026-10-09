"""Generate the five surround cameras of each clip: stage 1 renders cross-left and cross-right from the depth warp of the
front, stage 2 the three rear cameras from the ground prior built from the front and the stage-1 views. One model, both
stages, one process.

Raw tier (default): clips in the openlongtail.data.dashcam layout; the other cameras are generated on the reference rig,
the depth warp is computed live and the caption (if any) is encoded with UMT5.
Cached tier (--cached): latent-cache clips with their own calibration, warp sidecar and caption embedding.

Writes <output>/<clip>/: pred_{cross_left,cross_right,rear_left,rear_right,rear_tele}.mp4, front_input.mp4, surround.mp4
and done.json (+ gt_<view>.mp4 and pred_latents.pt on the cached tier).

    python -m openlongtail.scripts.infer --input CLIPS --output OUT
    python -m openlongtail.scripts.infer --input demo/cached --output OUT --cached --first-clip-per-uuid
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from openlongtail.config import CROSS_VIEWS, MODEL_CKPT, REAR_VIEWS, REFERENCE_RIG, VIEW_NAMES, WAN21_VACE_DIR
from openlongtail.data.dashcam import collect_clips, load_clip, load_rig, read_caption
from openlongtail.data.ground_prior import Segmenter
from openlongtail.data.latent_cache import caption_embedding, list_cache_clips, null_text_embedding, sidecar
from openlongtail.data.warp import live_warp, load_warp
from openlongtail.inference.sample import cross_noise, prior_condition, rear_noise, sample_cross, sample_rear
from openlongtail.inference.video import frames_u8, write_mp4, write_surround
from openlongtail.models.surround import load_model
from openlongtail.models.wan import UMT5Encoder, load_vae, vae_decode_batch, vae_encode_video


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True, help='raw clip root, or a latent-cache root with --cached')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cached', action='store_true')
    ap.add_argument('--ckpt', type=Path, default=MODEL_CKPT, help='checkpoint dir (trainable.pt + metadata.json)')
    ap.add_argument('--wan21-vace-dir', type=Path, default=WAN21_VACE_DIR)
    ap.add_argument('--reference-rig', type=Path, default=REFERENCE_RIG, help='raw tier: K/E of the generated rig')
    ap.add_argument('--caption-cache', type=Path, default=None, help='raw tier: dir with per_uuid/<uuid>.txt captions')
    ap.add_argument('--caption', type=str, default=None, help='raw tier: one caption for every clip')
    ap.add_argument('--num-steps', type=int, default=50)
    ap.add_argument('--cross-guide', type=float, default=3.5, help='CFG scale of stage 1')
    ap.add_argument('--rear-guide', type=float, default=3.5, help='CFG scale of stage 2')
    ap.add_argument('--seed', type=int, default=20260710, help='stage-1 noise; stage 2 uses seed + 1')
    ap.add_argument('--splat-radius', type=int, default=1, help='raw tier: depth-warp splat footprint')
    ap.add_argument('--max-clips', type=int, default=-1)
    ap.add_argument('--first-clip-per-uuid', action='store_true', help='cached tier: one clip per scene')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    device = torch.device('cuda')
    if a.cached:
        cached = list_cache_clips(a.input, a.max_clips, a.shard_index, a.num_shards, a.first_clip_per_uuid)
        clips = [(f'{uuid}/{cid}', path) for uuid, cid, path in cached]
    else:
        clips = [(str(c.relative_to(a.input)), c) for c in collect_clips(a.input, a.max_clips, a.shard_index, a.num_shards)]
        K_rig, E_rig = load_rig(a.reference_rig)
    model, meta = load_model(a.ckpt, device, a.wan21_vace_dir)
    print(f'checkpoint {a.ckpt} (step {meta.get("step")})', flush=True)
    vae = load_vae(device)
    seg = Segmenter(device)
    null_emb = null_text_embedding(a.input if a.cached else None).to(device)
    umt5 = None
    for idx, (name, src) in enumerate(clips, start=1):
        out = a.output / name
        if (out / 'done.json').exists() and not a.overwrite:
            continue
        t0 = time.time()
        info: dict = {'clip': name, 'num_steps': a.num_steps, 'cross_guide': a.cross_guide, 'rear_guide': a.rear_guide, 'seed': a.seed}
        if a.cached:
            if not sidecar(src, 'warp').exists():
                print(f'[{idx}/{len(clips)}] {name}: no warp sidecar, skipped', flush=True)
                continue
            payload = torch.load(src, map_location='cpu', weights_only=True)
            z_all = payload['z_all'].to(device=device, dtype=torch.bfloat16)
            z_front, K, E = z_all[0], payload['K'].float().to(device), payload['E'].float().to(device)
            T_front = torch.load(sidecar(src, 'p4'), map_location='cpu', weights_only=True)['T_anchor_front'].float().to(device)
            warp_latents, warp_visibility = load_warp(sidecar(src, 'warp'), device)
            text_emb = caption_embedding(a.input, name.split('/')[0], null_emb).to(device)
            fps = 16
        else:
            front, depth, pose, clip_meta = load_clip(src)
            K, E, T_front = K_rig.to(device), E_rig.to(device), pose.to(device)
            warp_latents, warp_visibility, info['warp_visibility_pct'] = live_warp(front, depth, K, E, vae, device, a.splat_radius)
            caption = a.caption or read_caption(a.caption_cache, str(clip_meta.get('uuid', '')))
            if caption and umt5 is None:
                umt5 = UMT5Encoder('cuda')
            text_emb = umt5.encode(caption)[0].to(device) if caption else null_emb
            info['caption'] = caption
            with torch.no_grad():
                z_front = vae_encode_video(vae, front.to(device)).to(device, torch.bfloat16)
            fps = int(clip_meta.get('src_fps', 16))
        cross = sample_cross(model, z_front, K, E, T_front, text_emb, null_emb, warp_latents, warp_visibility,
                             cross_noise(z_front.shape, a.seed, device), num_steps=a.num_steps, guide_scale=a.cross_guide)
        if a.cached:
            src_rgb = torch.stack(vae_decode_batch(vae, [z_front, cross[1], cross[2]], device))
        else:
            src_rgb = torch.stack([front.to(device).float(), *vae_decode_batch(vae, [cross[1], cross[2]], device)])
        ground_latents, ground_visibility, info['prior_coverage'] = prior_condition(src_rgb, K, E, T_front, seg, vae)
        rear = sample_rear(model, z_front, K, E, T_front, text_emb, null_emb, ground_latents, ground_visibility,
                           rear_noise(z_front.shape, a.seed + 1, device), num_steps=a.num_steps, guide_scale=a.rear_guide)

        out.mkdir(parents=True, exist_ok=True)
        videos = {VIEW_NAMES[v]: frames_u8(src_rgb[i]) for i, v in enumerate(CROSS_VIEWS, start=1)}
        videos |= {VIEW_NAMES[v]: frames_u8(rgb) for v, rgb in zip(REAR_VIEWS, vae_decode_batch(vae, [rear[v] for v in REAR_VIEWS], device))}
        for view_name, frames in videos.items():
            write_mp4(out / f'pred_{view_name}.mp4', frames, fps)
        front_rgb = frames_u8(src_rgb[0])
        write_mp4(out / 'front_input.mp4', front_rgb, fps)
        write_surround(out / 'surround.mp4', front_rgb, videos, fps)
        if a.cached:
            torch.save({v: lat.detach().cpu() for v, lat in sorted((cross | rear).items())}, out / 'pred_latents.pt')
            gt = vae_decode_batch(vae, [z_all[v] for v in CROSS_VIEWS + REAR_VIEWS], device)
            for v, rgb in zip(CROSS_VIEWS + REAR_VIEWS, gt):
                write_mp4(out / f'gt_{VIEW_NAMES[v]}.mp4', frames_u8(rgb), fps)
        info['sec'] = time.time() - t0
        (out / 'done.json').write_text(json.dumps(info, indent=1))
        cov = ' '.join(f"{VIEW_NAMES[v]} {c['history']:.2f}+{c['flip']:.2f}" for v, c in info['prior_coverage'].items())
        print(f'[{idx}/{len(clips)}] {name}  {info["sec"]:.0f}s  ground prior history+flip: {cov}', flush=True)
        torch.cuda.empty_cache()
    print(f'DONE: {len(clips)} clips -> {a.output}', flush=True)


if __name__ == '__main__':
    main()
