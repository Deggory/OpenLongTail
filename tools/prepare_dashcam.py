"""Turn plain front-camera videos into raw clips for inference (the openlongtail.data.dashcam layout).

Per video: 41 consecutive frames from the middle, centre-cropped to 832:480 and resized to 480x832; Depth-Anything-V2
inverse depth (normalised to [0,1]; the loader turns it into pseudo-metric depth); and either a constant-velocity
forward pose (--pose-mode synthetic) or no pose, to be filled in by tools/estimate_poses.py (--pose-mode mapanything,
see scripts/demo.sh). There is no calibration: the other cameras are generated on the reference rig.

    python tools/prepare_dashcam.py --video-root VIDEOS --out-root CLIPS --num-clips 20
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from openlongtail.inference.video import write_mp4

CLIP_FRAMES = 41
OUT_H, OUT_W = 480, 832
FORWARD_METRES = 10.0          # synthetic forward travel over the clip


def select_frames(n_frames: int) -> list[int]:
    """41 consecutive frames from the middle of the video (the last frame repeated if it is shorter)."""
    if n_frames >= CLIP_FRAMES:
        start = (n_frames - CLIP_FRAMES) // 2
        return list(range(start, start + CLIP_FRAMES))
    return list(range(n_frames)) + [n_frames - 1] * (CLIP_FRAMES - n_frames)


def load_video(path: Path) -> tuple[torch.Tensor, float]:
    """-> (41,3,480,832) uint8 and the source fps."""
    import imageio.v3 as iio
    fps = float(iio.immeta(path, plugin='pyav').get('fps', 30.0) or 30.0)
    frames = iio.imread(path, plugin='pyav')
    x = torch.from_numpy(frames[select_frames(frames.shape[0])]).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    if w / h > OUT_W / OUT_H:
        crop = int(round(h * OUT_W / OUT_H))
        x = x[..., (w - crop) // 2:(w - crop) // 2 + crop]
    else:
        crop = int(round(w * OUT_H / OUT_W))
        x = x[..., (h - crop) // 2:(h - crop) // 2 + crop, :]
    x = F.interpolate(x, size=(OUT_H, OUT_W), mode='bilinear', align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8), fps


@torch.no_grad()
def estimate_depth(frames: torch.Tensor, model, processor, device) -> torch.Tensor:
    """Depth-Anything-V2 inverse depth normalised to [0,1], (41,1,480,832) fp16."""
    inputs = processor(images=[f.permute(1, 2, 0).numpy() for f in frames], return_tensors='pt').to(device, torch.float16)
    pred = F.interpolate(model(**inputs).predicted_depth.float().unsqueeze(1), size=(OUT_H, OUT_W), mode='bilinear', align_corners=False).clamp(min=0)
    return (pred / (pred.max() + 1e-6)).to(torch.float16).cpu()


def synthetic_pose() -> torch.Tensor:
    """(11,4,4) constant-velocity forward motion along +Z, frame 0 = identity."""
    T = torch.eye(4).repeat(11, 1, 1)
    T[:, 2, 3] = FORWARD_METRES * torch.arange(11) / 10.0
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video-root', type=Path, required=True, help='directory searched recursively for *.mp4')
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--num-clips', type=int, default=-1, help='-1 = all')
    ap.add_argument('--include', type=str, default=None, help='only videos whose path contains this substring')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--depth-model', type=str, default='depth-anything/Depth-Anything-V2-Large-hf')
    ap.add_argument('--pose-mode', choices=('synthetic', 'mapanything'), default='synthetic',
                    help='mapanything: write the frames as PNGs for tools/estimate_poses.py, which writes pose.pt')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    videos = sorted(p for p in a.video_root.rglob('*.mp4') if a.include is None or a.include in str(p))[a.shard_index::a.num_shards]
    if a.num_clips > 0:
        videos = videos[:a.num_clips]
    if not videos:
        raise SystemExit(f'no mp4 under {a.video_root}')
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = torch.device('cuda')
    processor = AutoImageProcessor.from_pretrained(a.depth_model)
    model = AutoModelForDepthEstimation.from_pretrained(a.depth_model, torch_dtype=torch.float16).to(device).eval()
    manifest = []
    for i, video in enumerate(videos, 1):
        uuid = '_'.join(video.relative_to(a.video_root).with_suffix('').parts)
        clip = a.out_root / 'chunk_000' / uuid / 'clip_000000'
        manifest.append(json.dumps({'chunk': 0, 'uuid': uuid, 'clip_id': 'clip_000000'}))
        if (clip / 'meta.pt').exists() and not a.overwrite:
            continue
        shutil.rmtree(clip, ignore_errors=True)
        clip.mkdir(parents=True)
        frames, fps = load_video(video)
        write_mp4(clip / 'front.mp4', frames, max(1, int(round(fps))))
        depth = estimate_depth(frames, model, processor, device)
        torch.save({'depth_sequence': depth, 'source_indices': torch.arange(CLIP_FRAMES)}, clip / 'front_depth.pt')
        if a.pose_mode == 'synthetic':
            torch.save({'T_anchor_front': synthetic_pose()}, clip / 'pose.pt')
        else:
            from PIL import Image
            (clip / 'frames').mkdir()
            for j, frame in enumerate(frames):
                Image.fromarray(frame.permute(1, 2, 0).numpy()).save(clip / 'frames' / f'frame_{j:03d}.png')
        torch.save({'uuid': uuid, 'src_fps': int(round(fps)), 'source_video': str(video)}, clip / 'meta.pt')
        print(f'[{i}/{len(videos)}] {uuid}  fps={fps:.1f}', flush=True)
    if a.num_shards == 1:                       # shards would overwrite each other's list; without it the clips are globbed
        (a.out_root / 'manifest_clips.jsonl').write_text('\n'.join(manifest) + '\n')
    print(f'DONE: {len(manifest)} clips -> {a.out_root}', flush=True)


if __name__ == '__main__':
    main()
