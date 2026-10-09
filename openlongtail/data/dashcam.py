"""Raw clips (a front video with its depth and ego poses), the input of the raw inference tier.

    <root>/chunk_<c>/<uuid>/<clip_id>/
        front.mp4        41 RGB frames at 480x832
        front_depth.pt   {'depth_sequence': (41,1,H,W)} metric depth, or inverse depth normalised to [0,1]
        pose.pt          {'T_anchor_front': (11,4,4)} front-camera poses at the 11 latent frames (OpenCV, +Z forward)
        meta.pt          {'uuid', 'src_fps', ...}
    <root>/manifest_clips.jsonl   optional, one {"chunk", "uuid", "clip_id"} per line (sets the clip order)

tools/prepare_dashcam.py builds this layout from plain mp4 files. The other five cameras are generated on a reference
rig (assets/reference_rig.pt: K (6,3,3), E (6,4,4) of the training cameras).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from openlongtail.config import NUM_FRAMES

CLIP_FILES = ('front.mp4', 'front_depth.pt', 'pose.pt', 'meta.pt')


def collect_clips(root: Path, max_clips: int = -1, shard_index: int = 0, num_shards: int = 1) -> list[Path]:
    manifest = root / 'manifest_clips.jsonl'
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        clips = [root / f"chunk_{int(r['chunk']):03d}" / str(r['uuid']) / str(r['clip_id']) for r in rows]
    else:
        clips = sorted(root.glob('chunk_*/*/clip_*'))
    clips = [c for c in clips if all((c / name).exists() for name in CLIP_FILES)][shard_index::num_shards]
    return clips[:max_clips] if max_clips > 0 else clips


def read_front(path: Path) -> torch.Tensor:
    """-> (41,3,H,W) uint8."""
    import imageio.v3 as iio
    frames = iio.imread(path, plugin='pyav')[:NUM_FRAMES]
    if frames.shape[0] != NUM_FRAMES:
        raise ValueError(f'expected {NUM_FRAMES} frames in {path}, got {frames.shape[0]}')
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(torch.uint8)


def read_depth(path: Path, out_h: int, out_w: int) -> torch.Tensor:
    """-> (41,H,W) metres. Inverse depth normalised to [0,1] (max < 5) is converted to pseudo-metric depth."""
    depth = torch.load(path, map_location='cpu', weights_only=True)['depth_sequence'].float()
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if float(depth.max()) < 5.0:
        depth = (1.0 / depth.clamp(min=0.01, max=1.0)).clamp(max=100.0)
    return F.interpolate(depth[:NUM_FRAMES], size=(out_h, out_w), mode='bilinear', align_corners=False).squeeze(1)


def load_clip(clip_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """-> front (41,3,H,W) uint8, depth (41,H,W), T_anchor_front (11,4,4), meta."""
    front = read_front(clip_dir / 'front.mp4')
    depth = read_depth(clip_dir / 'front_depth.pt', front.shape[-2], front.shape[-1])
    pose = torch.load(clip_dir / 'pose.pt', map_location='cpu', weights_only=True)['T_anchor_front'].float()
    return front, depth, pose, torch.load(clip_dir / 'meta.pt', map_location='cpu', weights_only=True)


def load_rig(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return payload['K'].float(), payload['E'].float()


def read_caption(caption_dir: Path | None, uuid: str) -> str:
    path = caption_dir / 'per_uuid' / f'{uuid}.txt' if caption_dir is not None else None
    return path.read_text().strip() if path is not None and path.exists() else ''
