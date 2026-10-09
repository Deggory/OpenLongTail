"""Ego poses of a raw clip with MapAnything, for tools/prepare_dashcam.py --pose-mode mapanything.

MapAnything predicts metric camera-to-world poses in OpenCV axes (+X right, +Y down, +Z forward) for an ordered image
sequence. The front camera at the 11 latent frames (RGB frames 0, 4, ..., 40) relative to frame 0 is written to pose.pt
as {'T_anchor_front': (11,4,4)}.

    python tools/estimate_poses.py --frames CLIP/frames --out CLIP/pose.pt

Needs the mapanything package (https://github.com/facebookresearch/map-anything).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ANCHOR_FRAMES = list(range(0, 41, 4))          # RGB frame of each latent frame
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def predict_poses(frames: list[Path], model_id: str, minibatch_size: int) -> torch.Tensor:
    """-> (N,4,4) camera-to-world poses of the ordered frames."""
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MapAnything.from_pretrained(model_id).to(device).eval()
    views = load_images([str(f) for f in frames], resize_mode='fixed_mapping', resolution_set=518, verbose=False)
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, minibatch_size=minibatch_size, use_amp=True, amp_dtype='bf16',
                            apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    return torch.stack([pred['camera_poses'][0].float().cpu() for pred in preds])


def anchor_poses(poses_c2w: torch.Tensor) -> torch.Tensor:
    """(N,4,4) camera-to-world poses of N >= 41 frames -> the anchor frames relative to frame 0 (11,4,4)."""
    if poses_c2w.shape[0] <= ANCHOR_FRAMES[-1]:
        raise ValueError(f'need at least {ANCHOR_FRAMES[-1] + 1} poses, got {poses_c2w.shape[0]}')
    P = poses_c2w.double()[ANCHOR_FRAMES]
    T = torch.linalg.inv(P[0]).unsqueeze(0) @ P
    T[0] = torch.eye(4, dtype=torch.float64)
    return T.float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', type=Path, required=True, help='directory of the clip frames, sorted by file name')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='facebook/map-anything-apache', help='or facebook/map-anything (CC-BY-NC-4.0)')
    ap.add_argument('--minibatch-size', type=int, default=1, help='frames per forward pass (1 = lowest memory)')
    a = ap.parse_args()
    frames = sorted(p for p in a.frames.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    T = anchor_poses(predict_poses(frames, a.model, a.minibatch_size))
    torch.save({'T_anchor_front': T}, a.out)
    print(f'{a.out}: forward travel {float(T[-1, 2, 3]):.2f} m over {len(frames)} frames', flush=True)


if __name__ == '__main__':
    main()
