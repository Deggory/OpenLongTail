"""Reading and writing the mp4 outputs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def to_uint8(frames: torch.Tensor) -> torch.Tensor:
    """(T,3,H,W) in 0..255, -1..1 or 0..1 -> uint8."""
    x = frames.detach().float().cpu()
    if float(x.max()) > 2.0:
        x = x.clamp(0.0, 255.0) / 255.0
    elif float(x.min()) < 0.0:
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    else:
        x = x.clamp(0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8)


def frames_u8(video: torch.Tensor) -> np.ndarray:
    """(T,3,H,W) float in 0..255 -> (T,H,W,3) uint8."""
    return video.permute(0, 2, 3, 1).round().clamp(0, 255).byte().cpu().numpy()


def write_mp4(path: Path, frames: torch.Tensor | np.ndarray, fps: int) -> None:
    """frames: (T,3,H,W) tensor (any range to_uint8 accepts) or (T,H,W,3) uint8 array."""
    import imageio.v2 as imageio
    if isinstance(frames, torch.Tensor):
        frames = to_uint8(frames).permute(0, 2, 3, 1).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def read_mp4(path: Path, device: torch.device, n: int = 41) -> torch.Tensor:
    """-> (n,3,H,W) float in 0..255 on `device`."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != n:
        raise ValueError(f'{path}: expected {n} frames, got {len(frames)}')
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device)


def video_fps(path: Path, default: int = 16) -> int:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps)) if fps and fps > 0 else default


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (8 + 9 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_surround(path: Path, front: np.ndarray, views: dict[str, np.ndarray], fps: int) -> None:
    """Half-resolution 2x3 preview from (T,H,W,3) uint8 videos keyed by view name:
        [ cross-left | front     | cross-right ]
        [ rear-left  | rear-tele | rear-right  ]"""
    grid = []
    for t in range(front.shape[0]):
        top = np.concatenate([_label(views['cross_left'][t], 'cross-left  stage 1'), _label(front[t], 'front  input'),
                              _label(views['cross_right'][t], 'cross-right  stage 1')], 1)
        bottom = np.concatenate([_label(views['rear_left'][t], 'rear-left  stage 2'), _label(views['rear_tele'][t], 'rear-tele  stage 2'),
                                 _label(views['rear_right'][t], 'rear-right  stage 2')], 1)
        grid.append(np.concatenate([top, bottom], 0)[::2, ::2])
    write_mp4(path, np.stack(grid), fps)
