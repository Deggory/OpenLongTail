"""Stage-2 condition: the ground prior of each rear camera.

The road behind the car at time t, up to the distance driven inside the clip, was seen by the front cameras (front,
cross-left, cross-right) at earlier frames. The road is locally planar, so no depth is needed: a rear pixel's ray
is intersected with the ground plane under the rig at t, the hit point is moved into the rig frame of an earlier
key frame with the ego poses, and its colour is read from the front cameras there ("history", exact also on curves).
Pixels no earlier frame saw fall back to the mirror image of the road ahead across the rig's lateral axis ("flip").
Only pixels labelled as ground by a semantic segmenter are copied, so vehicles are not smeared onto the road.

The prior is VAE-encoded like the stage-1 warp and paired with a graded visibility: 1 = history, 0.5 = flip, 0 = none.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from openlongtail.config import REAR_VIEWS
from openlongtail.geometry import anchor_camera_transforms

SEGMENTER = 'facebook/mask2former-swin-large-mapillary-vistas-semantic'
GROUND_CLASSES = ('Road', 'Lane Marking', 'Curb', 'Sidewalk', 'Parking', 'Crosswalk', 'Service Lane', 'Bike Lane',
                  'Pothole', 'Manhole', 'Catch Basin', 'Terrain', 'Snow', 'Sand')
KEY_RGB = [0] + [4 * k - 1 for k in range(1, 11)]          # representative RGB frame of each latent frame
SOURCE_VIEWS = (0, 1, 2)                                   # front, cross-left, cross-right
MAX_DISTANCE = 60.0                                        # metres, along the rig's longitudinal axis
GROUND_PRIOR_ORDER = REAR_VIEWS                            # order of the rear cameras in *_gprior.pt / *_rearlane.pt


class Segmenter:
    """Mask2Former (Mapillary Vistas) at the checkpoint's processor size."""

    def __init__(self, device: torch.device) -> None:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        from transformers.utils import logging
        verbosity = logging.get_verbosity()
        logging.set_verbosity_error()          # the checkpoint lacks the unused final Swin layernorm; its load report is noise
        self.proc = AutoImageProcessor.from_pretrained(SEGMENTER)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(SEGMENTER).to(device).eval().half()
        logging.set_verbosity(verbosity)
        self.mean = torch.tensor(self.proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.proc.image_std, device=device).view(1, 3, 1, 1)
        labels = self.model.config.id2label
        self.ground_ids = torch.tensor([i for i, n in labels.items() if any(k in n for k in GROUND_CLASSES)], device=device)
        self.lane_ids = torch.tensor([i for i, n in labels.items() if 'Lane Marking' in n and 'Crosswalk' not in n], device=device)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, batch: int = 12) -> torch.Tensor:
        """(N,3,H,W) 0..255 on the GPU -> class map (N,H,W)."""
        H, W = frames.shape[2:]
        size = (self.proc.size['height'], self.proc.size['width'])
        out = []
        for chunk in frames.split(batch):
            x = chunk.float()
            if tuple(chunk.shape[2:]) != size:
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False, antialias=True)
            logits = self.model(pixel_values=((x / 255.0 - self.mean) / self.std).half())
            out += self.proc.post_process_semantic_segmentation(logits, target_sizes=[(H, W)] * len(chunk))
        return torch.stack(out)


def to_latent_res(mask: torch.Tensor) -> torch.Tensor:
    """(41,1,H,W) soft mask -> (1,11,60,104): spatial area average, then mean over the RGB frames of each latent frame."""
    m = F.adaptive_avg_pool2d(mask.float(), (60, 104))
    chunks = [m[0:1]] + [m[4 * i - 3:4 * i + 1].mean(0, keepdim=True) for i in range(1, 11)]
    return torch.cat(chunks, 0).permute(1, 0, 2, 3)


def interpolate_camera_poses(T_key: torch.Tensor, key_rgb: list[int], num_rgb: int) -> torch.Tensor:
    """(V,K,4,4) camera->anchor at the key RGB frames -> (V,num_rgb,4,4): translation lerp, rotation slerp."""
    from scipy.spatial.transform import Rotation, Slerp
    out = torch.zeros(T_key.shape[0], num_rgb, 4, 4, dtype=torch.float64)
    ts = np.clip(np.arange(num_rgb, dtype=np.float64), key_rgb[0], key_rgb[-1])
    for v in range(T_key.shape[0]):
        Tk = T_key[v].double().cpu().numpy()
        out[v, :, :3, :3] = torch.from_numpy(Slerp(key_rgb, Rotation.from_matrix(Tk[:, :3, :3]))(ts).as_matrix())
        out[v, :, :3, 3] = torch.from_numpy(np.stack([np.interp(ts, key_rgb, Tk[:, i, 3]) for i in range(3)], -1))
        out[v, :, 3, 3] = 1.0
    return out.to(T_key.device).float()


def _lookup(P_anchor: torch.Tensor, T_cam: torch.Tensor, K: torch.Tensor, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample maps (C,H,W) of the camera T_cam (camera->anchor) at anchor points P (...,4) (NaN = no point)."""
    H, W = maps.shape[-2:]
    C_ = (P_anchor @ torch.linalg.inv(T_cam).T)[..., :3]
    z = C_[..., 2]
    uv = C_ @ K.float().T
    uu, vv = uv[..., 0] / z.clamp(min=1e-6), uv[..., 1] / z.clamp(min=1e-6)
    ok = (z > 0.5) & (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
    g = torch.stack([uu / (W - 1) * 2 - 1, vv / (H - 1) * 2 - 1], -1).nan_to_num(-2.0).reshape(1, -1, 1, 2)
    vals = F.grid_sample(maps[None].float(), g, mode='bilinear', align_corners=True)[0, :, :, 0]
    return vals.reshape(maps.shape[0], *ok.shape), ok


def render_ground_prior(src_rgb: torch.Tensor, src_ground: torch.Tensor, K: torch.Tensor, E: torch.Tensor, T_cam: torch.Tensor,
                        target: int, key_rgb: list[int], out_hw: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """src_rgb (S,Kf,3,H,W), src_ground (S,Kf,1,H,W) of the source cameras at the key RGB frames; T_cam (6,N,4,4)
    camera->anchor at every output frame. Per frame, history keys are tried most recent first, then the mirror at
    the nearest key. Returns rgb (N,3,h,w) and source (N,1,h,w) (0 none, 1 history, 2 flip); (h,w) = out_hw or (H,W)."""
    S, Kf, C, H, W = src_rgb.shape
    h, w = out_hw or (H, W)
    dev = src_rgb.device
    N = T_cam.shape[1]
    frames = list(range(N))
    v_, u_ = torch.meshgrid((torch.arange(h, device=dev) + 0.5) * H / h, (torch.arange(w, device=dev) + 0.5) * W / w, indexing='ij')
    d_cam = torch.stack([u_, v_, torch.ones_like(u_)], -1) @ torch.linalg.inv(K[target].float()).T
    T_rig = T_cam[0] @ torch.linalg.inv(E[0].float())                      # rig(t) -> anchor
    Tv = T_cam[target]
    d = torch.einsum('hwj,nij->nhwi', d_cam, Tv[:, :3, :3])
    n, p0, o = T_rig[:, :3, 2], T_rig[:, :3, 3], Tv[:, :3, 3]
    denom = torch.einsum('nhwi,ni->nhw', d, n)
    s = ((p0 - o) * n).sum(-1).view(N, 1, 1) / torch.where(denom.abs() < 1e-6, torch.full_like(denom, -1e-6), denom)
    P = torch.cat([o.view(N, 1, 1, 3) + s[..., None] * d, torch.ones_like(s[..., None])], -1)
    P_rig = torch.einsum('nhwj,nij->nhwi', P, torch.linalg.inv(T_rig))
    hit = (s > 0) & (denom < 0) & (P_rig[..., 0].abs() < MAX_DISTANCE)
    nan = torch.full_like(P, float('nan'))
    maps = torch.cat([src_rgb.float(), src_ground.float()], 2)
    out = torch.zeros(N, C, h, w, device=dev)
    valid = torch.zeros(N, h, w, dtype=torch.bool, device=dev)
    source = torch.zeros(N, h, w, dtype=torch.uint8, device=dev)

    def fill(P_a: torch.Tensor, frame_mask: list[bool], key: int, tag: int) -> None:
        sel = torch.tensor([i for i, m in enumerate(frame_mask) if m], device=dev, dtype=torch.long)
        if sel.numel() == 0:
            return
        for cam in SOURCE_VIEWS:
            val, ok = _lookup(P_a[sel], T_cam[cam, key_rgb[key]], K[cam], maps[cam, key])
            cur = valid[sel]
            ok = ok & (val[C] > 0.999) & ~cur
            out[sel] = torch.where(ok[:, None], val[:C].transpose(0, 1), out[sel])
            valid[sel] = cur | ok
            source[sel] = torch.where(ok, torch.full_like(source[sel], tag), source[sel])

    P_hist = torch.where(hit[..., None], P, nan)
    for key in reversed(range(Kf)):
        fill(P_hist, [key_rgb[key] < t for t in frames], key, 1)
    Pm = P_rig.clone()
    Pm[..., 0] = -Pm[..., 0]
    Pm = torch.where(hit[..., None], torch.einsum('nhwj,nij->nhwi', Pm, T_rig), nan)
    nearest = [min(range(len(key_rgb)), key=lambda k: abs(key_rgb[k] - t)) for t in frames]
    for key in range(Kf):
        fill(Pm, [k == key for k in nearest], key, 2)
    return out, source[:, None]


@torch.no_grad()
def ground_prior(src: torch.Tensor, K: torch.Tensor, E: torch.Tensor, T_anchor_front: torch.Tensor, seg: Segmenter, seg_batch: int = 12):
    """src (3,41,3,H,W) front / cross-left / cross-right in 0..255 on the GPU; K (6,3,3), E (6,4,4), T_anchor_front (11,4,4).
    Returns the prior RGB video of each rear camera (list of 3 x (41,3,H,W)), its visibility at latent resolution
    (list of 3 x (1,11,60,104) float) and the history / flip coverage per camera."""
    dev = src.device
    cls = seg(src[:, KEY_RGB].flatten(0, 1), batch=seg_batch).view(3, len(KEY_RGB), *src.shape[3:])
    ground = torch.isin(cls, seg.ground_ids)[:, :, None]
    T_cam = interpolate_camera_poses(anchor_camera_transforms(E[None].cpu(), T_anchor_front[None].cpu())[0], KEY_RGB, src.shape[1]).to(dev)
    rgbs, vis, coverage = [], [], {}
    for v in REAR_VIEWS:
        rgb, source = render_ground_prior(src[:, KEY_RGB], ground, K, E, T_cam, v, KEY_RGB)
        rgbs.append(rgb)
        vis.append(to_latent_res((source == 1).float() + 0.5 * (source == 2).float()))
        coverage[v] = {'history': float((source == 1).float().mean()), 'flip': float((source == 2).float().mean())}
    return rgbs, vis, coverage
