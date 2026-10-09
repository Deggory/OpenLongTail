"""Stage-1 condition: the front video forward-splatted into each cross camera with its depth (frame t -> frame t through
the rig extrinsics), VAE-encoded, plus the splat visibility at latent resolution."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from openlongtail.config import CROSS_VIEWS, FRONT
from openlongtail.geometry import forward_splat_warp, se3_inverse
from openlongtail.models.wan import vae_encode_video


def visibility_to_latent(vis: torch.Tensor, t_lat: int = 11, h_lat: int = 60, w_lat: int = 104) -> torch.Tensor:
    """(41,1,H,W) splat visibility -> (1,11,h,w) fp16: bilinear resize, max over the RGB frames of each latent frame."""
    spatial = F.interpolate(vis.float(), size=(h_lat, w_lat), mode='bilinear', align_corners=False)
    chunks = [spatial[0:1]] + [spatial[4 * i - 3:4 * i + 1].max(dim=0, keepdim=True).values for i in range(1, t_lat)]
    return torch.cat(chunks, dim=0).permute(1, 0, 2, 3).to(torch.float16)


@torch.no_grad()
def encode_warp(vae, warped: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(41,3,H,W) uint8 -> (16,11,h,w) bf16 on the CPU."""
    return vae_encode_video(vae, warped.to(device=device, dtype=torch.float32)).detach().to('cpu', dtype=torch.bfloat16)


def warp_same_time(front: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, E: torch.Tensor, view: int,
                   splat_radius: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """front (41,3,H,W), depth (41,H,W) on the target device; K (6,3,3), E (6,4,4)."""
    T_front_to_view = se3_inverse(E[view].unsqueeze(0)).squeeze(0) @ E[FRONT]
    return forward_splat_warp(front, depth, K[FRONT], K[view], T_front_to_view, out_h=front.shape[-2], out_w=front.shape[-1],
                              splat_radius=splat_radius)


@torch.no_grad()
def live_warp(front: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, E: torch.Tensor, vae, device: torch.device,
              splat_radius: int = 1) -> tuple[torch.Tensor, torch.Tensor, dict[int, float]]:
    """Warp condition of a raw clip: latents (2,16,11,h,w) bf16 and visibility (2,1,11,h,w) fp16 in CROSS_VIEWS order,
    and the visible share of the latent cells (%) per view."""
    K, E = K.to(device).float(), E.to(device).float()
    front, depth = front.to(device), depth.to(device)
    h_lat, w_lat = front.shape[-2] // 8, front.shape[-1] // 8
    latents, visibility, coverage = [], [], {}
    for view in CROSS_VIEWS:
        rgb, vis = warp_same_time(front, depth, K, E, view, splat_radius)
        z, vis = encode_warp(vae, rgb, device), visibility_to_latent(vis.cpu().float(), h_lat=h_lat, w_lat=w_lat)
        latents.append(z.to(device=device, dtype=torch.bfloat16))
        visibility.append(vis.to(device=device, dtype=torch.float16))
        coverage[view] = float(vis.float().mean() * 100.0)
    return torch.stack(latents), torch.stack(visibility), coverage


def load_warp(warp_path: Path, device: torch.device | str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """Warp condition of a latent-cache clip (clip_<id>_warp.pt): latents (2,16,11,h,w) bf16 and visibility (2,1,11,h,w)
    fp16 in CROSS_VIEWS order."""
    warp = torch.load(warp_path, map_location='cpu', weights_only=True)
    ids = [int(v) for v in warp['target_view_ids']]
    slots = [ids.index(v) for v in CROSS_VIEWS]
    return (warp['warped_target_latents'][slots].to(device=device, dtype=torch.bfloat16),
            warp['warped_target_visibility'][slots].to(device=device, dtype=torch.float16))
