"""Camera geometry: rig poses over time, Plucker rays, pose features and forward depth splatting.

Conventions: K (6,3,3) intrinsics at 480x832; E (6,4,4) camera->rig extrinsics; T_anchor_front (T,4,4) the front
camera at each latent frame relative to the front camera at frame 0 (OpenCV axes, +Z forward).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from openlongtail.config import FRONT

PLUCKER_TOKEN_STRIDE = 16.0      # latent stride 8 x patch size 2
PLUCKER_MOMENT_SCALE = 25.0
POSE_TRANSLATION_SCALE = 25.0
POSE_FEATURE_CLIP = 4.0


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:4] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def mat3_inverse(M: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of (...,3,3) matrices, computed in fp32."""
    in_dtype = M.dtype
    m = M.float()
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    A, B, C = e * i - f * h, -(d * i - f * g), d * h - e * g
    D, E, F_ = -(b * i - c * h), a * i - c * g, -(a * h - b * g)
    G, H, I = b * f - c * e, -(a * f - c * d), a * e - b * d
    inv_det = 1.0 / (a * A + b * B + c * C)
    out = torch.stack([torch.stack([A * inv_det, D * inv_det, G * inv_det], dim=-1),
                       torch.stack([B * inv_det, E * inv_det, H * inv_det], dim=-1),
                       torch.stack([C * inv_det, F_ * inv_det, I * inv_det], dim=-1)], dim=-2)
    return out.to(in_dtype)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of (...,3,3) rotations via the trace formula."""
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    vee = torch.stack([rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0],
                       rotation[..., 1, 0] - rotation[..., 0, 1]], dim=-1)
    scale = torch.where(theta < 1e-05, torch.full_like(theta, 0.5), theta / (2.0 * torch.sin(theta).clamp_min(1e-08)))
    return vee * scale.unsqueeze(-1)


def _matrix_to_quaternion_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Adapted from matrix_to_quaternion of PyTorch3D (Copyright (c) Meta Platforms, Inc. and affiliates, BSD-3-Clause,
    see THIRD_PARTY_NOTICES)."""
    m = rotation.float()
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    q_abs = torch.sqrt(torch.clamp(torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                                                1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1), min=0.0))
    candidates = torch.stack([torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                              torch.stack([m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m02 + m20], dim=-1),
                              torch.stack([m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                              torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1)], dim=-2)
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(1e-08))
    index = q_abs.argmax(dim=-1)[..., None, None].expand(*q_abs.shape[:-1], 1, 4)
    quat = F.normalize(torch.gather(candidates, -2, index).squeeze(-2), dim=-1, eps=1e-08)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def so3_log_stable(rotation: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector via quaternions (well-conditioned near 0 and pi)."""
    quat = _matrix_to_quaternion_wxyz(rotation)
    xyz = quat[..., 1:]
    sin_half = xyz.norm(dim=-1)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 0].clamp_min(1e-08))
    scale = torch.where(sin_half < 1e-06, torch.full_like(sin_half, 2.0), angle / sin_half.clamp_min(1e-08))
    return xyz * scale.unsqueeze(-1)


def anchor_camera_transforms(E: torch.Tensor, T_anchor_front: torch.Tensor) -> torch.Tensor:
    """E (B,6,4,4) camera->rig, T_anchor_front (B,T,4,4) -> every camera at every frame in the anchor frame (B,6,T,4,4)."""
    E_front_cam = se3_inverse(E[:, FRONT:FRONT + 1].float()) @ E.float()
    T = T_anchor_front.float()
    return (T[:, :, None] @ E_front_cam[:, None]).permute(0, 2, 1, 3, 4).contiguous()


def trajectory_vectors(T_anchor_front: torch.Tensor, frames: int) -> torch.Tensor:
    """Frame-to-frame ego motion (B,T,6): translation + axis-angle of T[t-1]^-1 T[t] (identity at t=0)."""
    T = T_anchor_front[:, :frames].float()
    delta = torch.eye(4, device=T.device, dtype=T.dtype).view(1, 1, 4, 4).expand(T.shape[0], frames, -1, -1).clone()
    if frames > 1:
        delta[:, 1:] = se3_inverse(T[:, :-1]) @ T[:, 1:]
    return torch.cat([delta[:, :, :3, 3], so3_log(delta[:, :, :3, :3])], dim=-1)


def plucker_rays(K: torch.Tensor, E_dynamic: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
    """K (B,6,3,3), E_dynamic (B,T,6,4,4) camera->anchor -> Plucker rays (B,6,T,6,h_tok,w_tok) at token resolution."""
    dtype, device = K.dtype, K.device
    E_dynamic = E_dynamic.to(device=device, dtype=dtype)
    K_tok = K.clone()
    K_tok[..., 0, :] /= PLUCKER_TOKEN_STRIDE
    K_tok[..., 1, :] /= PLUCKER_TOKEN_STRIDE
    gy, gx = torch.meshgrid(torch.arange(h_tok, device=device, dtype=dtype), torch.arange(w_tok, device=device, dtype=dtype), indexing='ij')
    pixels = torch.stack([gx + 0.5, gy + 0.5, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    dirs_cam = torch.einsum('bvij,nj->bvni', mat3_inverse(K_tok), pixels)
    dirs = F.normalize(torch.einsum('btvij,bvnj->btvni', E_dynamic[..., :3, :3], dirs_cam), dim=-1, eps=1e-08)
    origin = E_dynamic[..., :3, 3].unsqueeze(-2)
    moment = torch.cross(origin.expand_as(dirs), dirs, dim=-1) / PLUCKER_MOMENT_SCALE
    B, T, V = E_dynamic.shape[:3]
    plucker = torch.cat([dirs, moment], dim=-1).reshape(B, T, V, h_tok, w_tok, 6).permute(0, 2, 1, 5, 3, 4).contiguous()
    abs_max = float(plucker.detach().abs().max().cpu())
    if abs_max >= 1000.0:
        raise ValueError(f'Plucker values out of range (abs max {abs_max:.1f}); check the scale of T_anchor_front')
    return plucker


def plucker_streams(K: torch.Tensor, E: torch.Tensor, T_anchor_front: torch.Tensor, view_ids: torch.Tensor,
                    latent_frames: int, h_tok: int, w_tok: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Plucker rays of the given views (B,S,T,6,h_tok,w_tok) following the ego motion."""
    E_dynamic = anchor_camera_transforms(E, T_anchor_front).permute(0, 2, 1, 3, 4).contiguous()
    plucker = plucker_rays(K.float(), E_dynamic, h_tok=h_tok, w_tok=w_tok)[:, view_ids.to(K.device), :latent_frames]
    return plucker if dtype is None else plucker.to(dtype=dtype)


def _normalize_pose_features(features: torch.Tensor) -> torch.Tensor:
    normalized = torch.cat([features[..., :3] / POSE_TRANSLATION_SCALE, features[..., 3:] / float(torch.pi)], dim=-1)
    return torch.nan_to_num(normalized, nan=0.0, posinf=POSE_FEATURE_CLIP, neginf=-POSE_FEATURE_CLIP).clamp(-POSE_FEATURE_CLIP, POSE_FEATURE_CLIP)


def target_pose_features(T_anchor_cam: torch.Tensor, target_view: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Pose of the target camera in the anchor frame at each frame, (B,T,6) normalized translation + rotation."""
    T_target = T_anchor_cam[:, target_view]
    trans = T_target[..., :3, 3]
    rot = so3_log_stable(T_target[..., :3, :3]).to(device=trans.device, dtype=trans.dtype)
    features = _normalize_pose_features(torch.cat([trans, rot], dim=-1))
    return features if dtype is None else features.to(dtype=dtype)


def forward_splat_warp(front_rgb: torch.Tensor, front_depth: torch.Tensor, K_front: torch.Tensor, K_target: torch.Tensor,
                       T_front_to_target: torch.Tensor, out_h: int, out_w: int, splat_radius: int = 1,
                       near: float = 0.5, far: float = 200.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-splat front pixels with metric depth into a target camera, writing far points first so that nearer ones
    overwrite them (CUDA does not order duplicate writes, so overlapping splats can differ between runs).
    front_rgb (T,3,H,W) 0..255, front_depth (T,H,W) metres -> warped (T,3,h,w) uint8 and visibility (T,1,h,w)."""
    device = front_rgb.device
    T, _, H, W = front_rgb.shape
    fx, fy, cx, cy = float(K_front[0, 0]), float(K_front[1, 1]), float(K_front[0, 2]), float(K_front[1, 2])
    fxt, fyt, cxt, cyt = float(K_target[0, 0]), float(K_target[1, 1]), float(K_target[0, 2]), float(K_target[1, 2])
    vs, us = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    rgb_flat = front_rgb.float().permute(0, 2, 3, 1).reshape(T, H * W, 3)
    d_flat = front_depth.reshape(T, H * W)
    P_z = d_flat
    P = torch.stack([((us.reshape(-1) - cx) / fx).unsqueeze(0) * d_flat, ((vs.reshape(-1) - cy) / fy).unsqueeze(0) * d_flat,
                     P_z, torch.ones_like(P_z)], dim=-1)
    Pt = torch.einsum('ij,thj->thi', T_front_to_target.to(device=device, dtype=torch.float32), P)
    Xt, Yt, Zt = Pt[..., 0], Pt[..., 1], Pt[..., 2]
    safe_Z = Zt.clamp(min=1e-06)
    ut = (fxt * Xt / safe_Z + cxt).round().long()
    vt = (fyt * Yt / safe_Z + cyt).round().long()
    keep = (Zt > near) & (d_flat > near) & (d_flat < far) & (ut >= 0) & (ut < out_w) & (vt >= 0) & (vt < out_h)
    warped = torch.zeros((T, out_h, out_w, 3), dtype=torch.float32, device=device)
    visibility = torch.zeros((T, out_h, out_w), dtype=torch.float32, device=device)
    offsets = [(du, dv) for du in range(-splat_radius, splat_radius + 1) for dv in range(-splat_radius, splat_radius + 1)]
    for t in range(T):
        idx = keep[t].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        order = torch.argsort(Zt[t][idx], descending=True)       # far first
        u_sort, v_sort, rgb_sort = ut[t][idx][order], vt[t][idx][order], rgb_flat[t][idx][order]
        for du, dv in offsets:
            ok = (u_sort + du >= 0) & (u_sort + du < out_w) & (v_sort + dv >= 0) & (v_sort + dv < out_h)
            u_off = (u_sort + du).clamp(0, out_w - 1)[ok]
            v_off = (v_sort + dv).clamp(0, out_h - 1)[ok]
            rgb_off = rgb_sort[ok]
            for ch in range(3):
                warped[t].index_put_((v_off, u_off, torch.full_like(u_off, ch)), rgb_off[:, ch], accumulate=False)
            visibility[t].index_put_((v_off, u_off), torch.ones_like(u_off, dtype=torch.float32), accumulate=False)
    return warped.permute(0, 3, 1, 2).clamp(0, 255).to(torch.uint8), visibility.unsqueeze(1)
