"""Sampling of the two stages: the cross views from the depth warp, then the rear views from the ground prior built from
the front and the generated cross views. Euler steps of the flow ODE with classifier-free guidance on the caption."""
from __future__ import annotations

import torch

from openlongtail.config import CROSS_VIEWS, MODEL
from openlongtail.data.ground_prior import GROUND_PRIOR_ORDER, Segmenter, ground_prior
from openlongtail.flow import FlowMatchScheduler
from openlongtail.models.surround import TASK_VIEWS, SurroundDiT, prepare_inputs, run_model
from openlongtail.models.wan import vae_encode_batch

TO_SEQUENCE = [GROUND_PRIOR_ORDER.index(v) for v in TASK_VIEWS['rear']]


def cross_noise(shape: tuple[int, ...], seed: int, device: torch.device) -> dict[int, torch.Tensor]:
    """Initial noise of the cross views: 0.5 * shared + sqrt(0.75) * own."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shared = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    return {v: 0.5 * shared + 0.75 ** 0.5 * torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for v in CROSS_VIEWS}


def rear_noise(shape: tuple[int, ...], seed: int, device: torch.device) -> dict[int, torch.Tensor]:
    """Initial noise of the rear views: 0.5 * shared + sqrt(0.75) * own. One draw per target id 1..5, in order, of which the
    rear views 3..5 are used; the draw order fixes the noise a seed gives."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, *shape)
    shared = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    own = [torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(5)]
    return {v: (0.5 * shared + 0.75 ** 0.5 * noise)[0] for v, noise in zip((1, 2, 3, 4, 5), own)}


def _inputs(task: str, z_front, K, E, T_anchor_front, prior_latents, prior_visibility) -> dict[str, torch.Tensor]:
    return prepare_inputs(task, z_front[None].to(torch.bfloat16), K[None].float(), E[None].float(), T_anchor_front[None].float(),
                          prior_latents[None], prior_visibility[None])


@torch.no_grad()
def sample_cross(model: SurroundDiT, z_front: torch.Tensor, K: torch.Tensor, E: torch.Tensor, T_anchor_front: torch.Tensor,
                 text_emb: torch.Tensor, null_emb: torch.Tensor, warp_latents: torch.Tensor, warp_visibility: torch.Tensor,
                 noise: dict[int, torch.Tensor], num_steps: int = 50, guide_scale: float = 3.5) -> dict[int, torch.Tensor]:
    """Stage 1: cross-left and cross-right, jointly. z_front (16,T,H,W); K (6,3,3), E (6,4,4), T_anchor_front (T,4,4);
    text / null (L,4096); warp_* (2,16|1,T,H,W) in CROSS_VIEWS order; noise: view -> (16,T,H,W). Returns view -> latent."""
    device = z_front.device
    views = TASK_VIEWS['cross']
    inputs = _inputs('cross', z_front, K, E, T_anchor_front, warp_latents, warp_visibility)
    sigmas = FlowMatchScheduler(MODEL.flow_shift).inference_sigmas(num_steps, device=device)
    x = torch.stack([noise[v] for v in views])[None].to(torch.bfloat16)
    text, null = text_emb[None].to(torch.bfloat16), null_emb[None].to(torch.bfloat16)
    for i in range(num_steps):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        cur = torch.full((1, len(views)), s, device=device)
        v = run_model(model, 'cross', x, cur, text, inputs)
        if guide_scale != 1.0:
            v_uncond = run_model(model, 'cross', x, cur, null, inputs)
            v = v_uncond + guide_scale * (v - v_uncond)
        x = x + (s_next - s) * v.to(x.dtype)
    return {view: x[0, i] for i, view in enumerate(views)}


@torch.no_grad()
def prior_condition(src: torch.Tensor, K: torch.Tensor, E: torch.Tensor, T_anchor_front: torch.Tensor, seg: Segmenter,
                    vae) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """src (3,41,3,H,W) front / cross-left / cross-right in 0..255 -> ground-prior latents (3,16,11,h,w) bf16 and visibility
    (3,1,11,h,w) fp16 in GROUND_PRIOR_ORDER, and the coverage per rear camera."""
    rgbs, vis, coverage = ground_prior(src, K, E, T_anchor_front, seg)
    return vae_encode_batch(vae, rgbs).to(torch.bfloat16), torch.stack(vis).to(torch.float16), coverage


@torch.no_grad()
def sample_rear(model: SurroundDiT, z_front: torch.Tensor, K: torch.Tensor, E: torch.Tensor, T_anchor_front: torch.Tensor,
                text_emb: torch.Tensor, null_emb: torch.Tensor, ground_latents: torch.Tensor, ground_visibility: torch.Tensor,
                noise: dict[int, torch.Tensor], num_steps: int = 50, guide_scale: float = 3.5) -> dict[int, torch.Tensor]:
    """Stage 2: rear-left, rear-tele and rear-right, jointly. ground_* in GROUND_PRIOR_ORDER (as prior_condition returns
    them), noise: view -> (16,T,H,W); the rest as in sample_cross. Returns view -> latent."""
    device = z_front.device
    views = TASK_VIEWS['rear']
    inputs = _inputs('rear', z_front, K, E, T_anchor_front, ground_latents[TO_SEQUENCE], ground_visibility[TO_SEQUENCE])
    sigmas = FlowMatchScheduler(MODEL.flow_shift).inference_sigmas(num_steps, device=device)
    x = torch.stack([noise[v] for v in views])[None].to(torch.bfloat16)
    text, null = text_emb[None].to(torch.bfloat16), null_emb[None].to(torch.bfloat16)
    for i in range(num_steps):
        cur = torch.full((1, len(views)), float(sigmas[i]), device=device)
        nxt = torch.full((1, len(views)), float(sigmas[i + 1]), device=device)
        v = run_model(model, 'rear', x, cur, text, inputs)
        if guide_scale != 1.0:
            v_uncond = run_model(model, 'rear', x, cur, null, inputs)
            v = v_uncond + guide_scale * (v - v_uncond)
        x = x + (nxt - cur).to(x.dtype).view(1, len(views), 1, 1, 1, 1) * v.to(x.dtype)
    return {view: x[0, i] for i, view in enumerate(views)}
