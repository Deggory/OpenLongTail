"""The OpenLongTail model: Wan2.1-VACE-1.3B generating several cameras of a clip jointly, in one sequence.

The target videos and the clean front video are laid end to end along the frame axis, and the pretrained Wan
self-attention runs over the whole sequence. Views are told apart by Plucker-ray, camera-id, role and ego-motion
embeddings; each view carries its own timestep (front = 0, so it stays a clean condition inside the sequence) and its
camera pose modulates its AdaLN. The geometric prior of every target view fills its VACE reactive slot with its
visibility as the mask; the front sits in the inactive slot.

One network, two tasks (the two inference stages):
  * 'cross': cross-left and cross-right, conditioned on the front video warped into them with its depth. Each attends to
    itself and the front (the two cameras do not overlap), the front to itself.
  * 'rear':  rear-left, rear-tele and rear-right, conditioned on their ground prior (openlongtail.data.ground_prior),
    with full attention over the four views.
Trained: a LoRA on every self-attention, the base weights of the main blocks' self-attention, and the embeddings.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from openlongtail.config import FRONT, MODEL, WAN21_VACE_DIR, ModelConfig
from openlongtail.geometry import anchor_camera_transforms, plucker_streams, target_pose_features, trajectory_vectors
from openlongtail.models.wan import inject_lora, load_vace_expert, rope_apply

TASK_VIEWS = {'cross': (1, 2), 'rear': (3, 5, 4)}       # rear-tele sits between the two rear views it overlaps
CONTEXT_VIEWS = (FRONT,)
TASK_GRAPH = {'cross': ((0, 2), (1, 2), (2,)), 'rear': None}   # per sequence position, the positions it attends to
ROLE_CONDITION, ROLE_TARGET = 0, 1
WAN_TIMESTEP_SCALE = 1000.0


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Wan2.1's timestep embedding (Apache-2.0, see THIRD_PARTY_NOTICES)."""
    half = dim // 2
    position = position.to(torch.float64)
    freqs = torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64).div(half))
    sinusoid = torch.outer(position, freqs)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ---- embeddings ------------------------------------------------------------------------------------------
class PluckerMLP(nn.Module):

    def __init__(self, out_dim: int, in_dim: int = 6, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.SiLU()
        for fc in (self.fc1, self.fc2):
            nn.init.normal_(fc.weight, std=0.02)
            nn.init.zeros_(fc.bias)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        """(B,S,T,6,h,w) -> (B,S,T,dim,h,w)."""
        x = self.fc2(self.act(self.fc1(plucker.permute(0, 1, 2, 4, 5, 3).contiguous())))
        return x.permute(0, 1, 2, 5, 3, 4).contiguous()


class CamIDEmbed(nn.Module):

    def __init__(self, dim: int, num_views: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_views, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, view_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(view_ids)


class TrajectoryMLP(nn.Module):

    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return self.net(motion)


def view_graph_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_views: int, graph: Sequence[Sequence[int]]) -> torch.Tensor:
    """q, k, v (B,H,L,D) in view-major order; graph[i] lists the views whose tokens view i attends to. One SDPA call per
    distinct key set, so every query keeps a single softmax over exactly its allowed keys (same as a boolean mask)."""
    B, H, L, D = q.shape
    n = L // num_views
    q5, k5, v5 = (t.view(B, H, num_views, n, D) for t in (q, k, v))
    groups: dict[tuple[int, ...], list[int]] = {}
    for qv, keys in enumerate(graph):
        groups.setdefault(tuple(keys), []).append(qv)
    out: list[torch.Tensor | None] = [None] * num_views
    for keys, qvs in groups.items():
        kk = k5[:, :, list(keys)].reshape(B, H, len(keys) * n, D)
        vv = v5[:, :, list(keys)].reshape(B, H, len(keys) * n, D)
        o = F.scaled_dot_product_attention(q5[:, :, qvs].reshape(B, H, len(qvs) * n, D), kk, vv).view(B, H, len(qvs), n, D)
        for i, qv in enumerate(qvs):
            out[qv] = o[:, :, i]
    return torch.stack(out, 2).reshape(B, H, L, D)


# ---- model -----------------------------------------------------------------------------------------------
class SurroundDiT(nn.Module):

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.dim = int(expert.dim)
        self.plucker_mlp = PluckerMLP(out_dim=self.dim)
        self.cam_id_embed = CamIDEmbed(dim=self.dim)
        self.role_embed = nn.Embedding(2, self.dim)
        self.trajectory_mlp = TrajectoryMLP(self.dim)
        self.target_pose_mlp = nn.Sequential(nn.Linear(6, self.dim), nn.SiLU(), nn.Linear(self.dim, 6 * self.dim))
        self.target_view_mod_embed = nn.Embedding(6, 6 * self.dim)
        for emb in (self.role_embed, self.target_view_mod_embed):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.zeros_(self.target_pose_mlp[-1].weight)
        nn.init.zeros_(self.target_pose_mlp[-1].bias)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def force_fp32_time_and_head_modules(self) -> None:
        self.expert.time_embedding.float()
        self.expert.time_projection.float()
        self.expert.head.float()

    def trainable_modules(self) -> nn.ModuleDict:
        names = ('plucker_mlp', 'cam_id_embed', 'role_embed', 'trajectory_mlp', 'target_pose_mlp', 'target_view_mod_embed')
        return nn.ModuleDict({name: getattr(self, name) for name in names})

    def trainable_attention(self) -> list[nn.Parameter]:
        """Base weights of every main-block self-attention projection and q/k norm (their LoRA is trained as well)."""
        params = []
        for block in self.expert.blocks:
            sa = block.self_attn
            for name in ('q', 'k', 'v', 'o'):
                lin = getattr(sa, name)
                base = lin.base if hasattr(lin, 'base') else lin
                params += [base.weight, base.bias]
            params += list(sa.norm_q.parameters()) + list(sa.norm_k.parameters())
        return params

    def trainable_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """New modules (camera / pose embeddings) and slow parameters (LoRA + self-attention base weights)."""
        new = list(self.trainable_modules().parameters())
        slow = [p for n, p in self.named_parameters() if 'lora_' in n] + self.trainable_attention()
        return new, slow

    # ---- per-view inputs -------------------------------------------------------------------------------------
    def _stream_embedding(self, plucker: torch.Tensor, view_ids: tuple[int, ...], role: int, T_anchor_front: torch.Tensor) -> torch.Tensor:
        B, V, T, _, H, W = plucker.shape
        tokens = T * H * W
        emb = self.plucker_mlp(plucker).permute(0, 1, 2, 4, 5, 3).reshape(B, V, tokens, self.dim)
        emb = emb + self.cam_id_embed(torch.tensor(view_ids, device=plucker.device)).view(1, V, 1, self.dim).to(emb.dtype)
        emb = emb + self.role_embed(torch.full((V,), role, device=plucker.device)).view(1, V, 1, self.dim).to(emb.dtype)
        motion = self.trajectory_mlp(trajectory_vectors(T_anchor_front, T).to(emb.dtype))
        emb = emb + motion[:, None, :, None, :].expand(B, V, T, H * W, self.dim).reshape(B, V, tokens, self.dim)
        return emb.reshape(B * V, tokens, self.dim)

    def _time(self, t_per_view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.force_fp32_time_and_head_modules()
        with torch.amp.autocast('cuda', enabled=False):
            t = t_per_view.reshape(-1).float() * WAN_TIMESTEP_SCALE
            e = self.expert.time_embedding(sinusoidal_embedding_1d(int(self.expert.freq_dim), t).float())
            e0 = self.expert.time_projection(e).unflatten(1, (6, self.dim))
        return e, e0

    def _pose_modulation(self, target_pose: torch.Tensor, view_ids: tuple[int, ...]) -> torch.Tensor:
        """target_pose (B,V,T,6) -> AdaLN offsets (B*V,6,dim)."""
        B, V = target_pose.shape[:2]
        dtype = self.target_pose_mlp[-1].weight.dtype
        pose = self.target_pose_mlp(target_pose.to(dtype).mean(dim=2)).view(B, V, 6, self.dim)
        view = self.target_view_mod_embed(torch.tensor(view_ids, device=pose.device)).to(dtype).view(1, V, 6, self.dim)
        return (pose + view).reshape(B * V, 6, self.dim)

    def _text(self, text_emb: torch.Tensor) -> torch.Tensor:
        text_len = int(self.expert.text_len)
        if text_emb.shape[1] < text_len:
            text_emb = torch.cat([text_emb, text_emb.new_zeros(text_emb.shape[0], text_len - text_emb.shape[1], text_emb.shape[2])], dim=1)
        return self.expert.text_embedding(text_emb[:, :text_len])

    # ---- Wan block, VACE block and head with one modulation per view -------------------------------------------
    # The forward of Wan2.1's attention block, VACE block and head (Apache-2.0, see THIRD_PARTY_NOTICES), with the modulation per view:
    # tokens of one view share a timestep, so the modulation is (B,V,6,dim) and broadcast over the view's tokens.
    @staticmethod
    def _mod(modulation: torch.Tensor, e_view: torch.Tensor, n: int) -> list[torch.Tensor]:
        with torch.amp.autocast('cuda', enabled=False):
            return list((modulation.float().unsqueeze(1) + e_view.float()).chunk(n, dim=2))

    @staticmethod
    def _per_view(x: torch.Tensor, V: int, scale: torch.Tensor | None = None, shift: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        y = x.view(B, V, L // V, D)
        if scale is not None:
            y = y * scale
        if shift is not None:
            y = y + shift
        return y.reshape(B, L, D)

    @staticmethod
    def _graph_self_attention(attn: nn.Module, x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, V: int, graph) -> torch.Tensor:
        """Wan self-attention (same projections, q/k norms and RoPE) restricted to a view graph."""
        b, s = x.shape[:2]
        n, d = attn.num_heads, attn.head_dim
        q = rope_apply(attn.norm_q(attn.q(x)).view(b, s, n, d), grid_sizes, freqs)
        k = rope_apply(attn.norm_k(attn.k(x)).view(b, s, n, d), grid_sizes, freqs)
        v = attn.v(x).view(b, s, n, d)
        out = view_graph_attention(*(t.transpose(1, 2).to(torch.bfloat16) for t in (q, k, v)), V, graph)
        return attn.o(out.transpose(1, 2).flatten(2))

    def _block(self, block, x, e_view, kwargs, hints):
        V = e_view.shape[1]
        ee = self._mod(block.modulation, e_view, 6)
        inp = self._per_view(block.norm1(x).float(), V, 1 + ee[1], ee[0])
        if kwargs['graph'] is not None:
            y = self._graph_self_attention(block.self_attn, inp, kwargs['grid_sizes'], kwargs['freqs'], V, kwargs['graph'])
        else:
            y = block.self_attn(inp, kwargs['seq_lens'], kwargs['grid_sizes'], kwargs['freqs'])
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + self._per_view(y, V, ee[2])
        x = x + block.cross_attn(block.norm3(x), kwargs['context'], None)
        y = block.ffn(self._per_view(block.norm2(x).float(), V, 1 + ee[4], ee[3]))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + self._per_view(y, V, ee[5])
        if hints is not None and getattr(block, 'block_id', None) is not None:
            x = x + hints[block.block_id]
        return x

    def _run(self, fn, x):
        return checkpoint(fn, x, use_reentrant=False) if self.gradient_checkpointing and self.training else fn(x)

    def _vace_hints(self, c, x, e_view, kwargs):
        hints = []
        for block in self.expert.vace_blocks:
            if block.block_id == 0:
                c = block.before_proj(c) + x

            def run(cc, block=block):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    return self._block(block, cc, e_view, kwargs, None)
            c = self._run(run, c)
            hints.append(block.after_proj(c))
        return hints

    def _head(self, x, e_head_view):
        head = self.expert.head
        m = self._mod(head.modulation, e_head_view.unsqueeze(2), 2)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return head.head(self._per_view(head.norm(x), e_head_view.shape[1], 1 + m[1], m[0]))

    def forward(self, task: str, z_tgt, sigma_tgt, z_ctx, text_emb, plucker_tgt, plucker_ctx, T_anchor_front, pose_tgt, vace_tgt,
                vace_ctx) -> torch.Tensor:
        """Target tensors in TASK_VIEWS[task] order, context in CONTEXT_VIEWS order.
        z_tgt (B,Vt,16,T,H,W), sigma_tgt (B,Vt), z_ctx (B,1,16,T,H,W), plucker_* (B,V,T,6,H/2,W/2), pose_tgt (B,Vt,T,6),
        vace_* (B,V,96,T,H,W). Returns the target velocities (B,Vt,16,T,H,W)."""
        views = TASK_VIEWS[task]
        B, Vt, _, T, H, W = z_tgt.shape
        Vc = z_ctx.shape[1]
        V = Vt + Vc
        h, w = H // 2, W // 2
        tokens = T * h * w
        z = torch.cat([z_tgt, z_ctx.to(z_tgt.dtype)], 1)
        x = self.expert.patch_embedding(z.reshape(B * V, *z.shape[2:]).to(self.expert.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        cond_tgt = self._stream_embedding(plucker_tgt.to(x.dtype), views, ROLE_TARGET, T_anchor_front).view(B, Vt, -1, self.dim)
        cond_ctx = self._stream_embedding(plucker_ctx.to(x.dtype), CONTEXT_VIEWS, ROLE_CONDITION, T_anchor_front).view(B, Vc, -1, self.dim)
        cond = torch.cat([cond_tgt, cond_ctx], 1).to(x.dtype)
        x = (x.view(B, V, -1, self.dim) + cond).reshape(B, V * tokens, self.dim)          # views laid end to end in time
        L = x.shape[1]
        t = torch.cat([sigma_tgt.to(x.device).float(), torch.zeros(B, Vc, device=x.device)], 1)
        e, e0 = self._time(t)
        e0 = e0.view(B, V, 6, self.dim).clone()
        e0[:, :Vt] = e0[:, :Vt] + self._pose_modulation(pose_tgt, views).view(B, Vt, 6, self.dim).to(e0.dtype)
        graph = TASK_GRAPH[task]
        kwargs = {'seq_lens': torch.full((B,), L, dtype=torch.long, device=x.device),
                  'grid_sizes': torch.tensor([[V * T, h, w]] * B, dtype=torch.long, device=x.device),
                  'freqs': self.expert.freqs.to(x.device), 'context': self._text(text_emb), 'graph': graph}
        vace = torch.cat([vace_tgt, vace_ctx.to(vace_tgt.dtype)], 1)
        c = self.expert.vace_patch_embedding(vace.reshape(B * V, *vace.shape[2:]).to(self.expert.vace_patch_embedding.weight.dtype))
        c = c.flatten(2).transpose(1, 2).reshape(B, L, self.dim) + cond.reshape(B, L, self.dim)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hints = self._vace_hints(c, x, e0, kwargs)
        for block in self.expert.blocks:
            def run(xx, block=block):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    return self._block(block, xx, e0, kwargs, hints)
            x = self._run(run, x)
        out = self._head(x, e.view(B, V, self.dim))
        tgt = out.view(B, V, tokens, -1)[:, :Vt].reshape(B * Vt, tokens, -1)
        vid = self.expert.unpatchify(list(tgt), torch.tensor([[T, h, w]] * (B * Vt), dtype=torch.long, device=x.device))
        return torch.stack(vid, 0).view(B, Vt, *vid[0].shape)


def prepare_inputs(task: str, z_front: torch.Tensor, K: torch.Tensor, E: torch.Tensor, T_anchor_front: torch.Tensor,
                   prior_latents: torch.Tensor, prior_visibility: torch.Tensor) -> dict[str, torch.Tensor]:
    """Everything the model needs besides the target latents, sigmas and text. z_front (B,16,T,H,W); K, E, T_anchor_front
    batched; prior_* (B,Vt,16|1,T,H,W) in TASK_VIEWS[task] order: the depth warp ('cross') or the ground prior ('rear'),
    which fills the VACE reactive slot of each target with its visibility as the mask; the front sits in the inactive slot."""
    views = TASK_VIEWS[task]
    B, _, T, H, W = z_front.shape
    Vt = len(views)
    dt = z_front.dtype
    plucker_tgt = plucker_streams(K, E, T_anchor_front, torch.tensor(views, device=z_front.device), T, H // 2, W // 2, dtype=dt)
    plucker_ctx = plucker_streams(K, E, T_anchor_front, torch.tensor(CONTEXT_VIEWS, device=z_front.device), T, H // 2, W // 2, dtype=dt)
    T_anchor_cam = anchor_camera_transforms(E, T_anchor_front)
    pose_tgt = torch.stack([target_pose_features(T_anchor_cam, v, dtype=dt) for v in views], 1)
    vace_tgt = torch.cat([z_front.new_zeros(B, Vt, 16, T, H, W), prior_latents.to(dt), prior_visibility.to(dt).expand(B, Vt, 64, T, H, W)], 2)
    z_ctx = z_front[:, None]
    vace_ctx = torch.cat([z_ctx, z_front.new_zeros(B, 1, 16, T, H, W), z_front.new_zeros(B, 1, 64, T, H, W)], 2)
    return {'z_ctx': z_ctx, 'plucker_tgt': plucker_tgt, 'plucker_ctx': plucker_ctx, 'T_anchor_front': T_anchor_front,
            'pose_tgt': pose_tgt, 'vace_tgt': vace_tgt, 'vace_ctx': vace_ctx}


def run_model(model: SurroundDiT, task: str, x: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor,
              inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(task, x, sigma, inputs['z_ctx'], text_emb, inputs['plucker_tgt'], inputs['plucker_ctx'], inputs['T_anchor_front'],
                 inputs['pose_tgt'], inputs['vace_tgt'], inputs['vace_ctx'])


# ---- construction / checkpoints --------------------------------------------------------------------------
def build_model(device: torch.device, vace_dir: Path = WAN21_VACE_DIR, config: ModelConfig = MODEL) -> SurroundDiT:
    """Wan2.1-VACE + LoRA + the camera / pose embeddings, in bf16, freshly initialised."""
    expert = load_vace_expert(vace_dir, device)
    inject_lora(expert, config.lora_rank, config.lora_alpha)
    model = SurroundDiT(expert).to(device=device, dtype=torch.bfloat16)
    model.force_fp32_time_and_head_modules()
    return model


def load_weights(model: SurroundDiT, ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load trainable.pt of a checkpoint directory: every trained tensor (the rest is the Wan2.1-VACE base model)."""
    state = torch.load(Path(ckpt_dir) / 'trainable.pt', map_location='cpu', weights_only=True)
    params = dict(model.named_parameters())
    trained_ids = {id(p) for p in sum(model.trainable_parameters(), [])}
    trained = {name for name, p in params.items() if id(p) in trained_ids}
    unexpected, missing = sorted(set(state) - set(params)), sorted(trained - set(state))
    if unexpected or missing:
        raise KeyError(f'{ckpt_dir}/trainable.pt does not match the model: unexpected {unexpected[:3]}, missing {missing[:3]}')
    model.load_state_dict(state, strict=False)
    return state


def load_model(ckpt_dir: Path, device: torch.device, vace_dir: Path = WAN21_VACE_DIR) -> tuple[SurroundDiT, dict]:
    model = build_model(device, vace_dir)
    load_weights(model, ckpt_dir)
    meta_path = Path(ckpt_dir) / 'metadata.json'
    return model.eval().requires_grad_(False), json.loads(meta_path.read_text()) if meta_path.exists() else {}
