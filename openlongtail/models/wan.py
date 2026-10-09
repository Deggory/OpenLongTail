"""Wan2.1-VACE backbone, LoRA, Wan VAE and UMT5 text encoder.

The DiT comes from the Wan2.1 code, the VAE and UMT5 encoder from the Wan2.2 code (vae2_1.py / t5.py). Both
checkouts are mounted as lightweight packages so their heavy __init__ imports are skipped and the two `wan`
packages do not clash.
"""
from __future__ import annotations

import importlib
import sys
import types
import warnings
from pathlib import Path

import torch
from torch import nn

from openlongtail.config import UMT5_PATH, UMT5_TOKENIZER, VAE_PATH, WAN21_CODE_ROOT, WAN22_CODE_ROOT

_WAN21 = '_openlongtail_wan21'
warnings.filterwarnings('ignore', message=r'`torch\.cuda\.amp\.autocast', category=FutureWarning)   # raised inside the Wan code
LORA_TARGETS = ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o')


def _mount(package: str, root: Path) -> str:
    if not (root / 'wan' / 'modules').is_dir():
        raise FileNotFoundError(f'expected a Wan checkout at {root} (bash scripts/download.sh)')
    for name, path in ((package, root / 'wan'), (f'{package}.modules', root / 'wan' / 'modules')):
        module = sys.modules.get(name)
        if module is None or not hasattr(module, '__path__'):
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    return package


# ---- Wan2.1 attention: SDPA fallback when flash-attn is missing, lower-memory RoPE ----------------------------
# Adapted from wan/modules/attention.py and wan/modules/model.py of Wan2.1 (Apache-2.0, see THIRD_PARTY_NOTICES);
# rope_apply writes into one preallocated tensor instead of stacking per-sample copies.
def _sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """Wan's flash_attention interface on PyTorch SDPA. q_lens / k_lens are not applied: no sequence here is padded."""
    if q_scale is not None:
        q = q * q_scale
    target_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else dtype
    q, k, v = (t.transpose(1, 2).to(target_dtype) for t in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    num_heads, c = x.size(2), x.size(3) // 2
    split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    out = torch.empty_like(x)
    for idx, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[idx, :seq_len].to(torch.float64).reshape(seq_len, num_heads, -1, 2))
        freqs_i = torch.cat([split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                             split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
        out[idx, :seq_len] = torch.view_as_real(x_i * freqs_i).flatten(2).to(dtype=x.dtype)
        if seq_len < x.shape[1]:
            out[idx, seq_len:] = x[idx, seq_len:]
    return out


def _patch_wan21_attention() -> None:
    package = _mount(_WAN21, WAN21_CODE_ROOT)
    attention = importlib.import_module(f'{package}.modules.attention')
    model = importlib.import_module(f'{package}.modules.model')
    if not (getattr(attention, 'FLASH_ATTN_2_AVAILABLE', False) or getattr(attention, 'FLASH_ATTN_3_AVAILABLE', False)):
        for module in (attention, model, sys.modules[f'{package}.modules']):
            module.flash_attention = _sdpa_attention
        attention.attention = _sdpa_attention
    model.rope_apply = rope_apply


def _reset_rope_freqs(expert: nn.Module, device: torch.device | str) -> None:
    rope_params = importlib.import_module(f'{_WAN21}.modules.model').rope_params
    head_dim = int(expert.dim) // int(expert.num_heads)
    expert.freqs = torch.cat([rope_params(1024, head_dim - 4 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6)),
                              rope_params(1024, 2 * (head_dim // 6))], dim=1).to(device=device)


def load_vace_expert(ckpt_dir: Path, device: torch.device | str, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Frozen Wan2.1-VACE DiT."""
    _patch_wan21_attention()
    VaceWanModel = importlib.import_module(f'{_WAN21}.modules.vace_model').VaceWanModel
    expert = VaceWanModel.from_pretrained(str(ckpt_dir))
    expert.to(device=device, dtype=dtype)
    _reset_rope_freqs(expert, device)
    expert.eval().requires_grad_(False)
    if getattr(expert, 'model_type', None) != 'vace' or int(expert.vace_in_dim) != 96:
        raise ValueError(f'{ckpt_dir} is not a Wan2.1-VACE checkpoint')
    return expert


# ---- LoRA ------------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


def inject_lora(expert: nn.Module, rank: int, alpha: int, targets: tuple[str, ...] = LORA_TARGETS) -> None:
    """Freeze `expert` and wrap every targeted nn.Linear (the self-attention projections) with a trainable LoRA."""
    expert.requires_grad_(False)
    for name, module in [(n, m) for n, m in expert.named_modules() if n.endswith(targets) and isinstance(m, nn.Linear)]:
        parent, child = expert.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1]
        setattr(parent, child, LoRALinear(module, rank=rank, alpha=alpha))


# ---- Wan VAE and UMT5 (Wan2.2 code) ------------------------------------------------------------------------
def load_vae(device: torch.device | str = 'cuda', vae_path: Path = VAE_PATH, dtype: torch.dtype = torch.bfloat16):
    _mount('wan', WAN22_CODE_ROOT)
    Wan2_1_VAE = importlib.import_module('wan.modules.vae2_1').Wan2_1_VAE
    vae = Wan2_1_VAE(vae_pth=str(vae_path), dtype=dtype, device=str(device))
    vae.model.eval().requires_grad_(False)
    return vae


def vae_encode_video(vae, rgb: torch.Tensor) -> torch.Tensor:
    """rgb (T,3,H,W) in 0..255 -> latent (16,T',H/8,W/8)."""
    return vae.encode([rgb.float().permute(1, 0, 2, 3).contiguous() / 127.5 - 1.0])[0]


@torch.no_grad()
def vae_decode_batch(vae, latents: list[torch.Tensor], device: torch.device, batch: int = 3) -> list[torch.Tensor]:
    """Latents (16,11,h,w) -> list of (41,3,H,W) float in 0..255."""
    out = []
    for i in range(0, len(latents), batch):
        z = torch.stack(latents[i:i + batch]).to(device, torch.bfloat16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x = vae.model.decode(z, vae.scale).float().clamp_(-1, 1)
        out += [((v + 1) * 127.5).permute(1, 0, 2, 3) for v in x]
    return out


@torch.no_grad()
def vae_encode_batch(vae, videos: list[torch.Tensor], batch: int = 3) -> torch.Tensor:
    """List of (41,3,H,W) in 0..255 -> (N,16,11,h,w) float."""
    out = []
    for i in range(0, len(videos), batch):
        x = torch.stack([v.permute(1, 0, 2, 3) / 127.5 - 1.0 for v in videos[i:i + batch]])
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out.append(vae.model.encode(x, vae.scale).float())
    return torch.cat(out)


class UMT5Encoder:
    """Prompt -> (text_emb (256,4096) bf16, text_mask (256,) bool), zero-padded like the cached embeddings."""

    def __init__(self, device: str = 'cuda') -> None:
        _mount('wan', WAN22_CODE_ROOT)
        T5EncoderModel = importlib.import_module('wan.modules.t5').T5EncoderModel
        self.device = device
        self.model = T5EncoderModel(text_len=256, dtype=torch.bfloat16, device=torch.device(device),
                                    checkpoint_path=str(UMT5_PATH), tokenizer_path=str(UMT5_TOKENIZER))

    @torch.no_grad()
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.model([prompt], torch.device(self.device))[0].detach().to('cpu', dtype=torch.bfloat16)
        length = min(encoded.shape[0], 256)
        text_emb = torch.zeros(256, 4096, dtype=torch.bfloat16)
        text_mask = torch.zeros(256, dtype=torch.bool)
        text_emb[:length] = encoded[:length]
        text_mask[:length] = True
        return text_emb, text_mask
