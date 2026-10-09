"""Training data: the multi-camera latent cache and its per-clip sidecars.

Layout of a cache root:
    per_uuid/<uuid>/clip_<id>.pt           {'z_all': (6,16,11,60,104) Wan-VAE latents of the six cameras, 'K': (6,3,3),
                                            'E': (6,4,4) camera->rig, 'uuid', 'cache_version', ...}
    per_uuid/<uuid>/clip_<id>_p4.pt        {'T_anchor_front': (11,4,4)} front-camera ego poses
    per_uuid/<uuid>/clip_<id>_warp.pt      stage 1: the front warped into the cross cameras
    per_uuid/<uuid>/clip_<id>_gprior.pt    stage 2: ground prior of the three rear cameras
    per_uuid/<uuid>/clip_<id>_rearlane.pt  stage 2: lane-marking mask of the GT rear views
    text_cache/null.pt, text_cache/per_uuid/<uuid>.pt   UMT5 caption embeddings {'text_emb', 'text_mask'}
    index.jsonl                            optional clip list ({"path": "per_uuid/<uuid>/clip_<id>.pt"} per line)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from openlongtail.config import CACHE_VERSION, TEXT_EMB_ROOT
from openlongtail.data.ground_prior import GROUND_PRIOR_ORDER
from openlongtail.data.warp import load_warp
from openlongtail.models.surround import TASK_VIEWS


def read_index(root: Path, index_name: str = 'index.jsonl') -> list[dict[str, Any]]:
    """Clip entries of a cache root: index.jsonl if present, otherwise every per_uuid/*/clip_<n>.pt."""
    index = root / index_name
    if index.exists():
        return [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    files = sorted(p for p in (root / 'per_uuid').glob('*/clip_*.pt') if re.fullmatch(r'clip_\d+', p.stem))
    return [{'path': str(p.relative_to(root))} for p in files]


def sidecar(clip_path: Path, suffix: str) -> Path:
    return clip_path.with_name(f'{clip_path.stem}_{suffix}.pt')


def load_text_embedding(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return payload['text_emb'].to(dtype=torch.bfloat16), payload['text_mask'].to(dtype=torch.bool)


def null_text_embedding(cache_root: Path | None = None) -> torch.Tensor:
    """UMT5 embedding (L,4096) of the empty prompt: <cache_root>/text_cache/null.pt, else the one shipped with the demo
    clips, else encoded on the spot."""
    for path in ([cache_root / 'text_cache' / 'null.pt'] if cache_root is not None else []) + [TEXT_EMB_ROOT / 'null.pt']:
        if path.exists():
            return load_text_embedding(path)[0]
    from openlongtail.models.wan import UMT5Encoder
    return UMT5Encoder('cuda').encode('')[0]


def caption_embedding(cache_root: Path, uuid: str, null: torch.Tensor) -> torch.Tensor:
    """Caption embedding of a cache clip, or `null` when the scene has none."""
    path = cache_root / 'text_cache' / 'per_uuid' / f'{uuid}.pt'
    return load_text_embedding(path)[0] if path.exists() else null


class LatentClipDataset(Dataset):
    """Latents, calibration, ego poses and caption embedding of each clip. Only clips with every sidecar in
    `require` are kept."""

    def __init__(self, root: Path, require: tuple[str, ...] = ('p4',), text_drop_prob: float = 0.1,
                 index_name: str = 'index.jsonl', max_items: int | None = None) -> None:
        self.root, self.text_drop_prob = Path(root), text_drop_prob
        entries = [e for e in read_index(self.root, index_name) if all(sidecar(self.root / e['path'], s).exists() for s in require)]
        if max_items is not None:
            entries = entries[:max_items]
        if not entries:
            raise FileNotFoundError(f'no clips with sidecars {require} under {self.root}')
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def clip_path(self, idx: int) -> Path:
        return self.root / self.entries[idx]['path']

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.clip_path(idx)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('cache_version') != CACHE_VERSION:
            raise ValueError(f'expected cache_version {CACHE_VERSION!r} in {path}, got {payload.get("cache_version")!r}')
        uuid = str(payload['uuid'])
        per_uuid = self.root / 'text_cache' / 'per_uuid' / f'{uuid}.pt'
        drop = random.random() < self.text_drop_prob
        text_emb, text_mask = load_text_embedding(self.root / 'text_cache' / 'null.pt' if drop or not per_uuid.exists() else per_uuid)
        return {'uuid': uuid, 'z_all': payload['z_all'].to(dtype=torch.bfloat16), 'K': payload['K'].float(), 'E': payload['E'].float(),
                'T_anchor_front': torch.load(sidecar(path, 'p4'), map_location='cpu', weights_only=True)['T_anchor_front'].float(),
                'text_emb': text_emb, 'text_mask': text_mask}


class SurroundClipDataset(LatentClipDataset):
    """Training clips with the conditions of both tasks: the front warped into the cross cameras, the ground prior of the
    rear cameras and the lane mask of the GT rear views, the rear ones reordered from GROUND_PRIOR_ORDER to the model's."""

    def __init__(self, root: Path, **kwargs) -> None:
        super().__init__(root, require=('p4', 'warp', 'gprior', 'rearlane'), **kwargs)
        self.to_sequence = [GROUND_PRIOR_ORDER.index(v) for v in TASK_VIEWS['rear']]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        path = self.clip_path(idx)
        item['warp_latents'], item['warp_visibility'] = load_warp(sidecar(path, 'warp'))
        prior = torch.load(sidecar(path, 'gprior'), map_location='cpu', weights_only=True)
        item['ground_latents'] = prior['prior_latents'][self.to_sequence].to(torch.bfloat16)
        item['ground_visibility'] = prior['prior_visibility'][self.to_sequence].to(torch.float16)
        item['rear_lane'] = torch.load(sidecar(path, 'rearlane'), map_location='cpu', weights_only=True)['lane'][self.to_sequence]
        return item


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack every tensor field; caption embeddings are zero-padded to the longest one."""
    batch: dict[str, Any] = {'uuid': [it['uuid'] for it in items]}
    length = max(it['text_emb'].shape[0] for it in items)
    batch['text_emb'] = torch.zeros(len(items), length, items[0]['text_emb'].shape[-1], dtype=items[0]['text_emb'].dtype)
    batch['text_mask'] = torch.zeros(len(items), length, dtype=torch.bool)
    for i, it in enumerate(items):
        batch['text_emb'][i, :it['text_emb'].shape[0]] = it['text_emb']
        batch['text_mask'][i, :it['text_mask'].shape[0]] = it['text_mask']
    for key in items[0]:
        if key not in batch:
            batch[key] = torch.stack([it[key] for it in items])
    return batch


def list_cache_clips(root: Path, max_clips: int = -1, shard_index: int = 0, num_shards: int = 1,
                     first_clip_per_uuid: bool = False) -> list[tuple[str, str, Path]]:
    """(uuid, clip id, path) of every clip of a cache root, in a fixed order, for inference."""
    clips = []
    for uuid_dir in sorted(p for p in (root / 'per_uuid').iterdir() if p.is_dir()):
        paths = sorted(p for p in uuid_dir.glob('clip_*.pt') if re.fullmatch(r'clip_\d+', p.stem))
        clips += [(uuid_dir.name, p.stem, p) for p in (paths[:1] if first_clip_per_uuid else paths)]
    clips = clips[shard_index::num_shards]
    return clips[:max_clips] if max_clips > 0 else clips
