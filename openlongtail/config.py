"""Paths, camera rig layout and the model configuration.

Model paths derive from OLT_CKPT_ROOT (default: checkpoints/); the OPENLONGTAIL_* variables below override single paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


CHECKPOINT_ROOT = _env_path('OLT_CKPT_ROOT', REPO_ROOT / 'checkpoints')
WAN21_VACE_DIR = _env_path('OPENLONGTAIL_WAN21_VACE_DIR', CHECKPOINT_ROOT / 'Wan2.1-VACE-1.3B')
WAN21_CODE_ROOT = _env_path('OPENLONGTAIL_WAN21_CODE_ROOT', CHECKPOINT_ROOT / 'Wan2.1')
WAN22_CODE_ROOT = _env_path('OPENLONGTAIL_WAN22_CODE_ROOT', CHECKPOINT_ROOT / 'Wan2.2')
VAE_PATH = _env_path('OPENLONGTAIL_VAE_PATH', WAN21_VACE_DIR / 'Wan2.1_VAE.pth')
UMT5_PATH = _env_path('OPENLONGTAIL_UMT5_PATH', WAN21_VACE_DIR / 'models_t5_umt5-xxl-enc-bf16.pth')
UMT5_TOKENIZER = _env_path('OPENLONGTAIL_UMT5_TOKENIZER', WAN21_VACE_DIR / 'google' / 'umt5-xxl')
# Directory holding null.pt, the UMT5 embedding of the empty prompt (the unconditional branch of CFG).
TEXT_EMB_ROOT = _env_path('OPENLONGTAIL_TEXT_EMB_ROOT', REPO_ROOT / 'demo' / 'cached' / 'text_cache')
MODEL_CKPT = CHECKPOINT_ROOT / 'openlongtail'                  # trainable.pt + metadata.json
REFERENCE_RIG = REPO_ROOT / 'assets' / 'reference_rig.pt'

# ---- camera rig ------------------------------------------------------------------------------------------
# View ids index the six cameras of the rig; K / E tensors are always (6, ...) in this order.
FRONT = 0
CROSS_VIEWS = (1, 2)
REAR_VIEWS = (3, 4, 5)
VIEW_NAMES = {0: 'front', 1: 'cross_left', 2: 'cross_right', 3: 'rear_left', 4: 'rear_right', 5: 'rear_tele'}
SENSOR_ORDER = ('camera_front_wide_120fov_undistorted', 'camera_cross_left_120fov_undistorted',
                'camera_cross_right_120fov_undistorted', 'camera_rear_left_70fov_undistorted',
                'camera_rear_right_70fov_undistorted', 'camera_rear_tele_30fov_undistorted')

# ---- clip format -----------------------------------------------------------------------------------------
# 41 RGB frames at 480x832 -> Wan-VAE latents (16, 11, 60, 104); latent frame k covers RGB frames 4k-3 .. 4k.
NUM_FRAMES = 41
CACHE_VERSION = 'latent_t41_stride1_v1'


@dataclass(frozen=True)
class ModelConfig:
    """LoRA on the Wan2.1-VACE-1.3B self-attention and the flow-matching timestep shift."""
    lora_rank: int = 32
    lora_alpha: int = 16
    flow_shift: float = 3.0


MODEL = ModelConfig()
