#!/usr/bin/env bash
# Shared environment, sourced by every script:  source scripts/env.sh
# Model paths derive from OLT_CKPT_ROOT (see openlongtail/config.py for the per-path OPENLONGTAIL_* overrides).

OLT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export OLT_ROOT
export OLT_CKPT_ROOT="${OLT_CKPT_ROOT:-$OLT_ROOT/checkpoints}"
export PYTHONPATH="$OLT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON="${PYTHON:-python}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
