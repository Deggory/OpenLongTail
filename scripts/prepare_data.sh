#!/usr/bin/env bash
# Training sidecars for a latent cache (see openlongtail/data/latent_cache.py for the layout):
#   clip_*_warp.pt                      stage 1: the front warped into the cross cameras (needs the source front videos
#                                       and their depth under DATA_ROOT)
#   clip_*_gprior.pt + clip_*_rearlane.pt  stage 2: ground prior and lane mask of the rear cameras (from the cache alone)
#
#   LATENT_CACHE_ROOT=data/openlongtail_cache DATA_ROOT=data/by_uuid bash scripts/prepare_data.sh
#   NUM_SHARDS=8 SHARD_INDEX=0 ... bash scripts/prepare_data.sh   # run SHARD_INDEX=0..7 in parallel (shard i on GPU i)
#   STAGES=2 ... bash scripts/prepare_data.sh                      # only the stage-2 sidecars
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-$OLT_ROOT/data/openlongtail_cache}"
DATA_ROOT="${DATA_ROOT:-$OLT_ROOT/data/by_uuid}"
SHARD=(--num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SHARD_INDEX:-0}}"     # shard i on GPU i unless set
case "${STAGES:-12}" in *1*)
  $PYTHON -m openlongtail.scripts.build_warp_sidecars --latent-cache-root "$LATENT_CACHE_ROOT" --data-root "$DATA_ROOT" "${SHARD[@]}" ;;
esac
case "${STAGES:-12}" in *2*)
  $PYTHON -m openlongtail.scripts.build_ground_prior --latent-cache-root "$LATENT_CACHE_ROOT" "${SHARD[@]}" ;;
esac
