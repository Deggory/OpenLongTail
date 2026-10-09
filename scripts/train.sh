#!/usr/bin/env bash
# Training on both tasks, starting from the released checkpoint (the defaults are the settings it was trained with:
# 40k steps, half of them on each task, lane weight 4.0).
#
#   LATENT_CACHE_ROOT=data/openlongtail_cache bash scripts/train.sh
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 bash scripts/train.sh      # + NODE_RANK=1 on node1
#
# Resumes from the newest checkpoint in OUTPUT_DIR; extra arguments go to openlongtail.training.train
# (e.g. --init-from none to start from Wan2.1-VACE instead of the released checkpoint).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-$OLT_ROOT/data/openlongtail_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$OLT_ROOT/outputs/train}"
exec $PYTHON -m torch.distributed.run --nnodes "${NNODES:-1}" --node_rank "${NODE_RANK:-0}" --nproc_per_node "${NUM_GPUS:-8}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29531}" \
  -m openlongtail.training.train --latent-cache-root "$LATENT_CACHE_ROOT" --output-dir "$OUTPUT_DIR" "$@"
