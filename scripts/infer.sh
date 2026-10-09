#!/usr/bin/env bash
# Inference on a folder of clips: stage 1 renders the two cross views, stage 2 the three rear views, with one model.
#
#   bash scripts/infer.sh --input CLIPS --output OUT              # raw clips (tools/prepare_dashcam.py layout)
#   bash scripts/infer.sh --input CACHE --output OUT --cached     # latent-cache clips
#   bash scripts/infer.sh ... --no-caption                        # raw clips: skip the Qwen2.5-VL scene captions
#
# Every clip directory gets pred_{cross_left,cross_right,rear_left,rear_right,rear_tele}.mp4, front_input.mp4 and surround.mp4.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

INPUT="" OUTPUT="" CACHED=0 CAPTION=1 EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --input) INPUT="$2"; shift 2 ;;
  --output) OUTPUT="$2"; shift 2 ;;
  --cached) CACHED=1; shift ;;
  --no-caption) CAPTION=0; shift ;;
  --max-clips|--num-shards|--shard-index|--num-steps|--seed|--cross-guide|--rear-guide|--ckpt|--caption|--splat-radius)
    EXTRA+=("$1" "$2"); shift 2 ;;
  --first-clip-per-uuid|--overwrite) EXTRA+=("$1"); shift ;;
  -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  *) echo "unknown argument: $1"; exit 2 ;;
esac; done
[ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo "usage: bash scripts/infer.sh --input CLIPS --output OUT [--cached]"; exit 2; }

TIER=(); [ "$CACHED" = 1 ] && TIER=(--cached)
if [ "$CACHED" = 0 ] && [ "$CAPTION" = 1 ]; then
  if $PYTHON -m openlongtail.scripts.build_captions --input "$INPUT" --out-root "$OUTPUT/captions"; then
    TIER=(--caption-cache "$OUTPUT/captions")
  else
    echo "captioning failed; continuing without captions"
  fi
fi
$PYTHON -m openlongtail.scripts.infer --input "$INPUT" --output "$OUTPUT" ${TIER[@]+"${TIER[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
