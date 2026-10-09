#!/usr/bin/env bash
# One-click demo: the five surround cameras from a front-camera video, stage 1 (cross views) then stage 2 (rear views),
# and an HTML gallery.
#
#   bash scripts/demo.sh                      # the demo clip, with a PRED-vs-GT gallery
#   bash scripts/demo.sh --gpus 0,1,2         # several cached clips under demo/cached, one per GPU
#   bash scripts/demo.sh --raw my_front.mp4   # your own front-camera video (depth + ego pose are estimated)
#   bash scripts/demo.sh --raw my_front.mp4 --synthetic-pose   # skip MapAnything, assume constant forward motion
#
# About 10-15 min per clip on one H200. Outputs go to demo/outputs/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$OLT_ROOT"

RAW="" GPUS="0" POSE=mapanything
while [ $# -gt 0 ]; do case "$1" in
  --raw) RAW="$2"; shift 2 ;;
  --gpus|--gpu) GPUS="$2"; shift 2 ;;
  --synthetic-pose) POSE=synthetic; shift ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "unknown argument: $1"; exit 2 ;;
esac; done
[ -f "$OLT_CKPT_ROOT/openlongtail/trainable.pt" ] || { echo "weights not found: bash scripts/download.sh"; exit 1; }
OUT="$OLT_ROOT/demo/outputs"

if [ -z "$RAW" ]; then
  RENDERS="$OUT/renders"
  IFS=',' read -ra GPU_LIST <<< "$GPUS"
  N=${#GPU_LIST[@]}
  mkdir -p "$RENDERS"
  pids=()
  for i in "${!GPU_LIST[@]}"; do            # clips are split round-robin over the GPUs
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" bash scripts/infer.sh --input demo/cached --output "$RENDERS" --cached \
      --first-clip-per-uuid --num-shards "$N" --shard-index "$i" > "$RENDERS/shard_$i.log" 2>&1 &
    pids+=("$!")
    echo "shard $i/$N on GPU ${GPU_LIST[$i]} (log: $RENDERS/shard_$i.log)"
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "shard $i failed, see $RENDERS/shard_$i.log"; exit 1; }; done
  $PYTHON tools/make_demo_viz.py --renders "$RENDERS" --gt demo/gt --out "$OUT/gallery" --with-gt
  echo "gallery: $OUT/gallery/index.html"
else
  export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"
  NAME="$(basename "${RAW%.*}")"
  WORK="$OUT/$NAME"
  mkdir -p "$WORK/video"
  cp "$RAW" "$WORK/video/$NAME.mp4"
  $PYTHON tools/prepare_dashcam.py --video-root "$WORK/video" --out-root "$WORK/clips" --pose-mode "$POSE" --overwrite
  CLIP="$WORK/clips/chunk_000/$NAME/clip_000000"
  if [ "$POSE" = mapanything ]; then          # needs the mapanything package (github.com/facebookresearch/map-anything)
    $PYTHON tools/estimate_poses.py --frames "$CLIP/frames" --out "$CLIP/pose.pt"
  fi
  bash scripts/infer.sh --input "$WORK/clips" --output "$WORK/renders"
  $PYTHON tools/make_demo_viz.py --renders "$WORK/renders" --out "$WORK/gallery"
  echo "gallery: $WORK/gallery/index.html"
fi
