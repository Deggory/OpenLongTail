#!/usr/bin/env bash
# Fetch the OpenLongTail checkpoint, the Wan2.1-VACE-1.3B base model, the Wan code and the demo clip.
#   bash scripts/download.sh              # everything
#   bash scripts/download.sh --no-demo    # skip the demo clip
#
# checkpoints/
#   openlongtail/                   the OpenLongTail checkpoint (trainable.pt, metadata.json)
#   Wan2.1-VACE-1.3B/               base model: DiT, VAE, UMT5 text encoder (~18 GB)
#   Wan2.1/  Wan2.2/                Wan code (the DiT modules come from 2.1, the VAE / UMT5 modules from 2.2)
# demo/                             the demo clip (latent cache)
# Files already in place (checkpoints/openlongtail, demo/cached) are not downloaded again.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

WANT_DEMO=1
for arg in "$@"; do case "$arg" in
  --no-demo) WANT_DEMO=0 ;;
  *) echo "unknown argument: $arg"; exit 2 ;;
esac; done

MODEL_REPO="${OLT_MODEL_REPO:-luuuulinnnn/Openlongtail}"   # checkpoint at the root, demo clip under demo/
VACE_REPO="${OLT_VACE_REPO:-Wan-AI/Wan2.1-VACE-1.3B}"

hf_get() { if command -v hf >/dev/null; then command hf download "$@"; else huggingface-cli download "$@"; fi; }
clone() {  # repo dir commit: the Wan code is pinned to the commits the release was verified with
  [ -d "$2/wan" ] || { git clone -q "$1" "$2" && git -C "$2" checkout -q "$3"; }
}

mkdir -p "$OLT_CKPT_ROOT"
[ -f "$OLT_CKPT_ROOT/openlongtail/trainable.pt" ] || hf_get "$MODEL_REPO" trainable.pt metadata.json --local-dir "$OLT_CKPT_ROOT/openlongtail"
hf_get "$VACE_REPO" --local-dir "$OLT_CKPT_ROOT/Wan2.1-VACE-1.3B"
clone https://github.com/Wan-Video/Wan2.1 "$OLT_CKPT_ROOT/Wan2.1" 9737cba9c1c3c4d04b33fcad41c111989865d315
clone https://github.com/Wan-Video/Wan2.2 "$OLT_CKPT_ROOT/Wan2.2" 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
if [ "$WANT_DEMO" = 1 ] && [ ! -d "$OLT_ROOT/demo/cached" ]; then hf_get "$MODEL_REPO" --include "demo/*" --local-dir "$OLT_ROOT"; fi

check() { [ -e "$2" ] && echo "  ok       $1" || echo "  MISSING  $1  ($2)"; }
echo "Checkpoints:"
check "OpenLongTail"     "$OLT_CKPT_ROOT/openlongtail/trainable.pt"
check "Wan2.1-VACE DiT"  "$OLT_CKPT_ROOT/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"
check "Wan VAE"          "$OLT_CKPT_ROOT/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"
check "UMT5"             "$OLT_CKPT_ROOT/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
check "Wan2.1 code"      "$OLT_CKPT_ROOT/Wan2.1/wan/modules/vace_model.py"
check "Wan2.2 code"      "$OLT_CKPT_ROOT/Wan2.2/wan/modules/vae2_1.py"
[ "$WANT_DEMO" = 1 ] && check "demo clip" "$OLT_ROOT/demo/cached/text_cache/null.pt"
echo "Next: bash scripts/demo.sh"
