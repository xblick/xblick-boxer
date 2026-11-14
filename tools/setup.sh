#!/bin/bash
# Sets xblick-boxer up from scratch or updates it: Boxer's source (the submodule), its checkpoints (1.2 GB,
# CC-BY-NC, from Hugging Face; only those missing, each checked against its SHA-256), the venv exactly as uv.lock
# fixes it (.venv):
#
#     tools/setup.sh
#
# Before it downloads a model it prints the model's terms: whoever runs the script is the one who uses the model.
#
# Needs uv (UV names it, default uv on PATH), curl and git. Run again, it does only what changed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
cd "$ROOT"

# The terms of what is downloaded, printed before the download. $1: what is missing.
model_terms() {
    cat <<TERMS
xblick-boxer: to download: $1
Boxer is the work of Meta, under its terms, and you are the one who uses it:
  * Boxer's code (github.com/facebookresearch/boxer) and its checkpoints (Hugging Face facebook/boxer) are under
    the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0).
    You may not use them for commercial purposes, and you must give credit to Boxer's authors.
TERMS
}

CKPTS=$ROOT/third_party/boxer/ckpts
NAMES="boxernet_hw960in2x6d768-c88128f8.ckpt dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth owlv2-base-patch16-ensemble.pt"
missing=""
for name in $NAMES; do [ -e "$CKPTS/$name" ] || missing="$missing $name"; done
[ -z "$missing" ] || model_terms "Boxer's code and its checkpoints$missing"
git submodule update --init third_party/boxer
mkdir -p "$CKPTS"
while read -r name sha256; do
  if [ ! -e "$CKPTS/$name" ]; then
    echo "downloading $name"
    curl -fL --retry 3 -o "$CKPTS/$name.part" "https://huggingface.co/facebook/boxer/resolve/main/$name"
    mv "$CKPTS/$name.part" "$CKPTS/$name"
  fi
  echo "$sha256  $CKPTS/$name" | sha256sum --check --quiet - \
    || { echo "$CKPTS/$name: wrong checksum (remove it, then run again)" >&2; exit 1; }
done << 'EOF'
boxernet_hw960in2x6d768-c88128f8.ckpt d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f
dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth 4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea
owlv2-base-patch16-ensemble.pt 14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf
EOF
"$UV" sync --frozen
