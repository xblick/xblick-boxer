#!/bin/bash
# Sets xblick-boxer up from scratch or updates it: Boxer's source (the submodule), the venv exactly as uv.lock fixes
# it (.venv), then Boxer's checkpoints through xblick-boxer --download-only (weights.py; 1.2 GB, CC-BY-NC, from
# Hugging Face; only those missing, each checked against its SHA-256) in ~/.cache/xblick/boxer. The node downloads
# missing checkpoints at its start too:
#
#     tools/setup.sh
#
# Before it downloads the checkpoints, xblick-boxer prints the model's terms (weights.TERMS): whoever runs the script
# is the one who uses the model.
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
cd "$ROOT"

git submodule update --init third_party/boxer
"$UV" sync --frozen
.venv/bin/xblick-boxer --download-only
