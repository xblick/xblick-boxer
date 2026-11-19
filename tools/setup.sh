#!/bin/bash
# Sets xblick-boxer up from scratch or updates it: Boxer's source (the submodule), the venv exactly as uv.lock fixes
# it (.venv), then Boxer's checkpoints through xblick-boxer --download-only (weights.py; 1.2 GB, CC-BY-NC, from
# Hugging Face; only those missing, each checked against its SHA-256) in ~/.cache/xblick/boxer. The node downloads
# missing checkpoints at its start too:
#
#     tools/setup.sh
#
# Before it downloads the checkpoints, xblick-boxer prints the model's terms (weights.TERMS): whoever runs the script
# is the one who uses the model. Boxer's code is under the same terms, so the script names them before the submodule.
#
# Needs uv (UV names it, default uv on PATH) and git: without them it stops with the command that installs them.
# Without an NVIDIA GPU with driver 580 or newer (CUDA 13) it says so and goes on: the node needs one to start,
# --download-only does not. It never asks and never uses sudo. Run again, it does only what changed.
set -euo pipefail

PROG=xblick-boxer
URL=https://github.com/xblick/xblick-boxer.git
MIN_DRIVER=580  # NVIDIA's minimum Linux driver for CUDA 13.0, which the cu130 torch is built for

usage() {
    cat <<'EOF'
Usage: tools/setup.sh

Sets xblick-boxer up in this clone, or updates it. It fetches Boxer's code (the submodule third_party/boxer),
creates .venv exactly as uv.lock fixes it (Python 3.12, PyTorch for CUDA 13, the core xblick from PyPI) and
downloads Boxer's checkpoints into the node's cache directory (~/.cache/xblick/boxer). Boxer's code and checkpoints
are under CC BY-NC 4.0: no commercial use. A second call does only what changed. It never asks and never uses sudo.

Needs git and uv (UV=/path/to/uv names another uv). The node needs an NVIDIA GPU with driver 580 or newer.

  -h, --help  this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in
        -h | --help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $arg" ;;
    esac
done

cd "$(dirname "$0")/.."
UV=${UV:-uv}

# ------------------------------------------------------------------------------------------------- prerequisites

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
git rev-parse --is-inside-work-tree > /dev/null 2>&1 ||
    die "$PWD is not a git clone (a ZIP download has no submodule): git clone --recurse-submodules $URL"

if ! command -v nvidia-smi > /dev/null; then
    echo "$PROG: no NVIDIA driver found (no nvidia-smi). The node needs an NVIDIA GPU with driver $MIN_DRIVER or" \
        "newer to start (Ubuntu: sudo apt install nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
elif ! driver=$(timeout 30 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null) ||
        [[ -z $driver ]]; then
    echo "$PROG: nvidia-smi failed or lists no GPU. The node needs an NVIDIA GPU with driver $MIN_DRIVER or newer" \
        "to start. The setup goes on."
else
    driver=$(cut -d. -f1 <<< "$driver" | sort -n | head -1 | tr -d ' ')
    ((driver >= MIN_DRIVER)) ||
        echo "$PROG: NVIDIA driver $driver. PyTorch for CUDA 13 needs $MIN_DRIVER or newer (Ubuntu: sudo apt install" \
            "nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
fi

# ------------------------------------------------------------------------------------------------- the setup

cat <<EOF
$PROG: setting up $PWD. The first call downloads:
  * Boxer's code, the submodule third_party/boxer (github.com/facebookresearch/boxer)
  * .venv as uv.lock fixes it: Python 3.12, the core xblick, and PyTorch for CUDA 13 with NVIDIA's CUDA libraries
    (NVIDIA's licence, License.txt in each nvidia package); about 3 GB, 6 GB on disk
  * Boxer's checkpoints (1.2 GB, Hugging Face facebook/boxer) into the node's cache directory
Boxer's code and checkpoints are under the Creative Commons Attribution-NonCommercial 4.0 International licence
(CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/): no commercial use, and credit to Boxer's authors.
EOF
git submodule update --init third_party/boxer
"$UV" sync --frozen
.venv/bin/xblick-boxer --download-only
echo "$PROG: set up. The first start: README.md, Install, step 5."
