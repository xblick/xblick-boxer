"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-boxer
--download-only``) both use it. ``STORE.ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* ``CHECKPOINTS``: Boxer's checkpoints (Hugging Face facebook/boxer), name -> (SHA-256, size in bytes), in
  ``third_party/boxer/ckpts`` where Boxer's code reads them: BoxerNet, its DINOv3 backbone, OWLv2
* ``needed()``: all of them (every start reads them)

Boxer's source (the submodule third_party/boxer) is code, not weights: ``tools/setup.sh`` fetches it, and the
checkpoints go into it (``REQUIRED``: without it nothing is downloaded).

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``.
"""

from __future__ import annotations

from pathlib import Path

from xblick.weights import HuggingFace, Store, Weight

PROG = "xblick-boxer"
REPO = Path(__file__).resolve().parents[2]
BOXER_ROOT = REPO / "third_party" / "boxer"
WEIGHTS = BOXER_ROOT / "ckpts"
REQUIRED = ((BOXER_ROOT / "boxernet" / "boxernet.py", "Boxer's source, the submodule third_party/boxer that holds "
                                                       "ckpts/"),)  # (path, what): code from tools/setup.sh
CHECKPOINT = "boxernet_hw960in2x6d768-c88128f8.ckpt"  # BoxerNet
# facebook/boxer: the commit its main was on (2026-11-14); the files there have the SHA-256 of CHECKPOINTS
REVISION = "bfba91291fb3f137c18eb6e45362eeda827eac7b"
CHECKPOINTS = {  # name -> (SHA-256, size)
    CHECKPOINT: ("d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f", 343_374_266),
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth": (
        "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea", 114_924_783),
    "owlv2-base-patch16-ensemble.pt": ("14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf", 742_984_963),
}
# The terms of what is downloaded, printed before the download (whoever downloads is the one who uses the model).
TERMS = """\
Boxer is the work of Meta, under its terms, and you are the one who uses it:
  * Boxer's code (github.com/facebookresearch/boxer) and its checkpoints (Hugging Face facebook/boxer) are under
    the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0).
    You may not use them for commercial purposes, and you must give credit to Boxer's authors."""


def needed() -> list[Weight]:
    """Every checkpoint a start reads."""
    return [Weight(WEIGHTS / name, HuggingFace("facebook/boxer", REVISION, name), size, sha256)
            for name, (sha256, size) in CHECKPOINTS.items()]


STORE = Store(PROG, WEIGHTS, TERMS, REQUIRED, repo=REPO)  # the download (xblick.weights), its lock in WEIGHTS
