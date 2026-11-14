"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-boxer
--download-only``) both use it. ``STORE.ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* ``CHECKPOINTS``: Boxer's checkpoints (Hugging Face facebook/boxer), name -> (SHA-256, size in bytes), in the node's
  cache directory (``~/.cache/xblick/boxer``, ``--weights-dir``): BoxerNet, its DINOv3 backbone, OWLv2. Boxer's own
  code would read them from its ``ckpts/``; ``boxer.py`` points it at the cache directory (``use_checkpoints``)
* ``needed()``: all of them (every start reads them)

Boxer's source (the submodule third_party/boxer) is code, not weights: ``tools/setup.sh`` fetches it (``REQUIRED``:
checked first). Its ``ckpts/`` was the checkpoints' place before 0.2.4 (``FORMER``).

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``.
"""

from __future__ import annotations

from pathlib import Path

from xblick.weights import HuggingFace, Store, Weight

PROG = "xblick-boxer"
NODE = "boxer"  # the short name: its cache directory is ~/.cache/xblick/boxer
REPO = Path(__file__).resolve().parents[2]
BOXER_ROOT = REPO / "third_party" / "boxer"
FORMER = BOXER_ROOT / "ckpts"  # the weights' place before 0.2.4 (the checkout): not used, the store names the move
REQUIRED = ((BOXER_ROOT / "boxernet" / "boxernet.py", "Boxer's source, the submodule third_party/boxer"),)  # setup.sh
CHECKPOINT = "boxernet_hw960in2x6d768-c88128f8.ckpt"  # BoxerNet
OWL = "owlv2-base-patch16-ensemble.pt"  # OWLv2 (Boxer writes its text-embedding caches beside it)
# facebook/boxer: the commit its main was on (2026-11-14); the files there have the SHA-256 of CHECKPOINTS
REVISION = "bfba91291fb3f137c18eb6e45362eeda827eac7b"
CHECKPOINTS = {  # name -> (SHA-256, size)
    CHECKPOINT: ("d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f", 343_374_266),
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth": (
        "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea", 114_924_783),
    OWL: ("14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf", 742_984_963),
}
# The terms of what is downloaded, printed before the download (whoever downloads is the one who uses the model).
TERMS = """\
Boxer is the work of Meta, under its terms, and you are the one who uses it:
  * Boxer's code (github.com/facebookresearch/boxer) and its checkpoints (Hugging Face facebook/boxer) are under
    the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0).
    You may not use them for commercial purposes, and you must give credit to Boxer's authors."""


def needed() -> list[Weight]:
    """Every checkpoint a start reads."""
    return [Weight(Path(name), HuggingFace("facebook/boxer", REVISION, name), size, sha256)
            for name, (sha256, size) in CHECKPOINTS.items()]


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return [] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")


# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = Store(PROG, node=NODE, terms=TERMS, required=REQUIRED, repo=REPO, former=[FORMER])
