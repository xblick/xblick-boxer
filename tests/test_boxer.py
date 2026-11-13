"""Boxer on a simulated frame (tests/data/sim_ycb*.png: a simulated head camera over a table with YCB objects, RTX,
2026-10-28; colour, depth in mm, and the camera's pose of that frame): its 3D boxes, through the pose into the world,
lie where the YCB objects stand. This repository does not contain that frame, so the two tests that need it skip.
The semi-dense points use a synthetic depth image. Needs a GPU and Boxer's checkpoints (README); skipped otherwise."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from xblick_boxer.boxer import BOXER_ROOT, CHECKPOINT

DATA = Path(__file__).parent / "data"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    pytest.mark.skipif(not (BOXER_ROOT / "ckpts" / CHECKPOINT).exists(), reason="no Boxer checkpoints (README)"),
]
needs_frame = pytest.mark.skipif(not (DATA / "sim_ycb.png").exists(),
                                 reason="tests/data/sim_ycb*.png is not in this repository")
# the objects' places on the table of that simulation (YCB) [m, world]
YCB = {"cracker box": (0.26, 0.66), "sugar box": (-0.05, 0.60), "tomato soup can": (0.12, 0.47),
       "mustard bottle": (-0.27, 0.76)}


@needs_frame
def test_boxes_where_the_objects_stand():
    from xblick_boxer.boxer import Boxer

    rgb = np.asarray(Image.open(DATA / "sim_ycb.png").convert("RGB"))
    depth = np.asarray(Image.open(DATA / "sim_ycb.depth.png")).astype(np.float32) / 1000.0
    h, w = rgb.shape[:2]
    f = 0.5 * h / math.tan(math.radians(74.3) / 2.0)  # the server's --vfov 74.3
    # the head camera's pose in that frame (the server's log: eye, look; no roll)
    eye = np.array([0.032, 0.313, 1.514])
    look = np.array([-0.03, 0.701, -0.713])
    look /= np.linalg.norm(look)
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    r_wc = np.stack([right, np.cross(look, right), look], axis=1)
    result = Boxer(list(YCB), thresh2d=0.1, thresh3d=0.3)(rgb, (f, f, (w - 1) / 2, (h - 1) / 2), r_wc, eye, depth)
    near = set()
    for b in result.boxes3d:
        x, y, _ = r_wc @ b.center + eye
        place, off = min(((k, math.hypot(x - px, y - py)) for k, (px, py) in YCB.items()), key=lambda t: t[1])
        if off < 0.02:
            near.add(place)
        assert 0.03 < b.size.max() < 0.4, b  # object-sized
    # 2026-10-28: the mustard bottle (0.6 cm), the sugar box (0.4 cm) and the cracker box (0.6 cm, which OWLv2 calls a
    # sugar box) are found; OWLv2 misses the soup can at this threshold
    assert len(near) >= 3, (near, result)


def test_sdp_from_depth_as_boxers():
    """The GPU port gives Boxer's own (NumPy) semi-dense points (num_samples above the valid pixels: no random
    choice)."""
    from xblick_boxer.boxer import _import_boxer, sdp_from_depth

    _import_boxer()
    from loaders.base_loader import BaseLoader

    depth = np.random.default_rng(0).uniform(0.3, 3.0, (480, 640)).astype(np.float32)  # synthetic, metres
    depth[:40] = 0.0  # pixels without depth
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    translation = np.array([0.1, 0.2, 1.5], np.float32)
    k = (500.0, 501.0, depth.shape[1] / 2, depth.shape[0] / 2)
    samples = int((depth > 0).sum())  # every valid pixel of the grid fits
    expected = BaseLoader.sdp_from_depth(depth, *k, rotation, translation, num_samples=samples)
    got = sdp_from_depth(torch.as_tensor(depth, device="cuda"), *k, rotation, translation, num_samples=samples)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got.cpu().numpy(), expected.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)
    assert sdp_from_depth(torch.zeros(4, 4, device="cuda"), *k, rotation, translation).shape == (0, 3)


@needs_frame
def test_labels_and_the_2d_threshold_change_while_it_runs():
    """New labels (the node's live parameter labels) are OWLv2's prompts from the next detection on; the 2D threshold
    is OWLv2's own."""
    from xblick_boxer.boxer import Boxer

    rgb = np.asarray(Image.open(DATA / "sim_ycb.png").convert("RGB"))
    boxer = Boxer(list(YCB), thresh2d=0.1, thresh3d=0.3)
    labels = lambda: {b.label for b in boxer.detect(rgb).boxes2d}  # noqa: E731
    assert labels() and labels() <= set(YCB)
    boxer.set_labels(["mustard bottle"])
    assert labels() == {"mustard bottle"}
    boxer.thresh2d = 0.99
    assert boxer.owl.min_confidence == 0.99 and labels() == set()
    with pytest.raises(ValueError):
        boxer.set_labels([])
