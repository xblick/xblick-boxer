"""Meta's Boxer on one image: OWLv2 finds 2D boxes for text prompts, BoxerNet lifts them to 3D boxes (third_party/boxer,
facebookresearch/boxer at a fixed commit; CC-BY-NC 4.0, its checkpoints too).

Boxer works in a world whose z is up (it aligns its frame with gravity from the camera's pose) and on a square image
of BoxerNet's size (960): the image is scaled to it, and the camera model with it, as Boxer's own loaders do. Depth,
if given, becomes Boxer's semi-dense points (its loaders' ``sdp_from_depth``). The boxes come back in the camera's
frame (OpenCV axes: x right, y down, z forward), the 2D boxes in the pixels of the image given.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BOXER_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "boxer"
CHECKPOINT = "boxernet_hw960in2x6d768-c88128f8.ckpt"


def depth_k(k, size, depth_shape) -> tuple[float, float, float, float]:
    """The colour camera's (fx, fy, cx, cy) for its image of ``size`` (w, h) as the depth's of ``depth_shape`` (h, w):
    a depth in the colour camera's geometry at another size (xblick-fast-foundationstereo's with --max-width),
    f and the principal point scaled about the pixels' centres (c' = (c + 0.5) s - 0.5)."""
    fx, fy, cx, cy = k
    w, h = size
    dh, dw = depth_shape
    if (dw, dh) == (w, h):
        return fx, fy, cx, cy
    sx, sy = dw / w, dh / h
    return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5


def sdp_from_depth(depth_m: torch.Tensor, fx: float, fy: float, cx: float, cy: float, r_wc: np.ndarray,
                   t_wc: np.ndarray, num_samples: int = 10000) -> torch.Tensor:
    """Boxer's BaseLoader.sdp_from_depth on the depth's device: the semi-dense points of a depth image ((H, W) float32
    metres, 0: none) in the world, from a grid over the image (step so that it holds about 2 * num_samples pixels),
    at most num_samples of them chosen at random, NaN-padded to num_samples; no points: an empty (0, 3) tensor."""
    h, w = depth_m.shape
    step = max(1, int(np.sqrt(h * w / (num_samples * 2))))
    z = depth_m[::step, ::step]
    ys, xs = torch.meshgrid(torch.arange(0, h, step, device=z.device, dtype=torch.float32),
                            torch.arange(0, w, step, device=z.device, dtype=torch.float32), indexing="ij")
    valid = z > 0
    z, ys, xs = z[valid], ys[valid], xs[valid]
    if z.numel() > num_samples:
        chosen = torch.randperm(z.numel(), device=z.device)[:num_samples]
        z, ys, xs = z[chosen], ys[chosen], xs[chosen]
    if z.numel() == 0:
        return torch.zeros(0, 3, dtype=torch.float32)
    points_c = torch.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], dim=-1)
    points_w = points_c @ torch.as_tensor(r_wc, device=z.device).T + torch.as_tensor(t_wc, device=z.device)
    if len(points_w) < num_samples:
        points_w = torch.cat([points_w, points_w.new_full((num_samples - len(points_w), 3), float("nan"))])
    return points_w


def _import_boxer():
    """Boxer's modules (it is no package: its root goes on sys.path); fails with what to do."""
    if not (BOXER_ROOT / "boxernet" / "boxernet.py").exists():
        raise SystemExit(f"Boxer's source is missing at {BOXER_ROOT}: git submodule update --init")
    missing = [name for name in (CHECKPOINT, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                                 "owlv2-base-patch16-ensemble.pt") if not (BOXER_ROOT / "ckpts" / name).exists()]
    if missing:
        raise SystemExit(f"Boxer's checkpoints are missing in {BOXER_ROOT / 'ckpts'}: {', '.join(missing)} "
                         "(README: download them)")
    if str(BOXER_ROOT) not in sys.path:
        sys.path.insert(0, str(BOXER_ROOT))


@dataclass
class Box2D:
    x1: float  # pixels of the image given
    y1: float
    x2: float
    y2: float
    label: str
    score: float  # OWLv2's


@dataclass
class Box3D:
    center: np.ndarray    # (3,) metres, camera frame
    rotation: np.ndarray  # (3, 3) the box's axes in the camera frame
    size: np.ndarray      # (3,) metres along those axes
    label: str
    score: float          # Boxer's: the mean of OWLv2's and BoxerNet's


@dataclass
class Detections:
    boxes2d: list[Box2D]   # every 2D box OWLv2 kept
    size: tuple[int, int]  # width, height of the image given
    img: torch.Tensor      # (1, 3, hw, hw) the scaled image on the GPU, in [0, 1]
    bb2d: torch.Tensor     # (N, 4) OWLv2's boxes in the scaled image, (x1, x2, y1, y2), on the CPU
    ready: torch.cuda.Event | None  # recorded on the detection stream after img and bb2d (None on the CPU)
    dino: torch.Tensor | None = None  # BoxerNet's DINOv3 features of img (dino_beside_owl), ready with ``ready``


@dataclass
class Result:
    boxes2d: list[Box2D]  # every 2D box OWLv2 kept
    boxes3d: list[Box3D]  # those BoxerNet lifted above its threshold
    # with features: DINOv3's features of the scaled image, (hw/16, hw/16, 384) float16 on the GPU, one per 16 x 16
    # patch (feature_camera() maps them to the image); computed when lift() returns
    features: torch.Tensor | None = None


class Boxer:
    """OWLv2 and BoxerNet, loaded once (on the GPU: about 2 GB)."""

    def __init__(self, labels: list[str], thresh2d: float = 0.2, thresh3d: float = 0.5, device: str = "cuda",
                 dino_beside_owl: bool = False, features: bool = False):
        if not labels:
            raise ValueError("Boxer needs at least one text prompt")
        _import_boxer()
        from boxernet.boxernet import BoxerNet
        from owl.owl_wrapper import OwlWrapper

        self.labels = list(labels)
        self.thresh3d = thresh3d
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        # OWLv2 in bfloat16 where the GPU has it, as Boxer's run_boxer.py: half the time (55 -> 27 ms on an RTX PRO
        # 6000), the same boxes on the test frame
        self.owl = OwlWrapper(device, text_prompts=self.labels, min_confidence=thresh2d,
                              precision="bfloat16" if self.dtype == torch.bfloat16 else "float32")
        self.net = BoxerNet.load_from_checkpoint(str(BOXER_ROOT / "ckpts" / CHECKPOINT), device=device)
        self.hw = int(self.net.hw)
        if self.dtype == torch.bfloat16:
            # the weights of its Linear and Conv2d layers in bfloat16 once, instead of autocast casting the float32 ones
            # in every frame (115 layers, about 300 cast kernels per frame): the same numbers (autocast computes these
            # layers in bfloat16 either way; the test frame's boxes came out bit-identical), on the shared GPU 19.5 ->
            # 17.4 ms for the fastest frame, 32 -> 28 ms median (2026-10-29)
            for module in self.net.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    module.to(torch.bfloat16)
        # one CUDA stream per stage, so that detect() of the next frame overlaps lift() of this one when they run in
        # two threads (on the shared GPU 21-26 -> 27-29 frames/s, measured 2026-10-29)
        self._streams = (torch.cuda.Stream(), torch.cuda.Stream()) if device.startswith("cuda") else None
        # BoxerNet's DINOv3 needs only the image: with dino_beside_owl the detection stage computes it on a third
        # stream while OWLv2 runs, and BoxerNet takes the result through its dino.forward (batch_dino calls that).
        # Shortens the lift stage by DINOv3's 5 ms where the GPU has room for both; on the GPUs shared with other jobs
        # (2026-10-29) it was slower (fastest frame 18 -> 24 ms), so off by default until measured on an idle GPU
        self._dino_forward = self.net.dino.forward
        self._dino_given: tuple[torch.Tensor, torch.Tensor] | None = None  # (img, features) for the next lift
        self.dino_beside_owl = dino_beside_owl
        # DINOv3's features of every frame in the Result (for other nodes, e.g. a semantic map): those BoxerNet
        # computes anyway; for a frame without 2D boxes, which BoxerNet skips, DINOv3 runs for them alone
        self.features = features
        if dino_beside_owl:
            if self._streams is None:
                raise ValueError("dino_beside_owl needs a CUDA device")
            self._dino_stream = torch.cuda.Stream()
            self.net.dino.forward = self._given_dino

    def _given_dino(self, img: torch.Tensor) -> torch.Tensor:
        """BoxerNet's call of DINOv3 with dino_beside_owl: the features detect() computed for this very image."""
        given, self._dino_given = self._dino_given, None
        if given is None or given[0] is not img:
            raise RuntimeError("BoxerNet asked for DINOv3 features of an image detect() did not compute them for")
        return given[1]

    def _on(self, stage: int):
        return torch.cuda.stream(self._streams[stage]) if self._streams else contextlib.nullcontext()

    def __call__(self, rgb, k: tuple[float, float, float, float], rotation_wc: np.ndarray, translation_wc: np.ndarray,
                 depth_m=None) -> Result:
        """rgb: (H, W, 3) uint8; k: fx, fy, cx, cy of that image; the camera's pose in the world (OpenCV axes, the
        world z up); depth_m: (H, W) float32 metres in that camera, 0 where none, or None. Images: NumPy arrays
        (uploaded) or device arrays (torch tensors, CUDA ring frames)."""
        return self.lift(self.detect(rgb), k, rotation_wc, translation_wc, depth_m)

    def set_labels(self, labels: list[str]):
        """New text prompts from the next detection on (OWLv2 encodes them: tens of milliseconds)."""
        if not labels:
            raise ValueError("at least one label")
        self.labels = list(labels)
        self.owl.set_text_prompts(self.labels)

    @property
    def thresh2d(self) -> float:
        return self.owl.min_confidence

    @thresh2d.setter
    def thresh2d(self, value: float):
        self.owl.min_confidence = value

    def detect(self, rgb) -> Detections:
        """The first stage, OWLv2's 2D boxes; needs neither pose nor depth, so it can run on the next frame while
        :meth:`lift` works on this one (the node does: two threads, one CUDA stream each). ``rgb``: (H, W, 3 or 4)
        uint8, RGB(A) order, a NumPy array (uploaded) or a device array (a torch tensor, a CUDA ring frame)."""
        h, w = rgb.shape[:2]
        with self._on(0):
            return self._detect(rgb, w, h)

    def _detect(self, rgb, w: int, h: int) -> Detections:
        # every input on the GPU before BoxerNet sees it (its input check, img.max(), took 17 ms on the CPU), as uint8,
        # becoming float there; taken onto the device under this stage's stream, which a ring frame's copy is ordered
        # before. Resized on the GPU, bilinear (as cv2.INTER_LINEAR, without its rounding to uint8)
        image = torch.as_tensor(rgb, device=self.device)[..., :3].permute(2, 0, 1)[None].float()
        img = torch.nn.functional.interpolate(image, size=(self.hw, self.hw), mode="bilinear",
                                              align_corners=False) / 255.0
        dino = None
        if self.dino_beside_owl:
            uploaded = torch.cuda.Event()
            uploaded.record(self._streams[0])
            self._dino_stream.wait_event(uploaded)
            with torch.cuda.stream(self._dino_stream), torch.autocast(device_type="cuda", dtype=self.dtype):
                img.record_stream(self._dino_stream)
                dino = self._dino_forward(img)  # as BoxerNet's encode calls it (inside its autocast)
            computed = torch.cuda.Event()
            computed.record(self._dino_stream)
            self._streams[0].wait_event(computed)  # ``ready`` below then covers the features too
            dino.record_stream(self._streams[0])
        bb2d, scores2d, label_ids, _ = self.owl.forward(img * 255.0, resize_to_HW=(self.hw, self.hw))
        sx, sy = self.hw / w, self.hw / h
        labels2d = [self.labels[int(i)] for i in label_ids]
        # bb2d: (x1, x2, y1, y2) in the scaled image
        boxes2d = [Box2D(float(b[0]) / sx, float(b[2]) / sy, float(b[1]) / sx, float(b[3]) / sy, label, float(s))
                   for b, s, label in zip(bb2d, scores2d, labels2d, strict=True)]
        ready = None
        if self._streams:
            ready = torch.cuda.Event()
            ready.record(self._streams[0])
        return Detections(boxes2d, (w, h), img, bb2d, ready, dino)

    def lift(self, detections: Detections, k: tuple[float, float, float, float], rotation_wc: np.ndarray,
             translation_wc: np.ndarray, depth_m=None) -> Result:
        """The second stage, BoxerNet's 3D boxes for :meth:`detect`'s 2D boxes; arguments as :meth:`__call__`'s, for
        the image detect was given."""
        boxes2d, bb2d, img = detections.boxes2d, detections.bb2d, detections.img
        if not boxes2d and not self.features:
            return Result([], [])
        caller = torch.cuda.current_stream() if self._streams else None
        with self._on(1):
            if self._streams:
                # the detection stream's work on img first; and img stays allocated until this stream is done with it
                # (the caching allocator would otherwise reuse it for the next detection). bb2d is on the CPU.
                self._streams[1].wait_event(detections.ready)
                img.record_stream(self._streams[1])
                # a depth the caller made on the GPU (the node's) on its stream: that work first, likewise kept
                if isinstance(depth_m, torch.Tensor) and depth_m.is_cuda:
                    self._streams[1].wait_stream(caller)
                    depth_m.record_stream(self._streams[1])
            if self.dino_beside_owl:
                if detections.dino is None:
                    raise ValueError("dino_beside_owl: these detections carry no DINOv3 features")
                detections.dino.record_stream(self._streams[1])
                if not boxes2d:
                    features = _patches(detections.dino)
                    self._streams[1].synchronize()
                    return Result([], [], features)
                self._dino_given = (img, detections.dino)
            if not boxes2d:  # features only
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    features = _patches(self._dino_forward(img))
                if self._streams:
                    self._streams[1].synchronize()  # as _lift's results, which it copies to the CPU
                return Result([], [], features)
            return self._lift(boxes2d, img, bb2d, detections.size, k, rotation_wc, translation_wc, depth_m)

    def feature_camera(self, k: tuple[float, float, float, float], size: tuple[int, int]):
        """fx, fy, cx, cy of the feature grid (Result.features) for an image of ``size`` (w, h) with ``k``: the image
        is scaled to hw x hw, each feature a 16 x 16 patch of that (pixel centres at integers, as camera_info)."""
        fx, fy, cx, cy = k
        w, h = size
        patch = int(self.net.dino.patch_size)
        sx, sy = self.hw / w / patch, self.hw / h / patch
        return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5

    def _lift(self, boxes2d, img, bb2d, size, k, rotation_wc, translation_wc, depth_m) -> Result:
        from loaders.base_loader import BaseLoader
        from utils.tw.pose import PoseTW

        labels2d = [b.label for b in boxes2d]
        scores2d = [b.score for b in boxes2d]
        w, h = size
        fx, fy, cx, cy = k
        sx, sy = self.hw / w, self.hw / h
        r_wc = rotation_wc.astype(np.float32)
        t_wc = translation_wc.astype(np.float32)
        sdp_w = (sdp_from_depth(torch.as_tensor(depth_m, device=self.device),
                                *depth_k(k, size, depth_m.shape[:2]), r_wc, t_wc)
                 if depth_m is not None else torch.zeros(0, 3))
        camera = BaseLoader.pinhole_from_K(self.hw, self.hw, fx * sx, fy * sy, cx * sx, cy * sy)
        datum = {"img0": img,
                 "cam0": camera.to(self.device),
                 "T_world_rig0": PoseTW.from_Rt(torch.from_numpy(r_wc), torch.from_numpy(t_wc)).to(self.device),
                 "sdp_w": sdp_w.to(self.device),
                 "bb2d": bb2d.to(self.device)}
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            out = self.net.forward(datum)
            # the features before the copy to the CPU, which waits for this stream's work: the caller reads them on
            # another stream (publishing copies on the legacy default stream, which does not wait for this one)
            features = _patches(out["dino0"]) if self.features else None
            obbs = out["obbs_pr_w"].cpu()[0]
        scores3d = obbs.prob.squeeze(-1).float()
        boxes3d = []
        for i in range(len(obbs)):
            if float(scores3d[i]) < self.thresh3d:
                continue
            pose = obbs.T_world_object[i]
            r_wo = pose.R.float().numpy()
            t_wo = pose.t.float().numpy()
            bb3 = obbs.bb3_object[i].float().numpy()  # xmin, xmax, ymin, ymax, zmin, zmax in the box's frame
            center_object = np.array([bb3[0] + bb3[1], bb3[2] + bb3[3], bb3[4] + bb3[5]]) / 2.0
            center_world = r_wo @ center_object + t_wo
            # into the camera: p_c = R_wc^T (p_w - t_wc)
            boxes3d.append(Box3D(center=r_wc.T @ (center_world - t_wc), rotation=r_wc.T @ r_wo,
                                 size=np.array([bb3[1] - bb3[0], bb3[3] - bb3[2], bb3[5] - bb3[4]]),
                                 label=labels2d[i], score=(float(scores2d[i]) + float(scores3d[i])) / 2.0))
        return Result(boxes2d, boxes3d, features)


def _patches(dino: torch.Tensor) -> torch.Tensor:
    """(1, C, h, w) DINOv3 output -> (h, w, C) float16, contiguous: an image of C channels (16FC<C>)"""
    return dino[0].permute(1, 2, 0).to(torch.float16).contiguous()
