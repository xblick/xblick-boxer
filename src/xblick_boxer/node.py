"""xblick-boxer: Boxer (boxer.py) on a camera's Durchblick streams, its results as ROS's vision_msgs.

    xblick-boxer --camera cam/azurekinect --depth cam/azurekinect/depth_to_color --labels box can bottle \
                     [--color <camera>/color] [--no-depth, in place of --depth] [--node boxer] \
                     [--thresh2d 0.2] [--thresh3d 0.5] [--max-offset-ms MS] [--pose-wait-ms 200] [--features]
                     [--local shm|cuda] [--no-download]
                     [--connect ...] [--listen ...] [--config FILE]   (default: the PC's Zenoh router)

Reads the newest colour frame with its camera_info and, taken with it, the depth frame of its time (the nearest within
--max-offset-ms, by default half the depth's frame period; 16UC1 mm or 32FC1 m, in the colour camera's geometry: the
simulation's depth, the Azure Kinect's depth_to_color) and the camera's pose at the frame's stamp from ``<camera>/tf``
(Boxer needs the gravity direction, the world's z): the pose of that stamp, else interpolated between the two around it
(xblick.poses, Durchblick 0.3.4: the robot's camera poses come with its state, at their own rate).
Publishes, stamped as the frame and in its camera frame:

* ``<camera>/<node>/detections_2d``  vision_msgs/Detection2DArray: OWLv2's 2D boxes (pixels of the colour image)
* ``<camera>/<node>/boxes_3d``       vision_msgs/Detection3DArray: Boxer's 3D boxes (metres, box axes as orientation)

and with --features, while somebody subscribes to it:

* ``<camera>/<node>/dino``  BoxerNet's DINOv3 features of the frame as an image stream: 16FC384 (float16), one pixel
  per 16 x 16 patch of the image scaled to 960 x 960 (60 x 60), its camera_info that of the patch grid; on the GPU
  (``cuda`` transport, Durchblick's CUDA ring) for readers on the same host, else ``raw`` (2.8 MB a frame)

each detection with its prompt as class_id and its score; a frame without boxes publishes empty arrays (a reader's
view clears). A frame whose pose (or, with depth, whose depth) does not come is skipped and counted; the counts go to
``<camera>/diagnostics`` (status ``boxer``) once a second. It processes as fast as Boxer runs, always the newest
frame, in two stages: OWLv2 on the next frame while BoxerNet works on this one (about 20 ms per frame on an RTX PRO
6000: README, "Speed").

Weights (weights.py; in the node's cache directory ~/.cache/xblick/boxer, or --weights-dir): missing checkpoints are
downloaded at the start, before the models load; --no-download ends the node instead. ``xblick-boxer --download-only``
only downloads and checks them (no camera, no Zenoh, no GPU), for preparation scripts. Boxer's source (the submodule)
comes from tools/setup.sh.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import xblick as db
from xblick import mps, node, poses, tensors, topics
from xblick import weights as xblick_weights
from xblick.camera import check_local
from xblick.history import PartnerHistory
from xblick.poses import CameraPoses
from xblick.session import add_arguments, open_from_args

from . import weights


def to_messages(result, header: db.Header) -> tuple[db.Detection2DArray, db.Detection3DArray]:
    def hypothesis(label: str, score: float) -> list[db.ObjectHypothesisWithPose]:
        return [db.ObjectHypothesisWithPose(db.ObjectHypothesis(label, score))]

    boxes2d = db.Detection2DArray(header, [
        db.Detection2D(header, hypothesis(b.label, b.score),
                       db.BoundingBox2D(db.Pose2D(db.Point2D((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2), 0.0),
                                        b.x2 - b.x1, b.y2 - b.y1))
        for b in result.boxes2d])
    boxes3d = db.Detection3DArray(header, [
        db.Detection3D(header, hypothesis(b.label, b.score),
                       db.BoundingBox3D(db.Pose(db.Point(*(float(v) for v in b.center)),
                                                db.Quaternion(*poses.quaternion_from_rotation(b.rotation))),
                                        db.Vector3(*(float(v) for v in b.size))))
        for b in result.boxes3d])
    return boxes2d, boxes3d


# the parameters that change while the node runs (Parameters.declare_options); every other option applies at the start
LIVE = {"thresh2d": {"range": (0.0, 1.0, 0.01)},
        "thresh3d": {"range": (0.0, 1.0, 0.01)},
        "pose_wait_ms": {"range": (0.0, 2000.0, 10.0), "unit": "ms"}}


def split_labels(given: list[str]) -> list[str]:
    """Labels as the option takes them: each also comma-separated (``box,can`` or ``box "soup can"``)."""
    return [s.strip() for word in given for s in word.split(",") if s.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-boxer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's base, e.g. cam/azurekinect")
    parser.add_argument("--labels", required=True, nargs="+",
                        help="text prompts: words, each also comma-separated, e.g. 'box,can,bottle' or box can "
                             "\"tomato soup can\"")
    parser.add_argument("--color", default=None, help="the colour stream (default <camera>/color)")
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument("--depth", default=None, help="the depth stream in the colour camera's geometry "
                                                     "(default <camera>/depth)")
    depth.add_argument("--no-depth", action="store_true", help="without depth (Boxer then guesses distances)")
    parser.add_argument("--node", default="boxer", help="the results' part of the topic: <camera>/<node>/...")
    parser.add_argument("--thresh2d", type=float, default=0.2, help="OWLv2's score threshold")
    parser.add_argument("--thresh3d", type=float, default=0.5, help="BoxerNet's confidence threshold")
    parser.add_argument("--max-offset-ms", type=float, default=None,
                        help="largest difference of the colour and the depth frame's stamps that pairs them (default: "
                             "half the depth's frame period, measured: each colour frame with its nearest depth "
                             "frame; 0: equal stamps)")
    parser.add_argument("--pose-wait-ms", type=float, default=200.0,
                        help="how long to wait for the camera's pose at a frame's stamp on <camera>/tf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-beside-owl", action="store_true",
                        help="compute BoxerNet's DINOv3 features beside OWLv2 on a stream of their own (shorter time "
                             "to the boxes where the GPU has room; slower on a GPU shared with other jobs; README, "
                             "\"Speed\")")
    parser.add_argument("--features", action="store_true",
                        help="publish BoxerNet's DINOv3 features of every frame on <camera>/<node>/dino while somebody "
                             "subscribes (on a frame without 2D boxes DINOv3 then runs for them alone)")
    xblick_weights.add_arguments(parser)  # --download-only (a mode of the command line), --no-download
    add_arguments(parser)  # --connect, --listen, --config; default: the PC's Zenoh router
    # --max-hz: at most 30 colour frames/s by default (0 or none: every frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, max_hz=30.0, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = xblick_weights.parse_args(parser, argv)  # --download-only: --camera and --labels not required
    if args.download_only:
        return weights.store_for(parser, args).download_only(
            weights.needed(), options=weights.options(args))
    mps.check()  # with CUDA_MPS_PIPE_DIRECTORY (MPS meant) its daemon has to run: else stop here
    camera = topics.check_name(args.camera)
    # the options as parameters (the terminal interface shows them): read-only, but LIVE and the labels
    session = open_from_args(args)
    params = node.Parameters(session, args.node, camera)
    params.configure(args)
    params.declare_options(parser, args, leave_out=("camera", "node", "download_only"), live={
        **LIVE, "labels": {"write": lambda v: None if split_labels(v) else "at least one label"}})
    params.check_overrides()
    labels = split_labels(args.labels)
    color_base = args.color or f"{camera}/color"
    depth_base = None if args.no_depth else (args.depth or f"{camera}/depth")

    node.start("xblick-boxer", max_hz=args.max_hz, local=args.local)
    store = weights.store_for(parser, args)  # --weights-dir, also given as a parameter
    try:
        store.ensure(weights.needed(), download=not args.no_download,
                     options=weights.options(args))
    except xblick_weights.WeightsError as error:
        params.close()
        session.close()
        return store.fail(error)

    import torch  # noqa: PLC0415 - with the models, after the arguments are checked

    from .boxer import Boxer  # loads torch and the models: after the arguments are checked

    boxer = Boxer(labels, args.thresh2d, args.thresh3d, args.device, dino_beside_owl=args.dino_beside_owl,
                  weights=store.root)
    features_pub = None  # with the first frame's camera_info (--features)
    world = CameraPoses(session, camera)
    color = db.ImageSubscriber(session, color_base, transport="auto", camera_info=True, local=args.local,
                               max_hz=args.max_hz, device=args.device)
    depth_sub = (db.ImageSubscriber(session, depth_base, transport="auto", local=args.local, device=args.device)
                 if depth_base else None)
    boxes2d_pub = db.DetectionPublisher(session, topics.detections(camera, args.node, "detections_2d"),
                                        db.Detection2DArray)
    boxes3d_pub = db.DetectionPublisher(session, topics.detections(camera, args.node, "boxes_3d"),
                                        db.Detection3DArray)
    report = node.Reporter(session, camera, args.node, "xblick-boxer", max_hz=args.max_hz)
    report.watch(color)
    print(f"xblick-boxer: {color_base}" + (f" + {depth_base}" if depth_base else "") +
          f" -> {camera}/{args.node}/{{detections_2d,boxes_3d}}, prompts {labels}", flush=True)
    counts = report.counts
    stop = threading.Event()
    # the depth samples of the last second; a colour frame's depth is read when the colour frame is taken, before OWLv2
    # works on it (a shared-memory or CUDA sample is only valid until the camera reuses its ring slot)
    max_offset_ns = None if args.max_offset_ms is None else round(args.max_offset_ms * 1e6)
    depth = PartnerHistory(depth_sub, 30, max_offset_ns) if depth_sub is not None else None

    # the first stage in its own thread: OWLv2 on the newest colour frame while the main thread lifts the previous
    # one to 3D; at most one detected frame waits, so the published frame is never more than one frame behind
    detected: queue.Queue = queue.Queue(maxsize=1)
    failure: list[BaseException] = []

    def detect_frames():
        try:
            while not stop.is_set():
                frame = color.latest(timeout=0.1)
                if frame is None:
                    continue
                check_local(color, args.local)
                started = time.monotonic()
                # the live parameters of this stage, between frames, in this thread (OWLv2 is its own)
                boxer.thresh2d = params["thresh2d"]
                wanted = split_labels(params["labels"])
                if wanted != boxer.labels:
                    boxer.set_labels(wanted)
                    print(f"xblick-boxer: prompts {wanted} (parameter labels)", flush=True)
                depth_frame = None
                if depth is not None:  # with the colour frame, before the work on it (it may come a little after)
                    depth.check()
                    depth_frame = depth.wait(frame.header.stamp.to_ns(), params["pose_wait_ms"] / 1000.0)
                    if depth_frame is not None:
                        check_local(depth_sub, args.local)
                # without its depth (counted as no depth) not worth OWLv2's time
                detections = boxer.detect(frame.image) if depth is None or depth_frame is not None else None
                item = (frame, color.camera_info, depth_frame, detections, started)
                while not stop.is_set():
                    try:
                        detected.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except BaseException as error:  # noqa: BLE001 - raised in the main thread
            failure.append(error)

    threads = [threading.Thread(target=detect_frames, name="detect", daemon=True)]
    for thread in threads:
        thread.start()
    try:
        while True:
            try:
                item = detected.get(timeout=1.0)
            except queue.Empty:
                item = None
            if failure:
                raise RuntimeError("xblick-boxer: the detection stage failed") from failure[0]
            boxer.thresh3d = params["thresh3d"]  # the lifting's live parameter, in its thread
            if item is not None:
                frame, info, depth_frame, detections, started = item
                stamp = frame.header.stamp.to_ns()
                if detections is None:
                    counts["no_depth"] += 1
                elif info is None:
                    counts["no_info"] += 1
                elif (world_from_camera := world.wait(stamp, params["pose_wait_ms"] / 1000.0)) is None:
                    counts["no_pose"] += 1
                else:
                    depth_m = tensors.depth_metres(depth_frame, args.device) if depth_frame is not None else None
                    r_wc, t_wc = world_from_camera[:3, :3], world_from_camera[:3, 3]
                    k = (info.k[0], info.k[4], info.k[2], info.k[5])
                    frame_id = info.header.frame_id or frame.header.frame_id
                    if args.features and features_pub is None:
                        size = detections.size
                        grid = boxer.hw // 16
                        fx, fy, cx, cy = boxer.feature_camera(k, size)
                        device = torch.device(args.device)
                        features_pub = db.CameraPublisher(
                            session, f"{camera}/{args.node}/dino", encoding="16FC384", frame_id=frame_id,
                            camera_info=db.CameraInfo.pinhole(grid, grid, fx, fy, cx, cy, frame_id=frame_id),
                            source="boxer", cuda_device=device.index if device.index is not None else 0,
                            codec=None)  # half floats, 384 channels: no compressed form
                    # the features cost only while somebody reads them (DINOv3 on frames without boxes)
                    boxer.features = features_pub is not None and features_pub.subscribed
                    result = boxer.lift(detections, k, r_wc, t_wc, depth_m)
                    if result.features is not None:
                        # complete when lift returns (it waited for its stream): ordered after this thread's
                        # stream, publish need not wait for the GPU's other work (the next frame's detection)
                        features_pub.publish(result.features, stamp_ns=stamp, stream=torch.cuda.current_stream())
                    header = db.Header(frame.header.stamp, frame_id)
                    message2d, message3d = to_messages(result, header)
                    boxes2d_pub.publish(message2d)
                    boxes3d_pub.publish(message3d)
                    counts["frames"] += 1
                    counts["boxes3d"] += len(result.boxes3d)
                    counts["latency"] += time.monotonic() - started
            if report.due():
                d, seconds = report.take()
                ms = d["latency"] / max(d["frames"], 1) * 1e3
                skipped = {k: d[k] for k in ("no_pose", "no_depth", "no_info")}
                waiting = None if d["frames"] else (f"{color_base}: {color.describe()}" + (
                    f"; {depth_base}: {depth_sub.describe()}" if depth_sub is not None else ""))
                level, message = node.state(f"{d['frames'] / seconds:.2f} frames/s",
                                            [f"{v} {k.replace('_', ' ')}" for k, v in skipped.items() if v], waiting)
                report.publish(level, message, {"frames/s": d["frames"] / seconds,
                                                "boxes/frame": d["boxes3d"] / max(d["frames"], 1), "ms/frame": ms,
                                                **{k.replace("_", " "): v for k, v in skipped.items()},
                                                # depth frames lost to the camera's ring (all in all), logged too
                                                **({"depth overwritten": depth.lost} if depth is not None else {})},
                               log_extra=f", {ms:.0f} ms from frame to results")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        for closable in (world, color, depth_sub, boxes2d_pub, boxes3d_pub, features_pub, report):
            if closable is not None:
                closable.close()
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
