"""``--ros`` (the core's branch ros2): the node as its own process with a stand-in for Boxer (no checkpoints; a CUDA
device, as the node needs one), without depth, on a camera with a fixed pose. With ``--ros`` its 2D and 3D boxes also
appear on rmw_zenoh's keys, with ROS's 33-byte attachment and liveliness tokens, and a ROS subscriber reads them.
Without ``--ros`` nothing of ROS goes on the wire."""


import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch
import xblick as db
from xblick import ros, topics
from xblick.camera import StaticTransformPublisher
from xblick.msg import Header, Time, Transform, TransformStamped, Vector3
from xblick.session import RosOptions, open_session

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

CAM = "test/camera"
DOMAIN = 98
TOPICS = {f"/{topics.detections(CAM, 'boxer', 'detections_2d')}": "vision_msgs/msg/Detection2DArray",
          f"/{topics.detections(CAM, 'boxer', 'boxes_3d')}": "vision_msgs/msg/Detection3DArray",
          f"/{CAM}/diagnostics": "diagnostic_msgs/msg/DiagnosticArray"}
OURS = topics.detections(CAM, "boxer", "detections_2d")

# the node with a Boxer that finds one box per label, without its models and checkpoints
STAND_IN = """
import sys, types
import numpy as np
from xblick_boxer import node, weights

class StandIn:
    def __init__(self, labels, thresh2d, thresh3d, device, dino_beside_owl=False, weights=None):
        self.labels, self.thresh2d, self.thresh3d = list(labels), thresh2d, thresh3d
        self.features, self.hw = False, 960

    def set_labels(self, labels):
        self.labels = list(labels)

    def detect(self, image):
        return types.SimpleNamespace(size=tuple(image.shape[:2]))

    def lift(self, detections, k, r_wc, t_wc, depth_m):
        two = [types.SimpleNamespace(label=label, score=0.8, x1=10.0, y1=10.0 + 40 * i, x2=40.0, y2=40.0 + 40 * i)
               for i, label in enumerate(self.labels)]
        three = [types.SimpleNamespace(label=label, score=0.7, center=(0.1 * i, 0.0, 1.0), rotation=np.eye(3),
                                       size=(0.1, 0.1, 0.1)) for i, label in enumerate(self.labels)]
        return types.SimpleNamespace(boxes2d=two, boxes3d=three, features=None)

sys.modules["xblick_boxer.boxer"] = types.SimpleNamespace(Boxer=StandIn)
weights.needed = lambda: []
weights.STORE.required = ()  # no Boxer source needed either
sys.exit(node.main(sys.argv[1:]))
"""

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_every_default_key_is_a_ros_name():
    for key in (f"{CAM}/color/image_raw", f"{CAM}/depth/image_raw", OURS, topics.detections(CAM, "boxer", "boxes_3d"),
                f"{CAM}/boxer/dino/image_raw", f"{CAM}/diagnostics"):
        ros.topic_of(key)


def run_node(with_ros: bool, seconds: float = 90.0):
    """The stand-in node (and ``--ros``) on a published grey frame of a camera with a fixed pose. A reader that takes
    part in ROS subscribes to ``TOPICS`` (a mirror sends only to a ROS subscriber); an observer that takes part in
    nothing records every key with its attachment and every ROS liveliness token. Returns ({ROS topic: [payload]},
    {key: attachment}, {token}, the node's output)."""
    bgr = np.full((480, 640, 3), 128, np.uint8)
    h, w = bgr.shape[:2]
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    session = open_session(listen=[endpoint], ros_options=RosOptions(DOMAIN, "xblick_reader") if with_ros else None)
    observer = open_session(connect=[endpoint], declare=False)
    keys: dict[str, bytes | None] = {}
    tokens: set[str] = set()
    got: dict[str, list[bytes]] = {topic: [] for topic in TOPICS}
    watch = observer.declare_subscriber(
        "**", lambda s: keys.setdefault(str(s.key_expr), s.attachment.to_bytes() if s.attachment else None))
    live = observer.liveliness().declare_subscriber("@ros2_lv/**", lambda s: tokens.add(str(s.key_expr)),
                                                    history=True)
    readers = [session.ros.node.subscriber(topic, kind, lambda payload, *_, topic=topic: got[topic].append(payload))
               for topic, kind in TOPICS.items()] if with_ros else []
    info = db.CameraInfo.pinhole(w, h, 525.0, 525.0, w / 2 - 0.5, h / 2 - 0.5, frame_id="office")
    publisher = db.CameraPublisher(session, f"{CAM}/color", encoding="bgr8", frame_id="office", camera_info=info,
                                   source="test")
    pose = StaticTransformPublisher(session, CAM, [TransformStamped(Header(Time(1, 0), topics.WORLD), "office",
                                                                    Transform(Vector3(0.0, 0.0, 1.0)))])
    # --ros-node: the stand-in is python -c, whose program name is not xblick-boxer
    ros_options = ["--ros", "--ros-domain", str(DOMAIN), "--ros-node", "xblick_boxer"] if with_ros else []
    process = subprocess.Popen([sys.executable, "-c", STAND_IN, "--camera", CAM, "--labels", "box,can", "--no-depth",
                                "--no-download", "--local", "shm", "--connect", endpoint, *ros_options],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    try:
        stamp_ns, deadline, seen = 1_000_000_000, time.monotonic() + seconds, 0
        while time.monotonic() < deadline:
            assert process.poll() is None, "".join(output)
            publisher.publish(bgr, stamp_ns=stamp_ns)
            stamp_ns += 100_000_000
            time.sleep(0.1)
            seen += OURS in keys
            if all(got.values()) if with_ros else seen >= 15:
                break
        time.sleep(0.3)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
        for reader in readers:
            reader.close()
        pose.close()
        publisher.close()
        watch.undeclare()
        live.undeclare()
        observer.close()
        session.close()
    assert process.returncode == 0, "".join(output)
    return got, keys, tokens, "".join(output)


def test_with_ros_the_2d_and_3d_boxes_reach_ros():
    got, keys, tokens, output = run_node(with_ros=True)
    assert [topic for topic, payloads in got.items() if not payloads] == [], output
    message = db.Detection2DArray.deserialize(got[f"/{OURS}"][0])
    assert [d.results[0].hypothesis.class_id for d in message.detections] == ["box", "can"]
    boxes = db.Detection3DArray.deserialize(got[f"/{topics.detections(CAM, 'boxer', 'boxes_3d')}"][0])
    assert len(boxes.detections) == 2 and boxes.header.frame_id == "office"
    for topic, type_name in TOPICS.items():
        key = ros.data_key(DOMAIN, topic, type_name, ros.hash_of(type_name))
        assert len(keys[key]) == 33, key
        seq, stamp, _ = ros.decode_attachment(keys[key])
        assert seq >= 1 and abs(stamp - time.time_ns()) < 60 * 10**9
    parsed = [ros.parse_token(t) for t in tokens]
    publishers = {t.topic for t in parsed if t.kind == ros.PUBLISHER and t.fully_qualified_node == "/xblick_boxer"}
    assert set(TOPICS) <= publishers, publishers
    assert OURS in keys  # our keys as before


def test_without_ros_nothing_of_ros_goes_on_the_wire():
    _, keys, tokens, _ = run_node(with_ros=False)
    assert OURS in keys
    assert not [k for k in keys if "::dds_::" in k or k[:1].isdigit()] and tokens == set()
