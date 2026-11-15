"""The node's parts without a GPU: Boxer's result as vision_msgs, the camera's pose in the world."""

import socket

import numpy as np
import xblick as db
from xblick.camera import StaticTransformPublisher
from xblick.poses import BODY_FROM_OPTICAL, fixed_camera_transforms, pose_from, rotation_from_quaternion
from xblick.session import open_session

from xblick_boxer.boxer import Box2D, Box3D, Result


def random_rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_result_as_vision_msgs():
    from xblick_boxer.node import to_messages

    r = random_rotation(np.random.default_rng(1))
    result = Result([Box2D(10.0, 20.0, 50.0, 80.0, "mug", 0.4)],
                    [Box3D(np.array([0.1, -0.2, 1.5]), r, np.array([0.08, 0.12, 0.1]), "mug", 0.6)])
    header = db.Header(db.Time(5, 6), "AzureKinectCamera")
    boxes2d, boxes3d = to_messages(result, header)
    d2 = boxes2d.detections[0]
    assert (d2.bbox.center.position.x, d2.bbox.center.position.y, d2.bbox.size_x, d2.bbox.size_y) == (30, 50, 40, 60)
    assert d2.results[0].hypothesis.class_id == "mug" and d2.results[0].hypothesis.score == 0.4
    d3 = boxes3d.detections[0]
    p = d3.bbox.center.position
    assert (p.x, p.y, p.z) == (0.1, -0.2, 1.5) and (d3.bbox.size.x, d3.bbox.size.y, d3.bbox.size.z) == (0.08, 0.12, 0.1)
    o = d3.bbox.center.orientation
    np.testing.assert_allclose(rotation_from_quaternion(o.x, o.y, o.z, o.w), r, atol=1e-9)
    assert boxes3d.header == header and d3.header == header
    # the messages travel: their bytes read back as they were
    assert db.Detection3DArray.deserialize(boxes3d.serialize()) == boxes3d


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_fixed_cameras_pose_comes_from_tf_static_alone():
    """A fixed camera (xblick cam --pose): its pose only on tf_static, valid at any image's stamp."""
    from xblick_boxer.node import WorldPoses

    port = free_port()
    a = open_session(listen=[f"127.0.0.1:{port}"])
    b = open_session(connect=[f"127.0.0.1:{port}"])
    world_from_body = pose_from(0.5, -1.0, 1.5, 30, 10, 0)
    static = StaticTransformPublisher(a, "fixed/camera", fixed_camera_transforms("webcam", world_from_body, 0))
    world = WorldPoses(b, "fixed/camera")
    try:
        for stamp in (1_000, 5_000_000_000):
            pose = world.at("webcam_optical", stamp, 5.0)
            np.testing.assert_allclose(pose, world_from_body @ BODY_FROM_OPTICAL, atol=1e-9)
        assert world.why is None
    finally:
        world.close()
        static.close()
        b.close()
        a.close()


def test_a_missing_pose_is_none_with_its_reason():
    """No transform names the image's frame: None (the frame is skipped and counted), never an identity."""
    from xblick_boxer.node import WorldPoses

    session = open_session(listen=[f"127.0.0.1:{free_port()}"])
    world = WorldPoses(session, "lonely/camera")
    try:
        assert world.at("lonely_optical", 1_000, 0.2) is None
        assert "'lonely_optical'" in world.why and "\n" not in world.why
    finally:
        world.close()
        session.close()
