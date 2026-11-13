"""The node's conversions, without a GPU: Boxer's result as vision_msgs."""

import numpy as np
import xblick as db
from xblick.poses import rotation_from_quaternion

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
