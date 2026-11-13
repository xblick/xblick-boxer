"""The colour camera's K for a depth at another size (xblick-fast-foundationstereo's --max-width)."""

import pytest

from xblick_boxer.boxer import depth_k


def test_depth_k_at_the_same_and_at_half_size():
    k = (1000.0, 1001.0, 639.5, 479.5)
    assert depth_k(k, (1280, 960), (960, 1280)) == k
    fx, fy, cx, cy = depth_k(k, (1280, 960), (480, 640))
    assert (fx, fy) == pytest.approx((500.0, 500.5))
    assert (cx, cy) == pytest.approx((319.5, 239.5))  # the centre stays the centre
