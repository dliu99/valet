import numpy as np
import pytest

from valet.geometry import box_offset, box_to_world, camera_looking_down, pixel_to_world, world_to_pixel
from valet.judge import Box
from valet.robot import Camera, OutOfWorkspace, Workspace

CAM = Camera(640, 480, 500.0, 500.0, 320.0, 240.0)


def test_camera_looking_down_is_a_rotation():
    T = camera_looking_down((0.1, 0.2, 0.3), yaw_deg=30)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], [0, 0, -1])  # looks down


@pytest.mark.parametrize("yaw", [0, 45, -120])
def test_pixel_and_world_round_trip(yaw):
    T = camera_looking_down((0.2, 0.0, 0.35), yaw)
    for xyz in [(0.2, 0.0, 0.01), (0.25, 0.08, 0.0), (0.12, -0.1, 0.012)]:
        u, v = world_to_pixel(CAM, T, xyz)
        back = pixel_to_world(CAM, T, u, v, xyz[2])
        assert np.allclose(back, xyz, atol=1e-9)


def test_box_to_world_gives_centre_and_size():
    T = camera_looking_down((0.2, 0.0, 0.35))
    # an 8 cm square on the table, centred under the camera: 0.08 * 500 / 0.35 px wide
    half = 0.04 * 500 / 0.35
    box = Box((240 - half) / 480, (240 + half) / 480, (320 - half) / 640, (320 + half) / 640)
    (x, y, z), span = box_to_world(CAM, T, box, 0.0)
    assert (x, y, z) == pytest.approx((0.2, 0.0, 0.0), abs=1e-9)
    assert span == pytest.approx(0.08, abs=1e-6)
    assert box_offset(box) == pytest.approx(0.0, abs=1e-9)


def test_ray_parallel_to_plane_has_no_hit():
    T = np.eye(4)  # camera looking along +z: rays through the centre never reach z = -1 going forward
    assert pixel_to_world(CAM, T, 320, 240, -1.0) is None


def test_workspace_refuses_outside_targets():
    ws = Workspace((0.0, -0.1, 0.0), (0.3, 0.1, 0.3))
    ws.check((0.1, 0.0, 0.1))
    with pytest.raises(OutOfWorkspace, match="y=0.200"):
        ws.check((0.1, 0.2, 0.1))
