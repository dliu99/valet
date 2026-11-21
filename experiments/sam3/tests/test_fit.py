"""The pixel-to-metre step without SAM 3: the camera model round-trips, and the fits recover a box and a square
from silhouettes drawn with the same camera. No model, no network."""

import numpy as np
import pytest
from valet_sim.sim import START_JOINTS

from sam3p.camera import Camera
from sam3p.fit import Silhouette, cuboid_points, fit_box, fit_square, square_points

CAM = Camera()
JOINTS = dict(START_JOINTS)


def test_project_then_back_project_lands_on_the_point():
    p = np.array([0.26, 0.05, 0.03])
    px = CAM.project(JOINTS, p)
    back = CAM.on_plane(JOINTS, px, z=0.03)
    assert np.allclose(back, p, atol=1e-5)


def test_distort_inverts_undistort():
    px = np.array([[100.0, 50.0], [320.0, 180.0], [600.0, 340.0]])
    assert np.allclose(CAM.lens.distort(CAM.lens.undistort(px)), px, atol=1e-6)


def _mask(points: np.ndarray) -> np.ndarray:
    sil = Silhouette(CAM, JOINTS, np.ones((360, 640), bool), None, margin=0)  # the whole frame as the window
    return sil.raster(points) > 0.5


@pytest.mark.parametrize("x,y,yaw,side,h", [(0.26, 0.04, 12.0, 0.05, 0.075), (0.24, -0.06, -28.0, 0.045, 0.085)])
def test_fit_box_recovers_a_drawn_box(x, y, yaw, side, h):
    fit = fit_box(CAM, JOINTS, _mask(cuboid_points(x, y, yaw, side, h)))
    assert np.hypot(fit.x - x, fit.y - y) < 0.004
    assert abs((fit.yaw_deg - yaw + 45) % 90 - 45) < 3
    assert abs(fit.h - h) < 0.01


def test_fit_box_with_known_size_fits_position_only():
    fit = fit_box(CAM, JOINTS, _mask(cuboid_points(0.25, 0.0, 5.0, 0.05, 0.07)), hw=(0.05, 0.07))
    assert (fit.side, fit.h) == (0.05, 0.07)
    assert np.hypot(fit.x - 0.25, fit.y) < 0.003


def test_fit_square_recovers_a_drawn_square():
    fit = fit_square(CAM, JOINTS, _mask(square_points(0.22, 0.12, 20.0, 0.07)), 0.07)
    assert np.hypot(fit.x - 0.22, fit.y - 0.12) < 0.003
