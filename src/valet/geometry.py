"""Wrist-camera geometry: pixels to table points and back, without a depth camera.

A pixel defines a ray from the camera; where that ray meets a horizontal plane of known
height (the table, or the top of the sandwich) is the point. The camera pose comes from
the arm's forward kinematics, so every frame can be placed in the base frame.
"""

from __future__ import annotations

import numpy as np

from .robot import Camera, Vec3


def pixel_ray(cam: Camera, T_world_cam: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """Origin and unit direction, in the base frame, of the ray through pixel (u, v)."""
    d_cam = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
    d = R @ d_cam
    return o, d / np.linalg.norm(d)


def ray_hits_plane(origin: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray | None:
    """Where the ray meets the plane at height z, or None if it never does going forward."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (z - origin[2]) / direction[2]
    return None if t <= 0 else origin + t * direction


def pixel_to_world(cam: Camera, T_world_cam: np.ndarray, u: float, v: float, z: float) -> Vec3 | None:
    p = ray_hits_plane(*pixel_ray(cam, T_world_cam, u, v), z)
    return None if p is None else (float(p[0]), float(p[1]), float(p[2]))


def world_to_pixel(cam: Camera, T_world_cam: np.ndarray, xyz: Vec3) -> tuple[float, float] | None:
    """Where a base-frame point appears in the picture, or None if it is behind the camera."""
    T_cam_world = np.linalg.inv(T_world_cam)
    p = T_cam_world @ np.array([*xyz, 1.0])
    if p[2] <= 1e-6:
        return None
    return (cam.fx * p[0] / p[2] + cam.cx, cam.fy * p[1] / p[2] + cam.cy)


def box_center_px(cam: Camera, box: "BoxLike") -> tuple[float, float]:
    return ((box.left + box.right) / 2 * cam.width, (box.top + box.bottom) / 2 * cam.height)


def box_to_world(cam: Camera, T_world_cam: np.ndarray, box: "BoxLike", z: float) -> tuple[Vec3, float] | None:
    """The box centre on the plane at height z, and the box's larger side in metres there."""
    u, v = box_center_px(cam, box)
    center = pixel_to_world(cam, T_world_cam, u, v, z)
    a = pixel_to_world(cam, T_world_cam, box.left * cam.width, box.top * cam.height, z)
    b = pixel_to_world(cam, T_world_cam, box.right * cam.width, box.bottom * cam.height, z)
    if center is None or a is None or b is None:
        return None
    span = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    return center, span


def box_offset(box: "BoxLike") -> float:
    """Distance of the box centre from the picture centre, as a fraction of the picture (0 = centred)."""
    return float(np.hypot((box.left + box.right) / 2 - 0.5, (box.top + box.bottom) / 2 - 0.5))


class BoxLike:  # structural: anything with top, bottom, left, right as fractions of the picture
    top: float
    bottom: float
    left: float
    right: float


def camera_looking_down(at: Vec3, yaw_deg: float = 0.0) -> np.ndarray:
    """Pose of a camera at `at` looking straight down; picture x along +x, picture y along -y at yaw 0."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    x_axis = np.array([c, s, 0.0])
    y_axis = np.array([s, -c, 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x_axis, y_axis, z_axis, at
    return T
