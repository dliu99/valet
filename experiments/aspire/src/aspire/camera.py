"""The known camera geometry: where each camera is for a given joint reading, and how a pixel maps to a ray.

The wrist camera rides on the gripper, so its pose comes from forward kinematics of the joint reading (as the
real arm would get it from the URDF); the side camera is fixed. Wrist frames carry the barrel distortion of
`valet_sim.sim.Optics`; the side camera's do not. Pixels are (u, v) = (column, row) in the delivered frame,
origin at the top left."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from valet_sim.scene import MOTORS, SIDE_CAMERA, WRIST_CAMERA, mujoco
from valet_sim.sim import Optics

CAMERA_NAMES = {"wrist": WRIST_CAMERA, "side": SIDE_CAMERA}


@dataclass(frozen=True)
class View:
    """One camera at one joint reading: position, rotation (columns: camera x right, y up, z backward)."""

    name: str
    pos: np.ndarray
    rot: np.ndarray
    f_px: float
    width: int
    height: int
    k1: float

    def undistort(self, u: float, v: float) -> tuple[float, float]:
        if not self.k1:
            return u, v
        half = self.width / 2
        nx, ny = (u - half) / half, (v - self.height / 2) / half
        s = 1.0 + self.k1 * (nx * nx + ny * ny)
        return nx * s * half + half, ny * s * half + self.height / 2

    def distort(self, u: float, v: float) -> tuple[float, float] | None:
        """Inverse of undistort; None for an undistorted point the lens bends out of every frame."""
        if not self.k1:
            return u, v
        half = self.width / 2
        tx, ty = (u - half) / half, (v - self.height / 2) / half
        rt = float(np.hypot(tx, ty))
        if rt < 1e-9:
            return u, v
        f = lambda r: r * (1.0 + self.k1 * r * r)
        hi = float(np.sqrt(1.0 / (3.0 * abs(self.k1)))) if self.k1 < 0 else 10.0
        if f(hi) < rt:
            return None
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < rt else (lo, mid)
        r = (lo + hi) / 2
        return tx * r / rt * half + half, ty * r / rt * half + self.height / 2

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction, in world coordinates, of the ray through pixel (u, v)."""
        uu, vv = self.undistort(u, v)
        d_cam = np.array([(uu - self.width / 2) / self.f_px, -(vv - self.height / 2) / self.f_px, -1.0])
        d = self.rot @ d_cam
        return self.pos.copy(), d / np.linalg.norm(d)

    def to_plane(self, u: float, v: float, z: float) -> np.ndarray | None:
        """Where the ray through (u, v) meets the horizontal plane at height z (metres), or None if it does not."""
        o, d = self.ray(u, v)
        if abs(d[2]) < 1e-6:
            return None
        t = (z - o[2]) / d[2]
        return o + t * d if t > 0 else None

    def project(self, xyz) -> tuple[float, float] | None:
        """The pixel a world point lands on, or None if it is behind the camera."""
        p = self.rot.T @ (np.asarray(xyz, float) - self.pos)
        if p[2] >= -1e-6:
            return None
        u = self.width / 2 + self.f_px * p[0] / -p[2]
        v = self.height / 2 - self.f_px * p[1] / -p[2]
        return self.distort(u, v)


class Cameras:
    """Camera poses from a joint reading, using the arm's kinematic model only (no scene state)."""

    def __init__(self, model: mujoco.MjModel, image_size: tuple[int, int] = (640, 360), optics: Optics = Optics()):
        self.model = model
        self.data = mujoco.MjData(model)
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.cam = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, v) for k, v in CAMERA_NAMES.items()}
        self.size = image_size
        self.k1 = optics.barrel_k1

    def view(self, name: str, joints: dict[str, float]) -> View:
        if name not in self.cam:
            raise ValueError(f"unknown camera {name!r}; cameras are {sorted(self.cam)}")
        self.data.qpos[self.qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        c = self.cam[name]
        w, h = self.size
        f = (h / 2) / np.tan(np.radians(self.model.cam_fovy[c]) / 2)
        return View(name, self.data.cam_xpos[c].copy(), self.data.cam_xmat[c].reshape(3, 3).copy(), float(f), w, h,
                    self.k1 if name == "wrist" else 0.0)
