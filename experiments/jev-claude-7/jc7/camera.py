"""The wrist camera as a measuring instrument: joint readings in, rays and table points out.

Everything here comes from the arm's own model (forward kinematics, the fitted camera mount, its lens) and the
joint readings in `state()`. Nothing reads the box's true pose. On the real arm the same numbers come from the
calibrated URDF and the camera fit that valet-sim was built from."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valet_sim.scene import MOTORS, WRIST_CAMERA, mujoco
from valet_sim.sim import Optics

IMAGE_W, IMAGE_H = 640, 360


@dataclass(frozen=True)
class View:
    """Where the camera was when a frame was taken."""

    pos: np.ndarray  # camera centre, metres, world frame
    rot: np.ndarray  # 3x3, columns are the camera's x (right), y (up), z (backwards) in the world
    fovy_deg: float
    k1: float = Optics().barrel_k1
    w: int = IMAGE_W
    h: int = IMAGE_H

    @property
    def focal_px(self) -> float:
        return (self.h / 2) / np.tan(np.radians(self.fovy_deg) / 2)

    def ray(self, u: float, v: float) -> np.ndarray:
        """World direction through pixel (u, v) of the developed (barrel-distorted) frame, in pixels."""
        w, h = self.w, self.h
        nx, ny = (u - w / 2) / (w / 2), (v - h / 2) / (w / 2)
        s = 1.0 + self.k1 * (nx * nx + ny * ny)
        pu, pv = nx * s * (w / 2) + w / 2, ny * s * (w / 2) + h / 2
        f = self.focal_px
        d_cam = np.array([(pu - w / 2) / f, -(pv - h / 2) / f, -1.0])
        d = self.rot @ d_cam
        return d / np.linalg.norm(d)

    def ray_frac(self, fx: float, fy: float) -> np.ndarray:
        return self.ray(fx * self.w, fy * self.h)

    def on_plane(self, fx: float, fy: float, z: float = 0.0) -> np.ndarray | None:
        """Where the ray through a point of the frame (fractions of width and height) meets the plane at height z."""
        d = self.ray_frac(fx, fy)
        if abs(d[2]) < 1e-6:
            return None
        t = (z - self.pos[2]) / d[2]
        return None if t <= 0 else self.pos + t * d

    def project(self, p) -> tuple[float, float] | None:
        """Fraction coordinates of a world point in the developed frame (inverse of `ray`), or None if behind."""
        c = self.rot.T @ (np.asarray(p, float) - self.pos)
        if c[2] >= 0:
            return None
        f = self.focal_px
        pu, pv = -c[0] / c[2] * f + self.w / 2, c[1] / c[2] * f + self.h / 2
        # invert the barrel map by fixed-point iteration
        tx, ty = (pu - self.w / 2) / (self.w / 2), (pv - self.h / 2) / (self.w / 2)
        nx, ny = tx, ty
        for _ in range(20):
            s = 1.0 + self.k1 * (nx * nx + ny * ny)
            nx, ny = tx / s, ty / s
        return (nx * (self.w / 2) + self.w / 2) / self.w, (ny * (self.w / 2) + self.h / 2) / self.h


class Arm:
    """Forward kinematics of the arm's own model, separate from the simulation's state."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    def view(self, joints: dict[str, float]) -> View:
        self.data.qpos[self.qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        return View(self.data.cam_xpos[self.cam].copy(), self.data.cam_xmat[self.cam].reshape(3, 3).copy(),
                    float(self.model.cam_fovy[self.cam]))

    def jaw_axis(self, joints: dict[str, float]) -> np.ndarray:
        """The direction the jaws close along, in the world (the gripper body's x axis)."""
        self.data.qpos[self.qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xmat[self.gripper].reshape(3, 3)[:, 0].copy()
