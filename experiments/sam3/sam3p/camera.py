"""What the arm knows about its wrist camera: where it is (forward kinematics of the joint readings) and how it
images (pinhole at the fitted field of view, then the fitted barrel distortion of `valet_sim.sim.develop`).
A pixel becomes a ray and a ray meets a plane; a 3D point becomes a pixel. Nothing here reads the scene: the arm
model is built with its default box parked out of sight and never looked at.

The lens model matches point-and-reach's `geometry.py` (branch claude/point-and-reach-0d3ca8)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from valet_sim.scene import MOTORS, WRIST_CAMERA, BoxSpec, SceneSpec, build
from valet_sim.sim import Optics

W, H = 640, 360


def arm_model() -> mujoco.MjModel:
    """The arm's kinematics and camera mount, as the URDF and the camera fit give them on the real arm."""
    return build(SceneSpec(box=BoxSpec(x=-3.0, y=0.0), target_xy=None))


@dataclass(frozen=True)
class Lens:
    fovy_deg: float
    k1: float = Optics().barrel_k1
    w: int = W
    h: int = H

    @property
    def f(self) -> float:
        return (self.h / 2) / np.tan(np.radians(self.fovy_deg) / 2)

    def undistort(self, px: np.ndarray) -> np.ndarray:
        """Delivered-frame pixels (..., 2) -> pinhole pixels. `develop` samples the pinhole image at
        p (1 + k1 |p|^2), coordinates normalised by half the width, so this direction is closed-form."""
        px = np.asarray(px, float)
        c = np.array([self.w / 2, self.h / 2])
        n = (px - c) / (self.w / 2)
        r2 = (n * n).sum(-1, keepdims=True)
        return c + n * (1 + self.k1 * r2) * (self.w / 2)

    def distort(self, px: np.ndarray) -> np.ndarray:
        """Pinhole pixels (..., 2) -> delivered-frame pixels; NaN past the radius where the lens folds back."""
        px = np.asarray(px, float)
        c = np.array([self.w / 2, self.h / 2])
        t = (px - c) / (self.w / 2)
        r_raw = np.linalg.norm(t, axis=-1)
        r_max = 1 / np.sqrt(-3 * self.k1) if self.k1 < 0 else 10.0
        lo = np.zeros_like(r_raw)
        hi = np.full_like(r_raw, r_max)
        for _ in range(40):
            mid = (lo + hi) / 2
            below = mid * (1 + self.k1 * mid * mid) < r_raw
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        scale = np.where(r_raw > 1e-12, lo / np.maximum(r_raw, 1e-12), 1.0)
        out = c + t * scale[..., None] * (self.w / 2)
        bad = r_raw > r_max * (1 + self.k1 * r_max * r_max)
        out[bad] = np.nan
        return out


class Camera:
    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model or arm_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.lens = Lens(float(m.cam_fovy[self.cam]))

    def pose(self, joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and orientation (columns: image right, image up, backwards) in the world."""
        self.data.qpos[self.qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        return self.data.cam_xpos[self.cam].copy(), self.data.cam_xmat[self.cam].reshape(3, 3).copy()

    def project(self, joints: dict[str, float], xyz: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> delivered-frame pixels (..., 2); NaN behind the camera."""
        pos, rot = self.pose(joints)
        return self.project_from(pos, rot, xyz)

    def project_from(self, pos: np.ndarray, rot: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        pc = (np.asarray(xyz, float) - pos) @ rot
        depth = -pc[..., 2]
        f, lens = self.lens.f, self.lens
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.stack([lens.w / 2 + f * pc[..., 0] / depth, lens.h / 2 - f * pc[..., 1] / depth], -1)
        raw[depth <= 1e-6] = np.nan
        return lens.distort(raw)

    def rays(self, joints: dict[str, float], px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self.pose(joints)
        return self.rays_from(pos, rot, px)

    def rays_from(self, pos: np.ndarray, rot: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = self.lens.undistort(px)
        lens = self.lens
        d = np.stack([(u[..., 0] - lens.w / 2) / lens.f, -(u[..., 1] - lens.h / 2) / lens.f, -np.ones(u.shape[:-1])], -1) @ rot.T
        return pos, d / np.linalg.norm(d, axis=-1, keepdims=True)

    def on_plane(self, joints: dict[str, float], px: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Points at height `z` that these pixels (..., 2) see; NaN where the ray never comes down to it."""
        o, d = self.rays(joints, px)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - o[2]) / d[..., 2]
        out = o + d * t[..., None]
        out[(d[..., 2] > -1e-6) | (t < 0)] = np.nan
        return out

    def axis(self, joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self.pose(joints)
        return pos, -rot[:, 2]
