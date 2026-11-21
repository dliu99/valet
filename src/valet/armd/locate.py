"""Where a point of the upright wrist image lies on a horizontal plane.

The camera is the sim's fit to the real wrist camera (mount, field of view, barrel distortion), posed by
MuJoCo at the arm's joints. The fit was made on 16:9 frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .frames import wrist_upright_turn_deg
from .kinematics import MOTORS, Kinematics

SIM_BARREL_K1 = -0.205


@dataclass(frozen=True)
class Camera:
    fovy_deg: float
    quat: tuple[float, float, float, float]
    barrel_k1: float


# The real wrist camera, fitted so table features seen from seven poses land within 0.6 cm of each other
# (runs/tools/calib.py). The sim renders with its own camera, which the sim's Locator keeps.
REAL_CAMERA = Camera(46.6, (0.978327, -0.205435, -0.003679, 0.025671), 0.5722)
MIN_RAY_DOWN = 0.05
MIN_TRIANGULATION_DEG = 5.0


class Locator:
    def __init__(self, kin: Kinematics, table_z: float, camera: Camera | None = None):
        import mujoco

        from .simscene import WRIST_CAMERA, SceneSpec, build

        self.mujoco = mujoco
        self.kin = kin
        self.table_z = table_z
        self.model = build(SceneSpec(table_z=table_z))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in MOTORS]
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.barrel_k1 = SIM_BARREL_K1
        if camera is not None:
            self.model.cam_fovy[self.cam] = camera.fovy_deg
            self.model.cam_quat[self.cam] = camera.quat
            self.barrel_k1 = camera.barrel_k1

    def ray(self, q_deg: np.ndarray, u: float, v: float, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        w, h = 2.0 * aspect, 2.0
        x, y = (u - 0.5) * w, (v - 0.5) * h
        roll = float(q_deg[MOTORS.index("wrist_roll")])
        a = math.radians(wrist_upright_turn_deg(roll) + 180.0)
        x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        nx, ny = x / aspect, y / aspect
        k = 1.0 + self.barrel_k1 * (nx * nx + ny * ny)
        x, y = x * k, y * k
        self.data.qpos[self.qadr] = np.radians(q_deg)
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_camlight(self.model, self.data)
        f = 1.0 / math.tan(math.radians(self.model.cam_fovy[self.cam]) / 2.0)
        rot = self.data.cam_xmat[self.cam].reshape(3, 3)
        d = rot @ np.array([x / f, -y / f, -1.0])
        return self.data.cam_xpos[self.cam].copy(), d / np.linalg.norm(d)

    def place(self, p: np.ndarray) -> dict:
        rel = p[:2] - self.kin.pan_axis_xy
        f0, f90 = self.kin.forward(0.0), self.kin.forward(90.0)
        return {"pan_deg": round(math.degrees(math.atan2(float(rel @ f90), float(rel @ f0))), 2), "reach_m": round(float(np.linalg.norm(rel)), 4),
                "height_m": round(float(p[2]) - self.table_z, 4)}

    def locate(self, q_deg: np.ndarray, u: float, v: float, aspect: float, height_m: float) -> dict:
        origin, d = self.ray(q_deg, u, v, aspect)
        ray = {"origin_m": [round(float(c), 5) for c in origin], "dir": [round(float(c), 5) for c in d]}
        plane = self.table_z + height_m
        if d[2] > -MIN_RAY_DOWN or origin[2] <= plane:
            return {"hit": False, "why": "that point of the image does not look down onto the plane", "ray": ray}
        p = origin + d * (plane - origin[2]) / d[2]
        return {"hit": True, "xyz_m": [round(float(c), 4) for c in p], **self.place(p), "height_m": round(height_m, 4), "ray": ray}

    def triangulate(self, a: dict, b: dict) -> dict:
        """The point nearest two rays (each {origin_m, dir}) and how far apart they pass."""
        o1, d1, o2, d2 = (np.array(x, float) for x in (a["origin_m"], a["dir"], b["origin_m"], b["dir"]))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(d1 @ d2)))))
        w = o1 - o2
        aa, bb, cc, dd, ee = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
        den = aa * cc - bb * bb
        if den < 1e-9 or angle < MIN_TRIANGULATION_DEG:
            return {"ok": False, "why": f"the two views are {angle:.1f}° apart; move the camera at least 5 cm sideways or up between them", "angle_deg": round(angle, 1)}
        s, t = (bb * ee - cc * dd) / den, (aa * ee - bb * dd) / den
        p1, p2 = o1 + s * d1, o2 + t * d2
        p = (p1 + p2) / 2
        return {"ok": True, "xyz_m": [round(float(c), 4) for c in p], **self.place(p), "gap_m": round(float(np.linalg.norm(p1 - p2)), 4), "angle_deg": round(angle, 1)}
