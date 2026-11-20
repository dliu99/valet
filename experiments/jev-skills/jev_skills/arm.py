"""What the policy may touch: joint readings, servo currents, wrist frames, inverse kinematics and the camera model.

The camera model is the arm's own calibration (pose from forward kinematics of the joint reading, pinhole from the
camera's field of view, the lens's barrel distortion), as in the aspire and point-and-reach experiments. Nothing here
reads the box or the square from the simulator; `Arm.sim.truth()` is for grading only and is never called here."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS, WRIST_CAMERA, BoxSpec, mujoco
from valet_sim.sim import Sim, SimSpec, with_box

ARM_JOINTS = MOTORS[:5]
SETTLED_DEG = 0.5
STILL_DEG = 0.03
POLL_S = 0.05
PAN_AXIS_X = 0.0388


@dataclass(frozen=True)
class View:
    """The wrist camera at one joint reading. rot's columns: camera x right, y up, z backward (MuJoCo)."""

    pos: np.ndarray
    rot: np.ndarray
    f_px: float
    width: int
    height: int
    k1: float

    def undistort(self, u: float, v: float) -> tuple[float, float]:
        half = self.width / 2
        nx, ny = (u - half) / half, (v - self.height / 2) / half
        s = 1.0 + self.k1 * (nx * nx + ny * ny)
        return nx * s * half + half, ny * s * half + self.height / 2

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        uu, vv = self.undistort(u, v)
        d = self.rot @ np.array([(uu - self.width / 2) / self.f_px, -(vv - self.height / 2) / self.f_px, -1.0])
        return self.pos.copy(), d / np.linalg.norm(d)

    def to_plane(self, u: float, v: float, z: float) -> np.ndarray | None:
        o, d = self.ray(u, v)
        if abs(d[2]) < 1e-6:
            return None
        t = (z - o[2]) / d[2]
        return o + t * d if t > 0 else None

    def project(self, xyz) -> tuple[float, float] | None:
        """World point to delivered (distorted) pixel; None behind the camera or bent out of the frame."""
        p = self.rot.T @ (np.asarray(xyz, float) - self.pos)
        if p[2] >= -1e-6:
            return None
        u = self.width / 2 + self.f_px * p[0] / -p[2]
        v = self.height / 2 - self.f_px * p[1] / -p[2]
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
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < rt else (lo, mid)
        r = (lo + hi) / 2
        return tx * r / rt * half + half, ty * r / rt * half + self.height / 2


class Arm:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.ik = IK(sim.model)
        self._data = mujoco.MjData(sim.model)
        jid = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self._qadr = np.array([sim.model.jnt_qposadr[j] for j in jid])
        self._cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.width, self.height = sim.spec.image_size
        self.k1 = sim.spec.optics.barrel_k1

    # ---- reading ------------------------------------------------------------------------------------------------

    def t(self) -> float:
        return float(self.sim.data.time)

    def joints(self) -> dict[str, float]:
        return self.sim.joints()

    def gripper(self) -> float:
        return self.sim.joints()["gripper"]

    def gripper_amps(self) -> float:
        return float(self.sim.state()["amps"]["gripper"])

    def pose(self, joints: dict | None = None) -> tuple[np.ndarray, float]:
        """Grasp point (between the jaws) and the jaws' pitch below horizontal, from forward kinematics."""
        return self.ik.pose(joints or self.sim.joints())

    def frame(self) -> tuple[np.ndarray, dict[str, float]]:
        joints = self.sim.joints()
        return np.asarray(self.sim.frame("wrist")), joints

    def view(self, joints: dict[str, float]) -> View:
        d = self._data
        d.qpos[self._qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.sim.model, d)
        mujoco.mj_camlight(self.sim.model, d)
        f = (self.height / 2) / np.tan(np.radians(self.sim.model.cam_fovy[self._cam]) / 2)
        return View(d.cam_xpos[self._cam].copy(), d.cam_xmat[self._cam].reshape(3, 3).copy(), float(f),
                    self.width, self.height, self.k1)

    # ---- acting -------------------------------------------------------------------------------------------------

    def move(self, xyz, pitches, roll: float = -90.0, timeout_s: float = 4.0) -> dict:
        """Grasp point to xyz, trying each pitch until IK solves. status: reached, stopped (short of the goal:
        sag or contact), timeout, or unreachable (the arm did not move)."""
        xyz = [float(v) for v in xyz]
        now = self.sim.joints()
        goal_g = float(np.degrees(self.sim.goal[MOTORS.index("gripper")]))
        for pitch in pitches:
            try:
                q = self.ik.solve(xyz, float(pitch), float(roll), seed=now, gripper=goal_g)
            except Unreachable:
                continue
            goals = {j: q[j] for j in ARM_JOINTS}
            self.sim.set_goals(goals)
            status = self._wait(lambda j: all(abs(j[k] - goals[k]) < SETTLED_DEG for k in ARM_JOINTS), ARM_JOINTS, timeout_s)
            p, pt = self.pose()
            return {"status": status, "grasp_point": p, "pitch": pt, "error_m": float(np.linalg.norm(p - xyz))}
        p, pt = self.pose()
        return {"status": "unreachable", "grasp_point": p, "pitch": pt, "error_m": float(np.linalg.norm(p - xyz))}

    def grip(self, pct: float, timeout_s: float = 3.0) -> dict:
        """Drive the jaw to pct (0 closed, ~100 open). status: reached, stalled (on something) or timeout."""
        self.sim.set_goals({"gripper": float(pct)})
        status = self._wait(lambda j: abs(j["gripper"] - pct) < 1.0, ("gripper",), timeout_s, still="stalled")
        return {"status": status, "gripper": self.gripper(), "amps": self.gripper_amps()}

    def wait(self, seconds: float) -> None:
        self.sim.step(seconds)

    def _wait(self, done, joints, timeout_s: float, still: str = "stopped") -> str:
        sim = self.sim
        t_end = sim.data.time + timeout_s
        last = sim.joints()
        still_s = 0.0
        while sim.data.time < t_end:
            sim.step(POLL_S)
            now = sim.joints()
            if done(now):
                return "reached"
            moved = max(abs(now[k] - last[k]) for k in joints)
            still_s = still_s + POLL_S if moved < STILL_DEG else 0.0
            if still_s >= 0.25:
                return still
            last = now
        return "timeout"


@cache
def jaw_mask(width: int = 640, height: int = 360) -> np.ndarray:
    """Pixels where the arm's own fixed jaw shows in the wrist frame. The camera rides on the gripper, so this is a
    property of the arm, not of the scene: rendered once with nothing on the table but the table."""
    sim = Sim(with_box(SimSpec(), BoxSpec(x=-0.6, y=0.6), None))
    for name in ("table", "wall", "box"):
        sim.model.geom_rgba[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)][3] = 0.0
    arm = Arm(sim)
    arm.move([0.25, 0.0, 0.15], [90, 80, 70])
    a = np.asarray(sim.frame("wrist"), float)
    gray, sat = a.mean(2), a.max(2) - a.min(2)
    mask = (gray > 120) & (sat < 60)
    # grow by a few pixels so the jaw's soft edge does not leak into the box's top face
    grown = mask.copy()
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            grown |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return grown


def reach_angle_deg(x: float, y: float) -> float:
    return float(np.degrees(np.arctan2(y, x - PAN_AXIS_X)))
