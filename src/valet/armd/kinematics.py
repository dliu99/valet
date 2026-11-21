"""Joint values are LeRobot calibrated degrees, which `so101_new_calib.urdf` takes as radians
after conversion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).resolve().parents[3] / "SO101" / "so101_new_calib.urdf"
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = MOTORS[:5]
PAN, LIFT, ELBOW, WRIST_FLEX, ROLL, GRIPPER = range(6)
TIP_LINK = "gripper_frame_link"

IK_POSITION_TOL_MM = 3.0
IK_PITCH_TOL_DEG = 2.0


class Unreachable(ValueError):
    pass


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    x, yy, z = np.eye(3)
    return _rotation(z, y) @ _rotation(yy, p) @ _rotation(x, r)


def _homogeneous(rot: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, xyz
    return t


@dataclass(frozen=True)
class _Joint:
    name: str
    origin: np.ndarray
    axis: np.ndarray | None


def _chain_to(urdf: Path, tip: str) -> list[_Joint]:
    joints_by_child = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        rpy = [float(v) for v in o.get("rpy").split()]
        axis = None if j.get("type") == "fixed" else np.array([float(v) for v in j.find("axis").get("xyz").split()])
        joints_by_child[j.find("child").get("link")] = (j.find("parent").get("link"), _Joint(j.get("name"), _homogeneous(_rpy(*rpy), xyz), axis))
    chain, link = [], tip
    while link in joints_by_child:
        link, joint = joints_by_child[link]
        chain.append(joint)
    return chain[::-1]


@dataclass(frozen=True)
class TipPose:
    xyz: np.ndarray
    pan_deg: float
    reach_m: float
    pitch_deg: float
    roll_deg: float


class Kinematics:
    def __init__(self, urdf: Path = URDF_PATH):
        self.chain = _chain_to(urdf, TIP_LINK)
        movable = [j.name for j in self.chain if j.axis is not None]
        if tuple(movable) != ARM:
            raise ValueError(f"{urdf}: chain to {TIP_LINK} moves {movable}, expected {list(ARM)}")
        self.pan_axis_xy = self.chain[0].origin[:2, 3].copy()
        self._forward_at_zero_pan = self._radial(np.zeros(6))

    def frames(self, q: np.ndarray) -> list[np.ndarray]:
        by_name = dict(zip(MOTORS, np.radians(np.asarray(q, float))))
        t, out = np.eye(4), []
        for j in self.chain:
            t = t @ j.origin
            if j.axis is not None:
                t = t @ _homogeneous(_rotation(j.axis, by_name[j.name]), np.zeros(3))
            out.append(t)
        return out

    def tip(self, q: np.ndarray) -> np.ndarray:
        return self.frames(q)[-1]

    def forward(self, pan_deg: float) -> np.ndarray:
        return _rotation(np.array([0.0, 0.0, 1.0]), -np.radians(pan_deg))[:2, :2] @ self._forward_at_zero_pan

    def pose(self, q: np.ndarray) -> TipPose:
        t = self.tip(q)
        fwd = self.forward(q[PAN])
        reach = float((t[:2, 3] - self.pan_axis_xy) @ fwd)
        approach = t[:3, 2]
        pitch = float(np.degrees(np.arctan2(-approach[2], approach[:2] @ fwd)))
        return TipPose(t[:3, 3].copy(), float(q[PAN]), reach, pitch, float(q[ROLL]))

    def solve(self, pan_deg: float, reach_m: float, z_m: float, pitch_deg: float, roll_deg: float,
              seed: np.ndarray, limit: dict[str, float]) -> np.ndarray:
        lo = np.array([-limit[m] for m in ARM[1:4]])
        hi = -lo
        want = np.array([reach_m * 1000.0, z_m * 1000.0, pitch_deg])

        def q_of(x: np.ndarray) -> np.ndarray:
            return np.array([pan_deg, *x, roll_deg, 0.0])

        def residual(x: np.ndarray) -> np.ndarray:
            p = self.pose(q_of(x))
            return np.array([p.reach_m * 1000.0, p.xyz[2] * 1000.0, p.pitch_deg]) - want

        seeds = [np.asarray(seed, float)[1:4], np.array([0.0, 0.0, 60.0]), np.array([-40.0, 40.0, 40.0]), np.array([30.0, -30.0, 80.0])]
        best, best_cost = None, np.inf
        for start in seeds:
            x = np.clip(start.copy(), lo, hi)
            for _ in range(60):
                r = residual(x)
                jac = np.stack([(residual(x + np.eye(3)[i] * 0.25) - r) / 0.25 for i in range(3)], axis=1)
                step = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(3), -jac.T @ r)
                x = np.clip(x + np.clip(step, -10.0, 10.0), lo, hi)
                if np.linalg.norm(step) < 1e-4:
                    break
            r = residual(x)
            if np.hypot(r[0], r[1]) <= IK_POSITION_TOL_MM and abs(r[2]) <= IK_PITCH_TOL_DEG:
                cost = float(np.max(np.abs(x - seeds[0])))
                if cost < best_cost:
                    best, best_cost = x, cost
                if start is seeds[0]:
                    break
        if best is None:
            raise Unreachable(f"no joints put the grasp point at reach {reach_m:.3f} m, z {z_m:.3f} m, pitch {pitch_deg:.0f}° within the joint limits")
        return q_of(best)[:5]

    def pitches(self, pan_deg: float, reach_m: float, z_m: float, roll_deg: float, seed: np.ndarray,
                limit: dict[str, float]) -> list[float]:
        found = []
        for pitch in range(-10, 100, 10):
            try:
                self.solve(pan_deg, reach_m, z_m, float(pitch), roll_deg, seed, limit)
                found.append(float(pitch))
            except Unreachable:
                pass
        return found

    def _radial(self, q: np.ndarray) -> np.ndarray:
        d = self.tip(q)[:2, 3] - self.pan_axis_xy
        return d / np.linalg.norm(d)
