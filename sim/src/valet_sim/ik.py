"""Joint goals that put the grasp point (between the jaws) at a position, with the jaws pitched below horizontal by
`pitch_deg` and the wrist rolled to `roll_deg`. Damped Newton on finite differences of MuJoCo's own kinematics."""

from __future__ import annotations

import numpy as np

from .scene import GRASP_SITE, MOTORS, mujoco

SOLVED_WITHIN_M = 0.003
SOLVED_WITHIN_DEG = 2.0
SEEDS = ((0.0, 30.0, 30.0), (-30.0, 60.0, 60.0), (20.0, 0.0, 70.0), (-10.0, 80.0, -20.0), (10.0, 40.0, 0.0))


class Unreachable(ValueError):
    pass


class IK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.range = np.degrees(model.jnt_range[jid])
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
        self.gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    def pose(self, joints: dict[str, float]) -> tuple[np.ndarray, float]:
        """The grasp point and the jaws' pitch below horizontal, for these joints."""
        self.data.qpos[self.qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.model, self.data)
        approach = -self.data.xmat[self.gripper].reshape(3, 3)[:, 2]
        return self.data.site_xpos[self.site].copy(), float(np.degrees(np.arcsin(np.clip(-approach[2], -1, 1))))

    def solve(self, xyz, pitch_deg: float, roll_deg: float = -90.0, seed: dict[str, float] | None = None,
              gripper: float | None = None) -> dict[str, float]:
        target = np.asarray(xyz, float)
        starts = [seed] if seed else []
        starts += [{"shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": flex} for lift, elbow, flex in SEEDS]
        for start in starts:
            q = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 30.0, "wrist_flex": 30.0, "gripper": 60.0, **start}
            q["wrist_roll"] = roll_deg
            if gripper is not None:
                q["gripper"] = gripper
            q["shoulder_pan"] = float(np.degrees(np.arctan2(target[1], target[0])))
            if self._newton(q, target, pitch_deg):
                return q
        raise Unreachable(f"no joints put the grasp point at {np.round(target, 3).tolist()} with the jaws {pitch_deg:.0f} deg below horizontal")

    def _newton(self, q: dict[str, float], target: np.ndarray, pitch_deg: float) -> bool:
        free = MOTORS[:4]
        for _ in range(60):
            r = self._residual(q, target, pitch_deg)
            if np.linalg.norm(r[:3]) < SOLVED_WITHIN_M and abs(r[3]) * 100 < SOLVED_WITHIN_DEG:
                return True
            jac = np.zeros((4, 4))
            for k, j in enumerate(free):
                jac[:, k] = (self._residual(dict(q, **{j: q[j] + 0.1}), target, pitch_deg) - r) / 0.1
            step = np.clip(-np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(4), jac.T @ r), -10, 10)
            for k, j in enumerate(free):
                lo, hi = self.range[MOTORS.index(j)]
                q[j] = float(np.clip(q[j] + step[k], lo, hi))
        return False

    def _residual(self, q: dict[str, float], target: np.ndarray, pitch_deg: float) -> np.ndarray:
        p, pitch = self.pose(q)
        return np.array([*(p - target), (pitch - pitch_deg) / 100.0])
