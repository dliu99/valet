"""Moving the arm: joint goals from IK, waits for the servos, the gripper, and looks from above a table point.
`look_from_above` follows point-and-reach's `geometry.py` (branch claude/point-and-reach-0d3ca8)."""

from __future__ import annotations

import numpy as np
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS
from valet_sim.sim import Sim

from .camera import Camera

SETTLE_DEG = 0.5


def move(sim: Sim, goals: dict[str, float], timeout_s: float = 4.0) -> None:
    sim.set_goals(goals)
    t_end = sim.data.time + timeout_s
    while sim.data.time < t_end:
        sim.step(0.05)
        now = sim.joints()
        if all(abs(now[j] - goals[j]) < SETTLE_DEG for j in MOTORS if j != "gripper"):
            break


def set_gripper(sim: Sim, pct: float, settle_s: float = 0.8) -> None:
    sim.set_goals({"gripper": pct})
    sim.step(settle_s)


def close(sim: Sim) -> float:
    """Close until the jaws stall, then hold a few units inside the stall reading. Returns the reading."""
    sim.set_goals({"gripper": 0.0})
    last = sim.joints()["gripper"]
    for _ in range(60):
        sim.step(0.05)
        now = sim.joints()["gripper"]
        if abs(now - last) < 0.05:
            break
        last = now
    sim.set_goals({"gripper": max(0.0, now - 3.0)})
    sim.step(0.3)
    return sim.joints()["gripper"]


def solve(ik: IK, xyz, pitch: float, roll: float = -90.0, seed=None, gripper: float = 70.0,
          fallbacks=(0, -10, -20, -30, -40)) -> tuple[dict, float]:
    for dp in fallbacks:
        try:
            return ik.solve(xyz, pitch + dp, roll, seed=seed, gripper=gripper), pitch + dp
        except Unreachable:
            continue
    raise Unreachable(f"nothing reaches {np.round(np.asarray(xyz), 3).tolist()} near pitch {pitch:.0f}")


def look_from_above(ik: IK, cam: Camera, xy, height: float, gripper: float = 70.0,
                    pitches=(88.0, 75.0, 60.0, 45.0)) -> dict[str, float]:
    """Joints that put the wrist camera about `height` above the table with its optical axis through `xy`, as
    steep as the arm allows; lower looks are tried when the arm cannot reach that high."""
    target = np.array([xy[0], xy[1], 0.0])
    for hgt in (height, height - 0.04, height - 0.08):
        for pitch in pitches:
            q = _aim(ik, cam, target, hgt, pitch, gripper)
            if q is not None:
                return q
    raise Unreachable(f"no look from above at {np.round(target[:2], 3).tolist()}")


def _aim(ik: IK, cam: Camera, target: np.ndarray, height: float, pitch: float, gripper: float) -> dict | None:
    grasp = np.array([target[0], target[1], height - 0.05])
    seed = None
    for _ in range(10):
        try:
            q = ik.solve(grasp, pitch, -90.0, seed=seed, gripper=gripper)
        except Unreachable:
            return None
        pos, ax = cam.axis(q)
        if ax[2] > -0.3:
            return None
        hit = pos + ax * (0.0 - pos[2]) / ax[2]
        err = target - hit
        err[2] = height - pos[2]
        if np.linalg.norm(err) < 0.003:
            return q
        grasp, seed = grasp + err, q
    return None
