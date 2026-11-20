"""`execute_absolute_eef_sequence`: the agent's only way to act. Each target is an absolute end-effector pose,
solved to joint goals with `valet_sim.ik` and executed in order. The result only says what ran, never whether it
worked: the agent judges that from the next frames.

Yaw is dropped. The SO-101 has five joints: the shoulder pan points the arm at the target's xy, so the jaws'
heading about vertical is not free. Turning the jaws to match a yawed box is done with roll at a steep pitch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS
from valet_sim.sim import Sim

MAX_TARGETS = 8
TICK_S = 0.1
MAX_STEPS = 50
SETTLE_S = 3.0
SETTLE_DEG = 1.0
FIELDS = ("x", "y", "z", "roll_deg", "pitch_deg", "gripper", "control_steps")

TOOL = {
    "name": "execute_absolute_eef_sequence",
    "description": (
        "Move the arm through up to 8 absolute end-effector targets, in order. Positions are in metres in the "
        "robot base frame: x forward, y left, z up from the table. The point controlled is the centre between the "
        "jaw tips. pitch_deg is how far the jaws point below horizontal (90 = straight down). roll_deg is the "
        "wrist roll joint (-90 holds the wrist camera level). The arm has 5 joints, so the jaws always face along "
        "the line from the base to the target's x, y. gripper is 0 (closed) to 100 (open). control_steps is how "
        "many 0.1 s ticks the move takes (1-50); each target also waits for the arm to settle. The result only "
        "reports which targets ran."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "description": f"1 to {MAX_TARGETS} targets",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "roll_deg": {"type": "number"},
                        "pitch_deg": {"type": "number"},
                        "gripper": {"type": "number"},
                        "control_steps": {"type": "integer"},
                    },
                    "required": list(FIELDS),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    },
}


@dataclass
class Executed:
    text: str
    ran: int
    sim_s: float


class Arm:
    """The sim behind the tool: absolute targets in, joint goals out."""

    def __init__(self, sim: Sim):
        self.sim = sim
        self.ik = IK(sim.model)
        self.last = dict(sim.joints())

    def eef(self) -> dict:
        """The end effector as the tool names it, read from the joint encoders."""
        j = self.sim.joints()
        p, pitch = self.ik.pose(j)
        return {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "z": round(float(p[2]), 3),
                "roll_deg": round(j["wrist_roll"], 1), "pitch_deg": round(pitch, 1), "gripper": round(j["gripper"], 1)}

    def execute(self, actions: list[dict]) -> Executed:
        t0 = self.sim.data.time
        if not actions:
            return Executed("executed 0 targets: actions is empty", 0, 0.0)
        if len(actions) > MAX_TARGETS:
            return Executed(f"executed 0 targets: at most {MAX_TARGETS} targets per call, got {len(actions)}", 0, 0.0)
        for k, a in enumerate(actions):
            try:
                goal = self._solve(a)
            except (Unreachable, KeyError, TypeError, ValueError) as e:
                return Executed(f"executed {k} targets; target {k + 1} failed: {_reason(e)}", k, self.sim.data.time - t0)
            self._move(goal, int(np.clip(int(a["control_steps"]), 1, MAX_STEPS)))
        n = len(actions)
        return Executed(f"executed {n} target{'s' if n != 1 else ''}", n, self.sim.data.time - t0)

    def _solve(self, a: dict) -> dict[str, float]:
        missing = [f for f in FIELDS if f not in a]
        if missing:
            raise KeyError(f"missing {', '.join(missing)}")
        xyz = [float(a["x"]), float(a["y"]), float(a["z"])]
        q = self.ik.solve(xyz, float(a["pitch_deg"]), float(a["roll_deg"]), seed=self.last, gripper=float(a["gripper"]))
        return q

    def _move(self, goal: dict[str, float], steps: int) -> None:
        sim = self.sim
        start = dict(self.last)
        for i in range(1, steps + 1):
            f = i / steps
            sim.set_goals({j: start[j] + f * (goal[j] - start[j]) for j in MOTORS})
            sim.step(TICK_S)
        self.last = dict(goal)
        prev_grip = sim.joints()["gripper"]
        t_end = sim.data.time + SETTLE_S
        while sim.data.time < t_end:
            sim.step(0.05)
            now = sim.joints()
            grip = now["gripper"]
            arm_settled = all(abs(now[j] - goal[j]) < SETTLE_DEG for j in MOTORS if j != "gripper")
            if arm_settled and abs(grip - prev_grip) < 0.05:
                break
            prev_grip = grip


def _reason(e: Exception) -> str:
    msg = e.args[0] if e.args else str(e)
    return msg if isinstance(e, Unreachable) else f"bad target ({msg})"
