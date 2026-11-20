"""The agent's only way to act: a sequence of up to 8 end-effector targets, solved to joint goals with
`valet_sim.ik` and executed in order. The result only says what ran, never whether it worked: the agent judges
that from the next frames.

Two frames for the targets' positions:
- `base` (`execute_absolute_eef_sequence`, the paper's interface): absolute x, y, z in the robot base frame.
- `gripper` (`execute_relative_eef_sequence`): forward, left, up offsets from where the jaws are when the call
  starts, in the gripper's heading frame (forward is horizontal, away from the base through the jaws). At a
  steep pitch with roll -90, forward is up in the wrist image and left is left, so moves can be read off the image.
Pitch, roll and gripper are absolute in both.

Yaw is dropped. The SO-101 has five joints: the shoulder pan points the arm at the target's xy, so the jaws'
heading about vertical is not free. Turning the jaws to match a yawed box is done with roll at a steep pitch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS
from valet_sim.sim import Sim
from valet_sim.task import PAN_AXIS_X

MAX_TARGETS = 8
TICK_S = 0.1
MAX_STEPS = 50
SETTLE_S = 3.0
SETTLE_DEG = 1.0
FRAMES = ("base", "gripper")
POSITION = {"base": ("x", "y", "z"), "gripper": ("forward", "left", "up")}
REST = ("roll_deg", "pitch_deg", "gripper", "control_steps")

_ORIENTATION = (
    "pitch_deg is how far the jaws point below horizontal (90 = straight down). roll_deg is the wrist roll joint "
    "(-90 holds the wrist camera level). The arm has 5 joints, so the jaws always face along the line from the "
    "base to the target. gripper is 0 (closed) to 100 (open). control_steps is how many 0.1 s ticks the move "
    "takes (1-50); each target also waits for the arm to settle. The result only reports which targets ran."
)
DESCRIPTIONS = {
    "base": (
        "Move the arm through up to 8 absolute end-effector targets, in order. Positions are in metres in the "
        "robot base frame: x forward, y left, z up from the table. The point controlled is the centre between the "
        "jaw tips. " + _ORIENTATION
    ),
    "gripper": (
        "Move the arm through up to 8 end-effector targets, in order. Positions are offsets in metres from where "
        "the centre between the jaw tips is when this call starts, in the gripper's heading frame: forward is "
        "horizontal, away from the base through the jaws; left is horizontal, to the gripper's left; up is "
        "vertical. Every target in a call is relative to that same starting point. " + _ORIENTATION
    ),
}
NAMES = {"base": "execute_absolute_eef_sequence", "gripper": "execute_relative_eef_sequence"}


def tool(frame: str = "base") -> dict:
    fields = (*POSITION[frame], *REST)
    props = {f: {"type": "integer" if f == "control_steps" else "number"} for f in fields}
    return {
        "name": NAMES[frame],
        "description": DESCRIPTIONS[frame],
        "input_schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": f"1 to {MAX_TARGETS} targets",
                    "items": {"type": "object", "properties": props, "required": list(fields), "additionalProperties": False},
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


def heading(p: np.ndarray) -> float:
    """The gripper's heading about vertical: from the pan axis through the point between the jaws."""
    return float(np.arctan2(p[1], p[0] - PAN_AXIS_X))


def to_base(a: dict, start: np.ndarray) -> dict:
    """A gripper-frame target as a base-frame one, relative to `start` (the jaws' point when the call began)."""
    h = heading(start)
    fwd, left = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    xy = start[:2] + float(a["forward"]) * fwd + float(a["left"]) * left
    out = {k: v for k, v in a.items() if k not in POSITION["gripper"]}
    return {"x": float(xy[0]), "y": float(xy[1]), "z": float(start[2] + float(a["up"])), **out}


def to_gripper(a: dict, start: np.ndarray) -> dict:
    """The inverse of `to_base`, for showing a base-frame demonstration in the gripper frame."""
    h = heading(start)
    d = np.array([a["x"], a["y"]]) - start[:2]
    out = {k: v for k, v in a.items() if k not in POSITION["base"]}
    return {"forward": round(float(d @ [np.cos(h), np.sin(h)]), 3), "left": round(float(d @ [-np.sin(h), np.cos(h)]), 3),
            "up": round(float(a["z"] - start[2]), 3), **out}


class Arm:
    """The sim behind the tool: targets in, joint goals out."""

    def __init__(self, sim: Sim, frame: str = "base"):
        if frame not in FRAMES:
            raise ValueError(frame)
        self.sim = sim
        self.frame = frame
        self.ik = IK(sim.model)
        self.last = dict(sim.joints())

    def point(self) -> np.ndarray:
        return self.ik.pose(self.sim.joints())[0]

    def eef(self) -> dict:
        """The end effector in base-frame terms, read from the joint encoders."""
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
        start = self.point()
        for k, a in enumerate(actions):
            try:
                goal = self._solve(a, start)
            except (Unreachable, KeyError, TypeError, ValueError) as e:
                return Executed(f"executed {k} targets; target {k + 1} failed: {_reason(e)}", k, self.sim.data.time - t0)
            self._move(goal, int(np.clip(int(a["control_steps"]), 1, MAX_STEPS)))
        n = len(actions)
        return Executed(f"executed {n} target{'s' if n != 1 else ''}", n, self.sim.data.time - t0)

    def _solve(self, a: dict, start: np.ndarray) -> dict[str, float]:
        missing = [f for f in (*POSITION[self.frame], *REST) if f not in a]
        if missing:
            raise KeyError(f"missing {', '.join(missing)}")
        b = to_base(a, start) if self.frame == "gripper" else a
        xyz = [float(b["x"]), float(b["y"]), float(b["z"])]
        return self.ik.solve(xyz, float(b["pitch_deg"]), float(b["roll_deg"]), seed=self.last, gripper=float(b["gripper"]))

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
