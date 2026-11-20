"""Skills: what moves the arm between checks.

A skill is scripted (a few Cartesian moves) or learned (a LeRobot policy, see policy.py).
The recipe names a skill per action and the config binds the name, so a scripted pick
can be swapped for a trained one without touching the runtime. Skills do not judge
success beyond what the motors report; the runtime checks with Jev afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .recipe import Item
from .robot import Robot, Vec3


@dataclass
class SkillContext:
    item: Item
    target: Vec3  # pick: the item's centre on the table; place: the top of the stack
    yaw_deg: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    ok: bool  # the motions ran; says nothing about whether the scene is right
    note: str = ""
    gripper: float | None = None


class Skill(Protocol):
    def run(self, robot: Robot, ctx: SkillContext) -> SkillResult: ...


@dataclass(frozen=True)
class Gripper:
    open: float = 45.0  # opening wide enough for a slice; measure on your arm
    closed: float = 0.0
    holding_min: float = 3.0  # a close that stops this far above `closed` is on something


def _above(p: Vec3, dz: float) -> Vec3:
    return (p[0], p[1], p[2] + dz)


@dataclass(frozen=True)
class ScriptedPick:
    """Straight down, close, straight up. Good enough for stiff items; slices of ham or
    lettuce usually need a learned skill."""

    gripper: Gripper = Gripper()
    hover_m: float = 0.08
    grasp_dz_m: float = 0.0  # added to the item's height at the grasp

    def run(self, robot: Robot, ctx: SkillContext) -> SkillResult:
        robot.set_gripper(self.gripper.open)
        for xyz in (_above(ctx.target, self.hover_m), _above(ctx.target, self.grasp_dz_m)):
            m = robot.move_to(xyz, yaw_deg=ctx.yaw_deg)
            if not m.ok:
                return SkillResult(False, f"move to {xyz} failed: {m.note}")
        g = robot.set_gripper(self.gripper.closed)
        robot.move_to(_above(ctx.target, self.hover_m), yaw_deg=ctx.yaw_deg)
        on_something = g >= self.gripper.closed + self.gripper.holding_min
        return SkillResult(True, "fingers stopped on something" if on_something else "fingers closed fully", g)


@dataclass(frozen=True)
class ScriptedPlace:
    """Down over the stack, open, up."""

    gripper: Gripper = Gripper()
    hover_m: float = 0.08
    release_m: float = 0.012  # height above the stack top at which to let go

    def run(self, robot: Robot, ctx: SkillContext) -> SkillResult:
        for xyz in (_above(ctx.target, self.hover_m), _above(ctx.target, self.release_m)):
            m = robot.move_to(xyz, yaw_deg=ctx.yaw_deg)
            if not m.ok:
                return SkillResult(False, f"move to {xyz} failed: {m.note}")
        g = robot.set_gripper(self.gripper.open)
        robot.move_to(_above(ctx.target, self.hover_m), yaw_deg=ctx.yaw_deg)
        return SkillResult(True, "released", g)
