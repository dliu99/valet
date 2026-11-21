"""Deltas start from the goal armd last commanded, not the present position, so sag under load
is not compounded."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ArmdError
from .guard import ARM_JOINTS, Guard, NotEngaged, Reading
from .kinematics import ARM, GRIPPER, PAN, ROLL, Kinematics, Unreachable
from .poses import PoseBook

EASED_PEAK_OVER_MEAN_SPEED = 1.5
OPEN_PCT = 60.0
GRIP_STEP_PCT = 2.0
GRIP_STALL_AMPS = 0.9
GRIP_STALL_LAG_PCT = 8.0
SQUEEZE_PCT = 3.0
EMPTY_AT_OR_BELOW_PCT = 4.0
JAMMED_IF_CLOSED_LESS_THAN_PCT = 5.0
GRIP_MOVED_SINCE_PCT = 6.0
DROPPED_IF_CLOSED_FURTHER_PCT = 1.5
SHORT_OF_TARGET_M = 0.03
LAST_GRASP_POSE = "last_grasp"
DELTA_FIELDS = ("pan_deg", "reach_m", "height_m", "pitch_deg", "roll_deg")


@dataclass(frozen=True)
class Outcome:
    ok: bool
    refused: str | None = None
    stopped: str | None = None


@dataclass(frozen=True)
class GripStop:
    command: str
    started_pct: float
    stopped_pct: float


def _smoothstep(s: float) -> float:
    return s * s * (3 - 2 * s)


def _number(motion: dict, key: str) -> float:
    v = motion.get(key, 0.0)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ArmdError(400, "bad_request", f"{key} must be a number")
    return float(v)


class Mover:
    def __init__(self, guard: Guard, kin: Kinematics, poses: PoseBook, max_speed_dps: float, table_z: float = 0.0):
        self.guard = guard
        self.table_z = table_z
        self.kin = kin
        self.poses = poses
        self.max_speed_dps = max_speed_dps
        self._grip: GripStop | None = None

    def execute(self, motion: dict) -> Outcome:
        kind = motion.get("kind")
        try:
            if kind == "delta":
                return self.delta(motion)
            if kind == "to":
                return self.to(motion)
            if kind == "pose":
                return self.goto(motion.get("name"))
            if kind == "grip":
                return self.grip(motion.get("to"), motion.get("pct"))
            if kind == "remember":
                return self.remember(motion.get("name"))
        except NotEngaged as e:
            return Outcome(False, refused=str(e))
        raise ArmdError(400, "bad_request", f"unknown motion kind {kind!r}: delta, to, pose, grip or remember")

    def delta(self, motion: dict) -> Outcome:
        d = dict(zip(DELTA_FIELDS, (_number(motion, k) for k in DELTA_FIELDS)))
        speed = _number(motion, "speed_dps") or self.max_speed_dps
        start = self._engaged_goal()
        target = start.copy()
        target[PAN] += d["pan_deg"]
        target[ROLL] += d["roll_deg"]
        if d["reach_m"] or d["height_m"] or d["pitch_deg"]:
            now = self.kin.pose(start)
            target = self._clamped(target)
            reach, z = now.reach_m + d["reach_m"], now.xyz[2] + d["height_m"]
            try:
                target[:ARM_JOINTS] = self.kin.solve(target[PAN], reach, z, now.pitch_deg + d["pitch_deg"], target[ROLL], start, self.guard.limit)
            except Unreachable as e:
                raise self._unreachable(e, target[PAN], reach, z, target[ROLL], start) from e
        return self.move_joints(target, speed)

    def to(self, motion: dict) -> Outcome:
        start = self._engaged_goal()
        now = self.kin.pose(start)
        given = {k: _number(motion, k) for k in DELTA_FIELDS if k in motion}
        target = start.copy()
        target[PAN] = given.get("pan_deg", start[PAN])
        target[ROLL] = given.get("roll_deg", start[ROLL])
        target = self._clamped(target)
        z = self.table_z + given["height_m"] if "height_m" in given else now.xyz[2]
        reach = given.get("reach_m", now.reach_m)
        try:
            target[:ARM_JOINTS] = self.kin.solve(target[PAN], reach, z, given.get("pitch_deg", now.pitch_deg), target[ROLL], start, self.guard.limit)
        except Unreachable as e:
            raise self._unreachable(e, target[PAN], reach, z, target[ROLL], start) from e
        return self.move_joints(target, _number(motion, "speed_dps") or self.max_speed_dps)

    def _unreachable(self, e: Unreachable, pan: float, reach: float, z: float, roll: float, seed: np.ndarray) -> ArmdError:
        ok = self.kin.pitches(pan, reach, z, roll, seed, self.guard.limit)
        hint = f"; there, pitches {ok[0]:.0f} to {ok[-1]:.0f}° are reachable" if ok else "; no pitch reaches that point"
        return ArmdError(422, "unreachable", f"{e}{hint}")

    def goto(self, name: object) -> Outcome:
        joints = self.poses.get(name) if isinstance(name, str) else None
        if joints is None:
            raise ArmdError(404, "not_found", f"no remembered pose named {name!r}")
        target = self._engaged_goal().copy()
        target[:ARM_JOINTS] = joints
        return self.move_joints(target, self.max_speed_dps)

    def remember(self, name: object) -> Outcome:
        source = self.guard.goal if self.guard.goal is not None else self.guard.read().q
        try:
            self.poses.remember(name, source[:ARM_JOINTS])
        except ValueError as e:
            raise ArmdError(400, "bad_request", str(e)) from e
        return Outcome(True)

    def move_joints(self, target: np.ndarray, speed_dps: float) -> Outcome:
        start = self._engaged_goal()
        speed = min(abs(speed_dps), self.max_speed_dps) or self.max_speed_dps
        wanted = np.asarray(target, float)
        target = self._clamped(wanted)
        biggest = float(np.max(np.abs(target - start)[:ARM_JOINTS]))
        n = max(1, math.ceil(EASED_PEAK_OVER_MEAN_SPEED * biggest / speed / self.guard.tick_s))
        followed = self.guard.follow(start + (target - start) * _smoothstep(k / n) for k in range(1, n + 1))
        clamped = "clamped at the joint limits" if np.max(np.abs(target - wanted)) > 0.01 else None
        short = float(np.linalg.norm(self.kin.pose(followed.final.q).xyz - self.kin.pose(target).xyz))
        blocked = f"ended {short * 100:.1f} cm from the target (blocked)" if short > SHORT_OF_TARGET_M else None
        stopped = followed.stopped or clamped or blocked
        return Outcome(stopped is None, stopped=stopped)

    def grip(self, to: object, pct: object = None) -> Outcome:
        if to not in ("open", "close"):
            raise ArmdError(400, "bad_request", "grip `to` is open or close")
        start = self._engaged_goal()
        from_pct = float(self.guard.read().q[GRIPPER])
        if to == "open":
            wide = OPEN_PCT if pct is None else min(100.0, max(0.0, float(pct)))
            followed = self.guard.follow(self._gripper_path(start, from_pct, wide))
            self._grip = GripStop("open", from_pct, float(followed.final.q[GRIPPER]))
            return Outcome(followed.stopped is None, stopped=followed.stopped)

        def stalled(r: Reading, goal: np.ndarray) -> bool:
            return abs(r.amps[GRIPPER]) > GRIP_STALL_AMPS or r.q[GRIPPER] - goal[GRIPPER] > GRIP_STALL_LAG_PCT

        followed = self.guard.follow(self._gripper_path(start, from_pct, 0.0), stall=stalled)
        if followed.stopped:
            return Outcome(False, stopped=followed.stopped)
        stopped_pct = float((followed.stalled or followed.final).q[GRIPPER])
        if followed.stalled:
            squeeze = self.guard.goal.copy()
            squeeze[GRIPPER] = max(0.0, stopped_pct - SQUEEZE_PCT)
            settled = self.guard.follow([squeeze])
            if settled.stopped:
                return Outcome(False, stopped=settled.stopped)
            stopped_pct = float(settled.final.q[GRIPPER])
        self._grip = GripStop("close", from_pct, stopped_pct)
        if self.grip_fact(stopped_pct) == "holding":
            self.poses.remember(LAST_GRASP_POSE, self.guard.goal[:ARM_JOINTS])
        return Outcome(True)

    def grip_fact(self, gripper_pct: float) -> str:
        g = self._grip
        if g is None or abs(gripper_pct - g.stopped_pct) > GRIP_MOVED_SINCE_PCT:
            return "unknown"
        if g.command == "open":
            return "open"
        if g.stopped_pct <= EMPTY_AT_OR_BELOW_PCT:
            return "empty"
        if g.started_pct - g.stopped_pct < JAMMED_IF_CLOSED_LESS_THAN_PCT:
            return "jammed"
        if gripper_pct < g.stopped_pct - DROPPED_IF_CLOSED_FURTHER_PCT:
            return "dropped"
        return "holding"

    def forget_grip(self) -> None:
        self._grip = None

    def _engaged_goal(self) -> np.ndarray:
        if self.guard.goal is None:
            raise NotEngaged("the arm is not engaged (torque off, or armd just started): POST /hold first")
        return self.guard.goal.copy()

    def _clamped(self, q: np.ndarray) -> np.ndarray:
        lim = np.array([self.guard.limit[m] for m in ARM])
        out = np.asarray(q, float).copy()
        out[:ARM_JOINTS] = np.clip(out[:ARM_JOINTS], -lim, lim)
        out[GRIPPER] = float(np.clip(out[GRIPPER], 0.0, 100.0))
        return out

    def _gripper_path(self, start: np.ndarray, from_pct: float, to_pct: float) -> list[np.ndarray]:
        steps = max(1, math.ceil(abs(to_pct - from_pct) / GRIP_STEP_PCT))
        path = []
        for k in range(1, steps + 1):
            goal = start.copy()
            goal[GRIPPER] = from_pct + (to_pct - from_pct) * k / steps
            path.append(goal)
        return path
