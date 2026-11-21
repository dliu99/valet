from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, replace

import numpy as np

from .frames import UPRIGHT_VERIFIED_AT_ROLL_DEG
from .guard import Clock
from .kinematics import ARM, GRIPPER, MOTORS, Kinematics

FAKE_LIMIT_DEG = {"shoulder_pan": 107.0, "shoulder_lift": 97.0, "elbow_flex": 93.8, "wrist_flex": 92.0, "wrist_roll": 157.0}
START_JOINTS = (0.0, -27.2, -1.0, 88.2, UPRIGHT_VERIFIED_AT_ROLL_DEG, 60.0)
SERVO_SPEED_DPS = 360.0
IDLE_AMPS = 0.1
AMPS_PER_DEG_BEHIND = 0.02
GRIPPER_AMPS_PER_PCT_BEHIND = 0.12
HELD_LOAD_AMPS = 0.15
JAW_GAP_AT_FULL_OPEN_M = 0.10
JAW_REACH_MARGIN_M = 0.015
RELEASE_MARGIN_PCT = 2.0
VOLTAGE_V = 12.0


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    z: float
    w: float
    d: float
    h: float

    def contains(self, p: np.ndarray, margin_xy: float) -> bool:
        return (abs(p[0] - self.x) <= self.w / 2 + margin_xy and abs(p[1] - self.y) <= self.d / 2 + margin_xy
                and abs(p[2] - self.z) <= self.h / 2)

    def corners(self) -> np.ndarray:
        half = np.array([self.w, self.d, self.h]) / 2
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return np.array([self.x, self.y, self.z]) + signs * half


DEFAULT_BOX = Box(x=0.279, y=0.0, z=0.025, w=0.04, d=0.04, h=0.05)


def jaw_closing_axis_in_gripper(kin: Kinematics) -> np.ndarray:
    rot = kin.tip(np.array([0.0, 0.0, 0.0, 60.0, UPRIGHT_VERIFIED_AT_ROLL_DEG, 0.0]))[:3, :3]
    left_at_zero_pan = np.array([-kin.forward(0.0)[1], kin.forward(0.0)[0], 0.0])
    axis = rot.T @ -left_at_zero_pan
    axis[2] = 0.0
    return axis / np.linalg.norm(axis)


class FakeWorld:
    def __init__(self, box: Box, table_z: float, closing_axis: np.ndarray):
        self.closing_axis = closing_axis
        self.table_z = table_z
        self.box = box
        self.held = False
        self.held_at_pct = 0.0
        self._offset_in_grip: np.ndarray | None = None

    def box_now(self, tip: np.ndarray) -> Box:
        if not self.held:
            return self.box
        centre = tip[:3, 3] + tip[:3, :3] @ self._offset_in_grip
        return replace(self.box, x=float(centre[0]), y=float(centre[1]), z=float(centre[2]))

    def jaw_stop_pct(self, tip: np.ndarray) -> float | None:
        if self.held:
            return self.held_at_pct
        if not self.box.contains(tip[:3, 3], JAW_REACH_MARGIN_M):
            return None
        across = np.abs(tip[:3, :3] @ self.closing_axis) @ np.array([self.box.w, self.box.d, self.box.h])
        return float(100.0 * across / JAW_GAP_AT_FULL_OPEN_M)

    def grasp(self, tip: np.ndarray, at_pct: float) -> None:
        self.held, self.held_at_pct = True, at_pct
        self._offset_in_grip = tip[:3, :3].T @ (np.array([self.box.x, self.box.y, self.box.z]) - tip[:3, 3])

    def release(self, tip: np.ndarray) -> None:
        here = self.box_now(tip)
        self.box = replace(here, z=self.table_z + here.h / 2)
        self.held, self._offset_in_grip = False, None


class FakeArm:
    def __init__(self, kin: Kinematics, clock: Clock, table_z: float = 0.0, box: Box = DEFAULT_BOX):
        self.kin = kin
        self.clock = clock
        self.table_z = table_z
        self.limit = dict(FAKE_LIMIT_DEG)
        self._closing_axis = jaw_closing_axis_in_gripper(kin)
        self._lock = threading.RLock()
        self.reset(box)

    def reset(self, box: Box) -> None:
        with self._lock:
            self.world = FakeWorld(box, self.table_z, self._closing_axis)
            self._q = np.array(START_JOINTS, float)
            self._goal = self._q.copy()
            self._torque = True
            self._t = self.clock.now()

    def q(self) -> np.ndarray:
        with self._lock:
            self._advance()
            return self._q.copy()

    def amps(self) -> np.ndarray:
        with self._lock:
            self._advance()
            if not self._torque:
                return np.zeros(len(MOTORS))
            behind = np.abs(self._goal - self._q)
            out = IDLE_AMPS + AMPS_PER_DEG_BEHIND * behind
            out[GRIPPER] = IDLE_AMPS + GRIPPER_AMPS_PER_PCT_BEHIND * max(0.0, self._q[GRIPPER] - self._goal[GRIPPER])
            if self.world.held:
                out[1:3] += HELD_LOAD_AMPS
            return out

    def write_goals(self, goals: dict[str, float]) -> None:
        with self._lock:
            self._advance()
            for m, v in goals.items():
                self._goal[MOTORS.index(m)] = v

    def hold(self) -> None:
        with self._lock:
            self.write_goals(dict(zip(MOTORS, self.q())))
            self._torque = True

    def relax(self) -> None:
        with self._lock:
            self._advance()
            self._torque = False

    def torque_enabled(self) -> bool:
        with self._lock:
            return self._torque

    def voltage(self) -> np.ndarray:
        return np.full(len(MOTORS), VOLTAGE_V)

    def scene(self) -> tuple[np.ndarray, Box]:
        with self._lock:
            self._advance()
            return self._q.copy(), self.world.box_now(self.kin.tip(self._q))

    def truth(self) -> dict:
        q, box = self.scene()
        with self._lock:
            held = self.world.held
        return {"box": asdict(box), "held": held, "tip": dict(zip("xyz", map(float, self.kin.tip(q)[:3, 3])))}

    def _advance(self) -> None:
        now = self.clock.now()
        dt, self._t = now - self._t, now
        if dt <= 0 or not self._torque:
            return
        step = SERVO_SPEED_DPS * dt
        q = self._q + np.clip(self._goal - self._q, -step, step)
        tip = self.kin.tip(q)
        stop = self.world.jaw_stop_pct(tip)
        closing = q[GRIPPER] < self._q[GRIPPER]
        if stop is not None and closing and self._q[GRIPPER] >= stop and q[GRIPPER] < stop:
            q[GRIPPER] = stop
            if not self.world.held:
                self.world.grasp(tip, stop)
        elif self.world.held and q[GRIPPER] > self.world.held_at_pct + RELEASE_MARGIN_PCT:
            self.world.release(tip)
        lim = np.array([self.limit[m] for m in ARM])
        q[: len(ARM)] = np.clip(q[: len(ARM)], -lim - 3.0, lim + 3.0)
        self._q = q
