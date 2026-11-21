"""The only code that writes servo goals or switches torque."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np

from .kinematics import ARM, MOTORS

ARM_JOINTS = len(ARM)
LAG_STOP_DEG = 14.0
TICK_S = 1 / 30
SETTLE_S = 0.3


class Servos(Protocol):
    limit: dict[str, float]

    def q(self) -> np.ndarray: ...
    def amps(self) -> np.ndarray: ...
    def voltage(self) -> np.ndarray: ...
    def write_goals(self, goals: dict[str, float]) -> None: ...
    def hold(self) -> None: ...
    def relax(self) -> None: ...
    def torque_enabled(self) -> bool: ...


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


class NotEngaged(RuntimeError):
    pass


@dataclass(frozen=True)
class Reading:
    q: np.ndarray
    amps: np.ndarray


@dataclass(frozen=True)
class Followed:
    stopped: str | None
    stalled: Reading | None
    final: Reading


class Guard:
    def __init__(self, servos: Servos, clock: Clock | None = None, tick_s: float = TICK_S, settle_s: float = SETTLE_S):
        self.servos = servos
        self.clock = clock or RealClock()
        self.tick_s = tick_s
        self.settle_s = settle_s
        self.limit = dict(servos.limit)
        self.goal: np.ndarray | None = None
        self.peak = np.zeros(len(MOTORS))
        self._bus = threading.RLock()
        self._interrupted: str | None = None

    @property
    def peak_amps(self) -> float:
        return float(np.max(self.peak[:ARM_JOINTS]))

    def read(self) -> Reading:
        with self._bus:
            return Reading(self.servos.q(), self.servos.amps())

    def torque(self) -> bool:
        with self._bus:
            return self.servos.torque_enabled()

    def voltage(self) -> np.ndarray:
        with self._bus:
            return self.servos.voltage()

    def hold(self) -> None:
        with self._bus:
            self._interrupted = None
            if self.goal is not None and self.servos.torque_enabled():
                return
            here = self.servos.q()
            self.servos.hold()
            self.goal = here

    def relax(self, reason: str = "relaxed") -> None:
        self._interrupted = reason
        with self._bus:
            self.servos.relax()
            self.goal = None

    def interrupt(self, reason: str) -> None:
        self._interrupted = reason

    def follow(self, path: Iterable[np.ndarray], stall: Callable[[Reading, np.ndarray], bool] | None = None) -> Followed:
        if self.goal is None:
            raise NotEngaged("armd has not engaged the arm: POST /hold first")
        last = self.read()
        peak = np.abs(last.amps)
        stopped, stalled = None, None
        for goal in path:
            if self._interrupted:
                stopped = self._interrupted
                break
            t0 = self.clock.now()
            with self._bus:
                if self.goal is None:
                    stopped = self._interrupted or "relaxed"
                    break
                self.servos.write_goals(dict(zip(MOTORS, map(float, goal))))
                self.goal = np.asarray(goal, float).copy()
            self.clock.sleep(self.tick_s - (self.clock.now() - t0))
            last = self.read()
            peak = np.maximum(peak, np.abs(last.amps))
            stopped = self._hold_if_lagging(last)
            if stopped:
                break
            if stall and stall(last, goal):
                stalled = last
                break
        if stopped is None and stalled is None:
            self.clock.sleep(self.settle_s)
            last = self.read()
            peak = np.maximum(peak, np.abs(last.amps))
            stopped = self._hold_if_lagging(last)
        self.peak = peak
        return Followed(stopped, stalled, last)

    def _hold_if_lagging(self, r: Reading) -> str | None:
        with self._bus:
            if self.goal is None:
                return self._interrupted or "relaxed"
            lag = np.abs(r.q - self.goal)[:ARM_JOINTS]
            if np.max(lag) <= LAG_STOP_DEG:
                return None
            i = int(np.argmax(lag))
            hold_here = self.goal.copy()
            hold_here[:ARM_JOINTS] = r.q[:ARM_JOINTS]
            self.servos.write_goals(dict(zip(MOTORS, map(float, hold_here))))
            self.goal = hold_here
        return f"lag: {MOTORS[i]} lagged its goal by {lag[i]:.1f}°; holding where it stopped"
