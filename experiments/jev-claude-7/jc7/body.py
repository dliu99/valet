"""The arm as the policy may use it: joint goals and readings, the wrist frame, and its own kinematic model.

Motion is expressed as the grasp point G (between the jaws) at a position, with the jaws pitched below horizontal and
the wrist rolled. Code solves IK, slews the servos and waits for them to settle; it decides nothing about the task."""

from __future__ import annotations

import numpy as np
from PIL import Image

from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS
from valet_sim.sim import Sim

from .camera import Arm, View

SETTLE_DEG = 0.6


class Body:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.ik = IK(sim.model)
        self.arm = Arm(sim.model)
        self.moves = 0

    # --- readings -------------------------------------------------------------------------------------------
    def joints(self) -> dict[str, float]:
        return self.sim.joints()

    def state(self) -> dict:
        return self.sim.state()

    def frame(self) -> tuple[Image.Image, View]:
        j = self.sim.joints()
        return self.sim.frame("wrist"), self.arm.view(j)

    def pose(self) -> dict:
        """Where G is now, in cm, with the pitch, roll and gripper reading."""
        j = self.sim.joints()
        p, pitch = self.ik.pose(j)
        return {"G_cm": [round(v * 100, 1) for v in p], "pitch_deg": round(pitch, 0), "roll_deg": round(j["wrist_roll"], 0),
                "gripper": round(j["gripper"], 1)}

    def gripper_goal(self) -> float:
        """The gripper's goal, not its reading: a jaw stalled on a box reads wider than its goal, and moving with the
        reading as the goal would let go."""
        return float(np.degrees(self.sim.goal[MOTORS.index("gripper")]))

    def closing_dir(self, joints: dict[str, float] | None = None) -> np.ndarray:
        return self.arm.jaw_axis(joints or self.sim.joints())

    # --- motion ---------------------------------------------------------------------------------------------
    def solve(self, xyz, pitch: float, roll: float, gripper: float | None = None, pitch_slack: float = 0.0) -> tuple[dict, float]:
        """Joint goals for G at xyz. With pitch_slack, the nearest pitch within that many degrees that reaches."""
        seed = self.sim.joints()
        g = self.gripper_goal() if gripper is None else gripper
        tried = [pitch] + [pitch + s * d for d in range(5, int(pitch_slack) + 1, 5) for s in (-1, 1)]
        for p in tried:
            if not 0 <= p <= 95:
                continue
            try:
                return self.ik.solve(np.asarray(xyz, float), p, roll, seed=seed, gripper=g), p
            except Unreachable:
                continue
        raise Unreachable(f"G cannot reach {np.round(np.asarray(xyz) * 100, 1).tolist()} cm at pitch {pitch:.0f}+-{pitch_slack:.0f}, roll {roll:.0f}")

    def go(self, goals: dict[str, float], timeout_s: float = 5.0) -> None:
        self.sim.set_goals(goals)
        self.moves += 1
        t_end = self.sim.data.time + timeout_s
        while self.sim.data.time < t_end:
            self.sim.step(0.05)
            now = self.sim.joints()
            if all(abs(now[j] - goals[j]) < SETTLE_DEG for j in MOTORS if j != "gripper" and j in goals):
                break
        self.sim.step(0.1)

    def move_path(self, xyz, pitch: float, roll: float, gripper: float | None = None, pitch_slack: float = 0.0,
                  step_m: float = 0.015) -> float:
        """Move G to xyz in a straight line, in steps no longer than step_m. Returns the pitch used."""
        start, _ = self.ik.pose(self.sim.joints())
        end = np.asarray(xyz, float)
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / step_m)))
        q_end, p_used = self.solve(end, pitch, roll, gripper, pitch_slack)
        for k in range(1, n):
            pt = start + (end - start) * k / n
            try:
                q, _ = self.solve(pt, p_used, roll, gripper, 10)
            except Unreachable:
                continue
            self.go(q, timeout_s=1.5)
        self.go(q_end)
        return p_used

    def grip(self, value: float, timeout_s: float = 3.0) -> float:
        """Drive the gripper toward value until it reaches it or stalls on something. Returns the reading."""
        self.sim.set_goals({"gripper": value})
        i = MOTORS.index("gripper")
        act = self.sim.act[i]
        last = self.sim.joints()["gripper"]
        t_end = self.sim.data.time + timeout_s
        while self.sim.data.time < t_end:
            self.sim.step(0.05)
            now = self.sim.joints()["gripper"]
            command = float(np.degrees(self.sim.data.ctrl[act]))
            if abs(now - value) < 1.0:
                break
            # stalled: the slewing command has gone past the jaw, and the jaw has stopped
            passed = ((command - now) * (value - now) > 0 and abs(command - now) > 2.0) or abs(command - value) < 0.5
            if passed and abs(now - last) < 0.03:
                break
            last = now
        self.sim.step(0.2)
        return self.sim.joints()["gripper"]

    def look_at(self, target, cam_height: float = 0.20, pitches=(66, 60, 72, 55, 78, 50, 45, 40)) -> View | None:
        """Put the camera cam_height above a point, its centre ray through the point. Near pitch 66 the wrist camera
        looks straight down. Returns the view, or None if the arm cannot get there."""
        target = np.asarray(target, float)
        for pitch in pitches:
            q = self._camera_over(target, cam_height, pitch)
            if q is not None:
                self.go(q)
                return self.arm.view(self.sim.joints())
        return None

    def look_across(self, target, distance: float = 0.17, pitches=(25, 30, 20, 35, 15, 40)) -> View | None:
        """Look at a point from the base side, about 40 degrees below horizontal, its centre ray through the point:
        an oblique view that shows how tall something is."""
        target = np.asarray(target, float)
        for pitch in pitches:
            q = self._camera_at(target, distance, pitch)
            if q is not None:
                self.go(q)
                return self.arm.view(self.sim.joints())
        return None

    def _camera_at(self, target: np.ndarray, distance: float, pitch: float) -> dict | None:
        radial = target[:2] / max(np.linalg.norm(target[:2]), 1e-6)
        g = target + np.r_[-radial * distance * 0.5, distance * 0.4]
        seed = self.sim.joints()
        miss = np.ones(3)
        for _ in range(10):
            try:
                q = self.ik.solve(g, pitch, -90.0, seed=seed, gripper=self.gripper_goal())
            except Unreachable:
                return None
            v = self.arm.view(q)
            ax = -v.rot[:, 2]
            # the point on the centre ray closest to the target, and how far the camera is from it
            t = float(np.dot(target - v.pos, ax))
            closest = v.pos + t * ax
            miss = target - closest
            miss = miss + ax * (t - distance)
            if np.linalg.norm(miss) < 0.003:
                return q
            g = g + miss
        return q if np.linalg.norm(miss) < 0.01 else None

    def _camera_over(self, target: np.ndarray, cam_height: float, pitch: float) -> dict | None:
        g = target + [0.0, 0.0, max(0.06, cam_height - 0.06)]
        seed = self.sim.joints()
        for _ in range(8):
            try:
                q = self.ik.solve(g, pitch, -90.0, seed=seed, gripper=self.gripper_goal())
            except Unreachable:
                return None
            v = self.arm.view(q)
            hit = v.on_plane(0.5, 0.5, target[2])
            if hit is None:
                return None
            miss = np.r_[target[:2] - hit[:2], target[2] + cam_height - v.pos[2]]
            if np.linalg.norm(miss) < 0.003:
                return q
            g = g + miss
        return q if np.linalg.norm(miss[:2]) < 0.01 else None
