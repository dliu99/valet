"""The simulated arm as Claude's tools see it: frames, joint state, IK moves and the gripper, in simulated time.

Reads only `Sim.state()` and frames. `truth()` and `score()` belong to the grader in `run.py`."""

from __future__ import annotations

import io

from PIL import Image

from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS
from valet_sim.sim import Sim

from .overlay import Overlay

ARM_JOINTS = MOTORS[:5]
SETTLE_DEG = 0.5
MOVE_TIMEOUT_S = 4.0
STEP_S = 0.05
DEFAULT_OPEN = 70.0
SQUEEZE_DEG = 3.0

__all__ = ["Arm", "Unreachable"]


class Arm:
    def __init__(self, sim: Sim, max_side: int = 640, quality: int = 85, overlay: bool = False):
        self.sim = sim
        self.ik = IK(sim.model)
        self.overlay = Overlay(sim.model, sim.spec.optics) if overlay else None
        self.max_side = max_side
        self.quality = quality
        self.motion_s = 0.0

    def state(self) -> dict:
        s = self.sim.state()
        return {"joints": s["joints"], "grasp_point": s["grasp_point"], "amps": s["amps"]}

    def frames(self, cams: tuple[str, ...] = ("wrist", "side"), max_side: int | None = None, overlay: bool = True) -> dict[str, bytes]:
        out = {}
        side = max_side or self.max_side
        s = self.sim.state()
        for cam in cams:
            image: Image.Image = self.sim.frame(cam)
            if overlay and self.overlay:
                image = self.overlay.draw(image, cam, s["joints"], s["grasp_point"])
            image.thumbnail((side, side))
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=self.quality)
            out[cam] = buf.getvalue()
        return out

    def move_to(self, xyz: list[float], pitch_deg: float, roll_deg: float = -90.0) -> dict:
        """IK for the grasp point, then move. Raises Unreachable, saying which pitches would reach that point."""
        grip = self.sim.state()["goals"]["gripper"]
        try:
            q = self.ik.solve(xyz, float(pitch_deg), float(roll_deg), seed=self.sim.joints(), gripper=grip)
        except Unreachable as e:
            ok = [p for p in range(0, 91, 5) if self._reaches(xyz, p, roll_deg, grip)]
            raise Unreachable(f"{e}; " + (f"pitches from {min(ok)} to {max(ok)} deg reach it" if ok else "no pitch reaches it")) from None
        out = self._go(q)
        out["off_target_m"] = round(sum((a - b) ** 2 for a, b in zip(out["grasp_point"], xyz)) ** 0.5, 4)
        return out

    def _reaches(self, xyz, pitch: float, roll: float, grip: float) -> bool:
        try:
            self.ik.solve(xyz, float(pitch), float(roll), seed=self.sim.joints(), gripper=grip)
            return True
        except Unreachable:
            return False

    def move_joints(self, joints: dict[str, float]) -> dict:
        bad = set(joints) - set(ARM_JOINTS)
        if bad:
            raise ValueError(f"unknown joints {sorted(bad)}; use {list(ARM_JOINTS)}")
        return self._go({**self.sim.state()["goals"], **{j: float(v) for j, v in joints.items()}})

    def close(self) -> dict:
        """Squeeze until the jaws stop on something, then hold just past where they stopped."""
        self.sim.set_goals({"gripper": 0.0})
        last = self.sim.joints()["gripper"]
        for _ in range(60):
            self._step()
            now = self.sim.joints()["gripper"]
            if abs(now - last) < 0.05:
                break
            last = now
        self.sim.set_goals({"gripper": max(0.0, now - SQUEEZE_DEG)})
        self._step(0.3)
        return {"gripper_stopped_at": round(now, 1), **self.state()}

    def open(self, percent: float = DEFAULT_OPEN) -> dict:
        self.sim.set_goals({"gripper": float(percent)})
        for _ in range(40):
            self._step()
            if abs(self.sim.joints()["gripper"] - percent) < 1.0:
                break
        return self.state()

    def _go(self, goals: dict[str, float]) -> dict:
        self.sim.set_goals(goals)
        t = 0.0
        reached = False
        while t < MOVE_TIMEOUT_S:
            self._step()
            t += STEP_S
            now = self.sim.joints()
            if all(abs(now[j] - goals[j]) < SETTLE_DEG for j in ARM_JOINTS):
                reached = True
                break
        out = self.state()
        if not reached:
            out["note"] = f"stopped short of the goal after {MOVE_TIMEOUT_S:.0f} s"
        return out

    def _step(self, seconds: float = STEP_S) -> None:
        self.sim.step(seconds)
        self.motion_s += seconds
