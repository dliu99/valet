"""Part 4: a minimal wrist-only localize-then-reach policy, run with and without the tag.

Look (start view, then straight down at the box and at the square), localize (colour detectors, back-projected
through the method's camera poses), then grasp and place with IK through the method's arm model, borrowing the
scripted expert's grasp planner and motions. Checks are code: the gripper reading after closing (it stalls on a box
and closes fully on air) and the box's size in the wrist frame before and after the lift; a no on either retries the
grasp once from a fresh look. Jev was tried as the check and dropped; see REPORT.md.

Methods (see `localize`): `fk` has no tag in the scene; `tag` uses it at run time; `cal` uses it only once,
beforehand, to fit the arm's mount, joint zeros and sag, then runs without it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from valet_sim import expert
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import BoxSpec, TagSpec
from valet_sim.task import score

from . import detect
from .camera import Arm, Lens, on_plane
from .localize import ASSUMED_BOX_H, TARGET_Z, Frame, Poser, Trial, _exclude, calibrate, make_sim
from .look import look_at

BOX_W_RANGE = (0.04, 0.055)
GRASP_Z = 0.04
HELD_GRIPPER = (5.0, 50.0)
HELD_SIZE_RATIO = 0.92
RETREAT_M = 0.12


class Episode:
    def __init__(self, seed: int, kind: str, method: str, fits: dict | None = None, tag: TagSpec = TagSpec()):
        self.seed, self.kind, self.method, self.tag = seed, kind, method, tag
        self.sim, _ = make_sim(seed, kind, tag if method == "tag" else None)  # cal: the tag is gone by run time
        self.arm = fits["full"].arm() if method == "cal" else Arm()
        self.ik = IK(self.arm.model)
        self.lens = Lens()
        self.poser = Poser(self.arm, method == "tag", tag)
        self.trial = Trial(seed, kind, self.sim)
        self.jaw = detect.arm_mask(self.sim)
        self.log: list[dict] = []

    # -- seeing -------------------------------------------------------------------------------------------------

    def look(self, name: str) -> tuple[Frame, np.ndarray]:
        f = self.trial.grab(name, self.tag if self.method == "tag" else None)
        return f, self.poser(f)

    def find_box(self, f: Frame, T: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        """Box centre on the table, its yaw (mod 90) and its side, from one look straight down."""
        blob = detect.white(f.image, _exclude(self.jaw, f, self.lens))
        if blob is None:
            return None
        c = on_plane(*self.lens.ray(T, blob.centroid_px), ASSUMED_BOX_H)
        if c is None:
            return None
        edge = blob.pixels[:: max(1, len(blob.pixels) // 800)]
        pts = [on_plane(*self.lens.ray(T, uv), ASSUMED_BOX_H) for uv in edge]
        pts = np.array([p[:2] for p in pts if p is not None], np.float32)
        (_, _), (w, h), ang = cv2.minAreaRect(pts)
        side = float(np.clip(min(w, h), *BOX_W_RANGE))
        return c, float(ang % 90), side

    def find_target(self, f: Frame, T: np.ndarray) -> np.ndarray | None:
        blob = detect.blue(f.image, _exclude(self.jaw, f, self.lens))
        return None if blob is None else detect.on_table(blob, T, TARGET_Z, self.lens)

    # -- moving -------------------------------------------------------------------------------------------------

    def go(self, q_model: dict[str, float], timeout_s: float = 4.0) -> None:
        expert.move(self.sim, self.arm.command(q_model), timeout_s)

    def go_reading(self, goal: dict[str, float], settle_s: float = 2.5) -> None:
        self.sim.set_goals(goal)
        self.sim.step(settle_s)

    def held(self) -> tuple[bool, float]:
        """Code's check after the lift: a held box rose with the camera and fills the frame as before; a box left on
        the table looks smaller from 5 cm higher (held 0.94-1.04 of its size at the grasp, left 0.64-0.87)."""
        before = detect.white(self.closed_frame, self.jaw)
        after = detect.white(np.asarray(self.sim.frame("wrist")), self.jaw)
        ratio = 0.0 if before is None or after is None else len(after.pixels) / len(before.pixels)
        return ratio >= HELD_SIZE_RATIO, ratio

    # -- the task -----------------------------------------------------------------------------------------------

    def run(self) -> dict:
        t0 = time.time()
        a, T_a = self.look("start")
        found = self.find_box(a, T_a)
        target0 = self.find_target(a, T_a)
        if found is None or target0 is None:
            return self.finish(t0, "start view: box or square not seen")
        try:
            self.go_reading(look_at(self.arm, self.ik, found[0][:2], 0.20))
            f, T = self.look("box")
            found = self.find_box(f, T) or found
            self.go_reading(look_at(self.arm, self.ik, target0[:2], 0.20))
            f, T = self.look("target")
            target = self.find_target(f, T)
            target = target0 if target is None else target
        except Unreachable as e:
            return self.finish(t0, f"look: {e}")
        self.estimates = {"box": found[0].tolist(), "yaw": found[1], "side": found[2], "target": target.tolist()}

        for attempt in range(2):
            ok, note = self.grasp(found)
            if ok:
                return self.place(t0, target)
            self.log.append({"attempt": attempt, "grasp": note})
            if attempt == 0:  # look again: the box may have been nudged
                self.go_reading({**self.sim.joints(), "gripper": expert.OPEN_PCT}, 1.0)
                try:
                    self.go_reading(look_at(self.arm, self.ik, found[0][:2], 0.20))
                except Unreachable:
                    break
                f, T = self.look("box_retry")
                found = self.find_box(f, T) or found
        return self.finish(t0, "not held after two grasps")

    def grasp(self, found) -> tuple[bool, str]:
        centre, yaw, side = found
        c = np.array([centre[0], centre[1], GRASP_Z])
        box = BoxSpec(x=c[0], y=c[1], yaw_deg=yaw, w=side, d=side)
        try:
            path, self.pitch = expert.plan_grasp(self.ik, c, box)
        except Unreachable as e:
            return False, f"plan: {e}"
        for k, q in enumerate(path):
            self.go(q, timeout_s=3.0 if k == 0 else 1.5)
        expert.close_gripper(self.sim)
        self.closed_frame = np.asarray(self.sim.frame("wrist"))
        g = self.sim.joints()["gripper"]
        if not HELD_GRIPPER[0] < g < HELD_GRIPPER[1]:
            return False, f"gripper closed to {g:.0f}: nothing between the jaws"
        self.grip, self.roll, self.centre = g, path[-1]["wrist_roll"], c
        try:
            self.lift = expert.solve_near(self.ik, c + [0, 0, expert.LIFT_M], self.pitch, self.roll, path[-1], g)
        except Unreachable as e:
            return False, f"lift: {e}"
        self.go(self.lift)
        ok, ratio = self.held()
        self.log.append({"check": "held:code", "size_ratio": round(ratio, 3), "held": ok, "truth_held": self.sim.truth()["held"]})
        return (True, "held") if ok else (False, f"box did not rise with the gripper (size ratio {ratio:.2f})")

    def place(self, t0: float, target) -> dict:
        tx, ty = target[0], target[1]
        g = self.grip
        try:
            carry = expert.solve_near(self.ik, [tx, ty, self.centre[2] + expert.LIFT_M], self.pitch, self.roll, self.lift, g)
            down = expert.solve_near(self.ik, [tx, ty, self.centre[2] + 0.004], self.pitch, self.roll, carry, g)
        except Unreachable as e:
            return self.finish(t0, f"carry: {e}")
        self.go(carry)
        self.go(down)
        self.go({**down, "gripper": expert.OPEN_PCT})
        try:  # straight up past the top of the tallest box: 5 cm up still leaves the jaw tips beside a 9 cm box
            up = expert.solve_near(self.ik, [tx, ty, self.centre[2] + RETREAT_M], self.pitch, self.roll, carry, expert.OPEN_PCT)
        except Unreachable:
            up = carry
        self.go({**up, "gripper": expert.OPEN_PCT})
        self.sim.step(0.5)
        return self.finish(t0, None)

    def finish(self, t0: float, note: str | None) -> dict:
        truth = self.sim.truth()
        return {"seed": self.seed, "kind": self.kind, "method": self.method, **score(truth), "note": note,
                "estimates": getattr(self, "estimates", None),
                "truth": {"box": [truth["box"]["x"], truth["box"]["y"]], "target": truth["target"]},
                "log": self.log, "sim_s": round(float(self.sim.data.time), 1),
                "wall_s": round(time.time() - t0, 1), "tag_seen": {k: f.tag is not None for k, f in self.trial.frames.items()}}


def run(seed: int, kind: str, method: str, tag: TagSpec = TagSpec()) -> dict:
    fits = calibrate(seed, kind, tag) if method == "cal" else None
    return Episode(seed, kind, method, fits, tag).run()
