"""Segment, fit, reach: the box and the square are located by SAM 3 masks put on the table in code, IK reaches,
and a last look checks the placement. No language model picks anything; the prompts are the task's own nouns.

The policy reads only `sim.joints()` and wrist frames. `witness` lets a benchmark grade each estimate as it is
made; nothing it does reaches the policy. Grasp mechanics follow point-and-reach (claude/point-and-reach-0d3ca8):
straight down, the box in the middle of the open jaws, gripped 2.5 cm below its top."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from valet_sim.ik import IK, Unreachable
from valet_sim.sim import Sim

from .camera import Camera
from .fit import BoxFit, SquareFit, fit_box_views
from .perceive import Perception, View
from .reach import close, look_from_above, move, set_gripper, solve

OPEN = 70.0
GRASP_PITCH = 88.0
GRASP_BELOW_TOP_M = 0.025  # grip above the centre of mass so the box hangs instead of tipping
GRASP_Z = (0.03, 0.06)
CARRY_Z = 0.12
PLACE_DROP_M = 0.004
GAP_MIDDLE_M = 0.025  # the middle of the open jaws, from the grasp point along the closing axis
FIXED_JAW_M = 0.020  # the fixed jaw's inner face, from the grasp point against the closing axis
APPROACH_M = 0.07
LOOK_HEIGHT = 0.22
EMPTY_BELOW = 5.0  # gripper reading: jaws closed this far have nothing between them
PLACED_WITHIN_M = 0.02
UPRIGHT_IOU = 0.85  # an upright-box silhouette that fits worse than this is not an upright box
SURVEY_SPOTS = ((0.24, 0.12), (0.24, -0.12), (0.30, 0.0))


@dataclass
class Config:
    name: str = "sam3"
    grasp_tries: int = 3
    place_tries: int = 2
    pre_grasp_look: bool = False  # one more look from 15 cm before descending
    align_roll: bool = False  # turn the wrist so the jaws close square to the box's faces
    place_offset: bool = True  # put the box (pushed against the fixed jaw) on the square, not the grasp point
    two_view: bool = False  # fit the box to the survey and the look from above together


@dataclass
class Estimate:
    box: BoxFit | None = None
    hw: tuple[float, float] | None = None  # side, height from an oblique look
    square: SquareFit | None = None


class SamPolicy:
    def __init__(self, sim: Sim, perception: Perception, cfg: Config = Config(),
                 witness: Callable[[str, dict], None] | None = None):
        self.sim = sim
        self.p = perception
        self.cfg = cfg
        self.witness = witness
        self.cam: Camera = perception.cam
        self.ik = IK(self.cam.model)
        self.est = Estimate()
        self.events: list[dict] = []
        self.looks = 0
        self.survey_view: View | None = None

    # --- sensing ---------------------------------------------------------------------------------------------

    def log(self, what: str, **kw) -> None:
        self.events.append({"t": round(self.sim.data.time, 2), "what": what, **kw})

    def look(self, tag: str, fit=("box", "square")) -> View:
        joints = self.sim.joints()
        v = self.p.look(self.sim.frame("wrist"), joints, hw=self.est.hw if tag != "survey" else None, fit=fit)
        self.looks += 1
        rec = {"tag": tag, "box": None if v.box is None else [round(v.box.x, 4), round(v.box.y, 4), round(v.box.yaw_deg, 1),
                                                                round(v.box.side, 4), round(v.box.h, 4), round(v.box.iou, 3)],
               "square": None if v.square is None else [round(v.square.x, 4), round(v.square.y, 4), round(v.square.iou, 3)],
               "box_frac": round(v.box_frac, 3), "s": v.seconds}
        self.log("look", **rec)
        if self.witness:
            self.witness(tag, rec)
        return v

    def take_box(self, v: View, oblique: bool) -> bool:
        if v.box is None or v.box.iou < 0.5:
            return False
        self.est.box = v.box
        if oblique and self.est.hw is None:
            self.est.hw = (v.box.side, v.box.h)
        return True

    def take_square(self, v: View) -> bool:
        if v.square is None or v.square.iou < 0.5:
            return False
        if self.est.square is None or v.square.iou >= self.est.square.iou - 0.05:
            self.est.square = v.square
        return True

    def look_above(self, xy, height: float = LOOK_HEIGHT, gripper: float | None = None) -> None:
        g = self.sim.joints()["gripper"] if gripper is None else gripper
        move(self.sim, look_from_above(self.ik, self.cam, xy, height, gripper=g))

    # --- the task ----------------------------------------------------------------------------------------------

    def survey(self) -> bool:
        v = self.look("survey")
        if self.take_box(v, oblique=True):
            self.survey_view = v
        self.take_square(v)
        for xy in SURVEY_SPOTS:
            if self.est.box is not None and self.est.square is not None:
                break
            try:
                self.look_above(xy, 0.24)
            except Unreachable:
                continue
            v = self.look("survey-aside")
            if self.est.box is None:
                self.take_box(v, oblique=False)
            if self.est.square is None:
                self.take_square(v)
        return self.est.box is not None and self.est.square is not None

    def refine_box(self, tag: str) -> bool:
        try:
            self.look_above(self.est.box.xy, gripper=OPEN)
        except Unreachable as e:
            self.log("unreachable", error=str(e))
            return True  # keep what the survey saw
        v = self.look(tag)
        if not self.take_box(v, oblique=False):
            return False
        if self.cfg.two_view and self.survey_view is not None and v.box_inst is not None:
            sv = self.survey_view
            both = fit_box_views(self.cam, [(sv.joints, sv.box_inst.mask, sv.ignore), (v.joints, v.box_inst.mask, v.ignore)])
            self.log("two-view", box=None if both is None else [round(both.x, 4), round(both.y, 4), round(both.side, 4), round(both.h, 4), round(both.iou, 3)])
            if both is not None and both.iou >= 0.8:
                self.est.box, self.est.hw = both, (both.side, both.h)
                if self.witness:
                    self.witness(f"{tag}-two-view", {"box": [both.x, both.y], "square": None})
            self.survey_view = None  # the box may move from here on
        if self.est.hw is None:  # never seen obliquely: take this look's own size
            self.est.hw = (v.box.side, v.box.h)
        if v.square is not None:
            self.take_square(v)
        return True

    def closing_axis(self, q: dict[str, float]) -> np.ndarray:
        self.ik.pose(q)
        return self.ik.data.xmat[self.ik.gripper].reshape(3, 3)[:, 0].copy()

    def roll_for(self, q: dict[str, float], yaw_deg: float) -> float:
        """The wrist roll near -90 that turns the closing axis square to the box's faces."""
        ax = self.closing_axis(q)
        now = np.degrees(np.arctan2(ax[1], ax[0]))
        turn = (yaw_deg - now + 45) % 90 - 45
        return float(np.clip(-90.0 - turn, -150, -30))

    def grasp_path(self) -> tuple[list[dict], float, np.ndarray]:
        side, h = self.est.hw
        b = self.est.box
        self.grasp_z = float(np.clip(h - GRASP_BELOW_TOP_M, *GRASP_Z))
        centre = np.array([b.x, b.y, self.grasp_z])
        q, pitch = solve(self.ik, centre, GRASP_PITCH)
        roll = self.roll_for(q, b.yaw_deg) if self.cfg.align_roll else -90.0
        if roll != -90.0:
            q, pitch = solve(self.ik, centre, pitch, roll, seed=q)
        site = centre - self.closing_axis(q) * GAP_MIDDLE_M
        q, pitch = solve(self.ik, site, pitch, roll, seed=q)
        self.ik.pose(q)
        approach = -self.ik.data.xmat[self.ik.gripper].reshape(3, 3)[:, 2]
        path, seed = [], q
        for k in (1.0, 0.6, 0.3, 0.0):
            seed, _ = solve(self.ik, site - approach * APPROACH_M * k, pitch, roll, seed=seed)
            path.append(seed)
        self.roll = roll
        return path, pitch, self.closing_axis(path[-1])

    def grasp(self) -> bool:
        for attempt in range(self.cfg.grasp_tries):
            if not self.refine_box(f"above-box-{attempt}"):
                self.log("lost the box")
                if not self.survey_again():
                    return False
                continue
            try:
                path, pitch, _ = self.grasp_path()
            except Unreachable as e:
                self.log("unreachable", error=str(e))
                return False
            if self.cfg.pre_grasp_look:
                move(self.sim, path[0], timeout_s=2.0)
                v = self.look(f"pre-grasp-{attempt}", fit=("box",))
                if self.take_box(v, oblique=False):
                    try:
                        path, pitch, _ = self.grasp_path()
                    except Unreachable as e:
                        self.log("unreachable", error=str(e))
                        return False
            for q in path:
                move(self.sim, q, timeout_s=1.5)
            g = close(self.sim)
            q = self.sim.joints()
            lift, _ = solve(self.ik, self.ik.pose(q)[0] + [0, 0, 0.06], pitch, self.roll, seed=q, gripper=q["gripper"])
            move(self.sim, lift)
            g = self.sim.joints()["gripper"]
            self.log("closed", gripper=round(g, 1))
            if g > EMPTY_BELOW:
                self.pitch = pitch
                return True
            set_gripper(self.sim, OPEN)
            move(self.sim, path[0])
        return False

    def survey_again(self) -> bool:
        """The box moved or vanished: look from where it was, then the survey spots."""
        for xy in ((self.est.box.xy,) if self.est.box is not None else ()) + SURVEY_SPOTS:
            try:
                self.look_above(xy, 0.24, gripper=OPEN)
            except Unreachable:
                continue
            if self.take_box(self.look("search"), oblique=False):
                return True
        return False

    def place(self) -> None:
        q = self.sim.joints()
        g = q["gripper"]
        sq = self.est.square.xy
        target = np.array([sq[0], sq[1], CARRY_Z])
        carry, pitch = solve(self.ik, target, self.pitch, self.roll, seed=q, gripper=g)
        if self.cfg.place_offset:  # after closing, the box sits against the fixed jaw, not at the grasp point
            ax = self.closing_axis(carry)
            shift = ax * (self.est.hw[0] / 2 - FIXED_JAW_M)
            target[:2] -= shift[:2]
            carry, pitch = solve(self.ik, target, pitch, self.roll, seed=carry, gripper=g)
        move(self.sim, carry)
        down, _ = solve(self.ik, [target[0], target[1], self.grasp_z + PLACE_DROP_M], pitch, self.roll, seed=carry, gripper=g)
        move(self.sim, down)
        set_gripper(self.sim, OPEN)
        move(self.sim, {**carry, "gripper": OPEN})

    def check_placed(self) -> tuple[bool, str]:
        try:
            self.look_above(self.est.square.xy, gripper=OPEN)
        except Unreachable:
            return True, "no look"
        v = self.look("after-place")
        if v.box is None:
            return False, "box not seen"
        self.est.box = v.box
        off = float(np.hypot(*(v.box.xy - self.est.square.xy)))
        upright = v.box.iou >= UPRIGHT_IOU
        self.log("placed?", off_cm=round(off * 100, 2), iou=round(v.box.iou, 3), upright=upright)
        if not upright:
            return True, "tipped"  # nothing here can stand it back up
        return off <= PLACED_WITHIN_M, "off" if off > PLACED_WITHIN_M else "ok"

    def run(self) -> dict:
        t0 = time.time()
        outcome = "lost"
        if self.survey():
            outcome = "missed"
            for _ in range(self.cfg.place_tries):
                if not self.grasp():
                    break
                self.place()
                ok, why = self.check_placed()
                outcome = "done" if ok and why == "ok" else why
                if ok:
                    break
        self.sim.step(0.5)
        return {"outcome": outcome, "sim_s": round(self.sim.data.time, 1), "wall_s": round(time.time() - t0, 1), "looks": self.looks}
