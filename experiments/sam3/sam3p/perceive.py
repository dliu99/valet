"""One look: SAM 3 masks for the box and the square in a wrist frame, then the fits that put them on the table.
The prompts are the task's own words; nothing else reaches the segmenter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .camera import Camera
from .fit import BoxFit, SquareFit, fit_box, fit_square
from .render import JawMask
from .seg import Instance

BOX_PROMPT = "white box"
SQUARE_PROMPT = "blue square"
TARGET_SIZE_M = 0.07  # the blue square's side, a fact of the task setup
JAW_OVERLAP_MAX = 0.3  # a "white box" mostly on the gripper's own pixels is the white jaw
BOX_IN_SQUARE_MAX = 0.5  # a "blue square" mostly on the box is the box


@dataclass
class View:
    joints: dict[str, float]
    box: BoxFit | None = None
    square: SquareFit | None = None
    box_inst: Instance | None = None
    square_inst: Instance | None = None
    box_frac: float = 0.0  # share of the frame the box mask fills
    ignore: np.ndarray | None = None  # the gripper's own pixels
    seconds: dict[str, float] = field(default_factory=dict)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def solidity(mask: np.ndarray) -> float:
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None:
        return 0.0
    return float(mask.sum() / max(cv2.contourArea(cv2.convexHull(pts)), 1.0))


class Perception:
    """`box_prompt`: the words SAM 3 gets for the box. `merge`: join the best box instance with other box
    instances touching it when the union stays convex, since a face in shadow can come back as its own instance."""

    def __init__(self, segmenter, cam: Camera | None = None, jaw: JawMask | None = None, box_prompt: str = BOX_PROMPT,
                 merge: bool = False, fallback: str | None = None):
        self.seg = segmenter
        self.cam = cam or Camera()
        self.jaw = jaw or JawMask()
        self.box_prompt = box_prompt
        self.merge = merge
        self.fallback = fallback  # a second wording for the box, tried on the same frame when the first finds nothing

    def pick(self, frame: Image.Image, joints: dict[str, float]) -> tuple[Instance | None, Instance | None, np.ndarray, dict]:
        t0 = time.time()
        out = self.seg.segment(frame, [self.box_prompt, SQUARE_PROMPT])
        t_seg = time.time() - t0
        jaw = _dilate(self.jaw(joints["gripper"]), 3)
        box = None
        cands = [i for i in out[self.box_prompt] if (i.mask & jaw).sum() / max(i.area, 1) <= JAW_OVERLAP_MAX]
        if not cands and self.fallback:
            more = self.seg.segment(frame, [self.fallback])[self.fallback]
            cands = [i for i in more if (i.mask & jaw).sum() / max(i.area, 1) <= JAW_OVERLAP_MAX]
            t_seg = time.time() - t0
        if cands:
            best = cands[0]
            mask = best.mask & ~jaw
            if self.merge:
                for other in cands[1:]:
                    grown = mask | (other.mask & ~jaw)
                    if (_dilate(mask, 3) & other.mask).any() and grown.sum() <= 2.5 * mask.sum() and solidity(grown) >= 0.9:
                        mask = grown
            ys, xs = np.nonzero(mask)
            box = None if len(xs) == 0 else Instance(mask, best.score, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
        square = None
        for inst in out[SQUARE_PROMPT]:
            if box is not None and (inst.mask & box.mask).sum() / max(inst.area, 1) > BOX_IN_SQUARE_MAX:
                continue
            if (inst.mask & jaw).sum() / max(inst.area, 1) > JAW_OVERLAP_MAX:
                continue
            square = inst
            break
        return box, square, jaw, {"segment": round(t_seg, 3)}

    def look(self, frame: Image.Image, joints: dict[str, float], hw: tuple[float, float] | None = None,
             fit: tuple[str, ...] = ("box", "square")) -> View:
        box, square, jaw, secs = self.pick(frame, joints)
        v = View(joints, box_inst=box, square_inst=square, seconds=secs, ignore=jaw)
        t0 = time.time()
        if box is not None:
            v.box_frac = box.area / box.mask.size
            if "box" in fit:
                v.box = fit_box(self.cam, joints, box.mask, jaw, hw)
        if square is not None and "square" in fit:
            hide = jaw | (_dilate(box.mask, 2) if box is not None else False)
            v.square = fit_square(self.cam, joints, square.mask & ~hide, TARGET_SIZE_M, hide)
        v.seconds["fit"] = round(time.time() - t0, 3)
        return v
