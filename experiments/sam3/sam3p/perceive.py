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
    seconds: dict[str, float] = field(default_factory=dict)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


class Perception:
    def __init__(self, segmenter, cam: Camera | None = None, jaw: JawMask | None = None):
        self.seg = segmenter
        self.cam = cam or Camera()
        self.jaw = jaw or JawMask()

    def pick(self, frame: Image.Image, joints: dict[str, float]) -> tuple[Instance | None, Instance | None, np.ndarray, dict]:
        t0 = time.time()
        out = self.seg.segment(frame, [BOX_PROMPT, SQUARE_PROMPT])
        t_seg = time.time() - t0
        jaw = _dilate(self.jaw(joints["gripper"]), 3)
        box = None
        for inst in out[BOX_PROMPT]:
            on_jaw = (inst.mask & jaw).sum() / max(inst.area, 1)
            if on_jaw <= JAW_OVERLAP_MAX:
                box = Instance(inst.mask & ~jaw, inst.score, inst.box)
                break
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
        v = View(joints, box_inst=box, square_inst=square, seconds=secs)
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
