"""Realistic errors to switch on in the sim, drawn per trial from a seed, so every method sees the same arm.

Sizes follow what is known of the real SO-101: joint zeros about 5 deg off, ~1.5 cm of sag at reach, a camera pose
fitted from two frames (a few mm and a couple of degrees), and a tag taped down by hand (2 mm, 1 deg)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from valet_sim.scene import TagSpec
from valet_sim.sim import Errors

KINDS = ("none", "offsets", "sag", "mount", "all")
OFFSET_DEG = (2.0, 6.0)  # each arm joint's zero is off by this much, either way
COMPLIANCE_DEG_PER_NM = 4.0
MOUNT_SHIFT_SD_M = 0.004
MOUNT_TILT_SD_DEG = 2.0
TAG_SHIFT_SD_M = 0.002
TAG_YAW_SD_DEG = 1.0


def draw(seed: int, kind: str, tag: TagSpec = TagSpec()) -> tuple[Errors, TagSpec]:
    """The sim's errors and the tag's true place for one trial. The policy keeps believing `tag`."""
    rng = np.random.default_rng(10_000 + seed)
    offsets = rng.choice([-1, 1], 5) * rng.uniform(*OFFSET_DEG, 5)
    shift = rng.normal(0, MOUNT_SHIFT_SD_M, 3)
    tilt = rng.normal(0, MOUNT_TILT_SD_DEG, 3)
    tag_shift = rng.normal(0, TAG_SHIFT_SD_M, 2)
    tag_yaw = rng.normal(0, TAG_YAW_SD_DEG)
    on = {"offsets": kind in ("offsets", "all"), "sag": kind in ("sag", "all"), "mount": kind in ("mount", "all")}
    errors = Errors(
        joint_offsets_deg=tuple(float(v) for v in offsets) + (0.0,) if on["offsets"] else (0.0,) * 6,
        compliance_deg_per_nm=COMPLIANCE_DEG_PER_NM if on["sag"] else 0.0,
        mount_shift_m=tuple(float(v) for v in shift) if on["mount"] else (0.0, 0.0, 0.0),
        mount_tilt_deg=tuple(float(v) for v in tilt) if on["mount"] else (0.0, 0.0, 0.0),
    )
    true_tag = replace(tag, x=tag.x + float(tag_shift[0]), y=tag.y + float(tag_shift[1]),
                       yaw_deg=tag.yaw_deg + float(tag_yaw)) if kind == "all" else tag
    return errors, true_tag
