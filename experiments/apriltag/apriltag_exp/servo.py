"""The cheap probe for closed-loop approaches: servo the gripper over the box on the image alone, then descend, and
see what error is left for a tag to fix.

Above the box, the policy moves the grasp point until the box's centroid sits on a reference pixel, the pixel where
a box top right below the grasp point should appear. The camera rides on the gripper, so this loop cancels the
joints' zero errors and sag at the servo pose; what it cannot cancel is a wrong reference (a mount error, or the
box's height) and whatever changes on the way down. References:

  nominal  computed from the nominal camera model, like everything else the policy knows
  taught   where a box truly under the grasp point appears, as one would record it once on the real arm by
           holding a box between the open jaws and saving the pixel (here read from the sim's truth)
"""

from __future__ import annotations

import numpy as np
from valet_sim.scene import BoxSpec

from . import detect
from .camera import Arm, Lens, on_plane, true_camera
from .localize import ASSUMED_BOX_H, GRASP_Z, make_sim, reach, settle, solve

PREGRASP_Z = 0.15
TOLERANCE_PX = 1.5
MAX_STEPS = 8


def run(seed: int, kind: str, reference: str = "nominal") -> dict:
    sim, _ = make_sim(seed, kind, None)
    arm, lens = Arm(), Lens()
    jaw = detect.arm_mask(sim)
    image = np.asarray(sim.frame("wrist"))
    blob = detect.white(image, jaw)
    truth = sim.truth()["box"]
    out = {"seed": seed, "kind": kind, "reference": reference}
    if blob is None:
        return {**out, "failed": "box not seen"}
    site = on_plane(*lens.ray(arm.camera(sim.joints()), blob.centroid_px), ASSUMED_BOX_H)[:2]
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        q = solve(arm, [site[0], site[1], PREGRASP_Z], pitches=(85.0, 80.0, 75.0, 70.0, 65.0, 60.0))
        if q is None:
            return {**out, "failed": "pregrasp unreachable"}
        settle(sim, arm.command(q))
        image = np.asarray(sim.frame("wrist"))
        blob = detect.white(image, jaw)
        if blob is None:
            return {**out, "failed": "lost the box"}
        if reference == "taught":
            t = sim.truth()
            under = np.r_[np.array(t["grasp_point"][:2]), ASSUMED_BOX_H]
            ref = lens.project(true_camera(t), under)[0]
        else:
            j = sim.joints()
            ref = lens.project(arm.camera(j), np.r_[arm.grasp_point(j)[:2], ASSUMED_BOX_H])[0]
        err_px = blob.centroid_px - ref
        if np.linalg.norm(err_px) < TOLERANCE_PX:
            break
        # move the grasp point by where the two pixels land on the box-top plane, by the model's camera
        T = arm.camera(sim.joints())
        a, b = (on_plane(*lens.ray(T, uv), ASSUMED_BOX_H) for uv in (blob.centroid_px, ref))
        site = site + (a - b)[:2]
    # descend, measured where the box cannot get in the way: the same arm, the box moved off the table
    probe, _ = make_sim(seed, kind, None, box=BoxSpec(x=0.6, y=0.5), target_xy=None)
    end = reach(probe, arm, site, GRASP_Z)
    if end is None:
        return {**out, "failed": "grasp unreachable"}
    reach_xy = end[:2] - [truth["x"], truth["y"]]
    return {**out, "steps": steps, "final_px": float(np.linalg.norm(err_px)), "reach_mm": float(np.linalg.norm(reach_xy) * 1000),
            "box_h": truth["h"]}
