"""For approaches that draw a table grid on the wrist frame (as the Claude driver in PR #5 does, from FK and the
camera model): how far the drawn grid sits from where those table points truly appear, and where the arm goes
when told to go to a grid point, for grids drawn from FK, from the tag, and from the tag-calibrated arm.

Per frame (the start view and looks straight down at the box and the square), a 5 x 5 grid at 4 cm pitch on the
table around the point the camera looks at. Pixel error: drawn grid point minus its true projection. Table error:
where the drawn pixel truly lands on the table, minus the point it is labelled with. Reach: the arm told, through the
same model, to put its grasp point 4 cm above a grid point, minus that point (sampled at the grid's centre)."""

from __future__ import annotations

import numpy as np
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import BoxSpec, TagSpec

from .camera import Arm, Lens, on_plane
from .localize import Poser, Trial, calibrate, make_sim, reach
from .look import look_at

PITCH_M = 0.04


def run(seed: int, kind: str, tag: TagSpec = TagSpec()) -> dict:
    fits = calibrate(seed, kind, tag)
    sim, _ = make_sim(seed, kind, tag)
    arm, lens = Arm(), Lens()
    ik = IK(arm.model)
    t = Trial(seed, kind, sim)
    truth = sim.truth()
    looks = {"start": None, "box": (truth["box"]["x"], truth["box"]["y"]), "target": tuple(truth["target"])}
    out = {"seed": seed, "kind": kind, "views": {}}
    models = {"fk": (Arm(), False), "tag": (Arm(), True), "cal": (fits["full"].arm(), False)}
    posers = {m: Poser(a, use, tag) for m, (a, use) in models.items()}
    probe, _ = make_sim(seed, kind, None, box=BoxSpec(x=0.6, y=0.5), target_xy=None)
    for name, xy in looks.items():
        # the looks are aimed with the true positions: this measures the grid, not the localization before it
        if xy is not None:
            try:
                t.go(look_at(arm, ik, xy, 0.20))
            except Unreachable:
                continue
        f = t.grab(name, tag)
        centre = on_plane(f.true_cam[:3, 3], f.true_cam[:3, 2], 0.0)
        g = np.array([[centre[0] + i * PITCH_M, centre[1] + j * PITCH_M, 0.0] for i in range(-2, 3) for j in range(-2, 3)])
        true_px = lens.project(f.true_cam, g)
        seen = np.array([lens.visible(p, 5) for p in true_px])
        v = {"tag_seen": f.tag is not None}
        for m, poser in posers.items():
            T = poser(f)
            px = lens.project(T, g)
            d_px = np.linalg.norm(px - true_px, axis=1)[seen]
            lands = np.array([on_plane(*lens.ray(f.true_cam, p), 0.0) if np.all(np.isfinite(p)) else [np.nan] * 3 for p in px])
            d_cm = np.linalg.norm(lands[:, :2] - g[:, :2], axis=1)[seen] * 100
            r = {"px": float(np.nanmedian(d_px)), "cm": float(np.nanmedian(d_cm))}
            if name != "start":
                site = reach(probe, models[m][0], g[12, :2], 0.04)
                r["reach_cm"] = None if site is None else float(np.linalg.norm(site[:2] - g[12, :2]) * 100)
            v[m] = r
        out["views"][name] = v
    return out
