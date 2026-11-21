"""The figure in REPORT.md: a table grid drawn on one close look (seed 3, all errors on) from FK, from the tag and
from the tag-calibrated arm, over the truth. `uv run python make_figure.py`."""

import cv2
import numpy as np
from PIL import Image
from valet_sim.ik import IK
from valet_sim.scene import TagSpec

from apriltag_exp.camera import Arm, Lens, on_plane
from apriltag_exp.localize import Poser, Trial, calibrate, make_sim
from apriltag_exp.look import look_at

seed, kind, tag = 3, "all", TagSpec()
fits = calibrate(seed, kind, tag)
sim, _ = make_sim(seed, kind, tag)
arm, lens = Arm(), Lens()
t = Trial(seed, kind, sim)
posers = {"fk": Poser(Arm(), False, tag), "tag": Poser(Arm(), True, tag), "cal": Poser(fits["full"].arm(), False, tag)}
start = t.grab("start", tag)
for p in posers.values():
    p(start)
t.go(look_at(arm, IK(arm.model), tuple(sim.truth()["target"]), 0.20))
f = t.grab("target", tag)
img = f.image.copy()
centre = on_plane(f.true_cam[:3, 3], f.true_cam[:3, 2], 0.0)
colours = {"fk": (230, 60, 40), "tag": (250, 200, 0), "cal": (40, 200, 80), "truth": (255, 255, 255)}
poses = {**{m: p(f) for m, p in posers.items()}, "truth": f.true_cam}
for name, T in poses.items():
    for i in range(-3, 4):
        for a, b in (([i * 0.04, -0.12], [i * 0.04, 0.12]), ([-0.12, i * 0.04], [0.12, i * 0.04])):
            s = np.linspace(0, 1, 25)[:, None]
            pts = np.column_stack([centre[0] + a[0] + (b[0] - a[0]) * s[:, 0], centre[1] + a[1] + (b[1] - a[1]) * s[:, 0], np.zeros(25)])
            px = lens.project(T, pts)
            ok = np.isfinite(px).all(1)
            cv2.polylines(img, [np.round(px[ok]).astype(np.int32)], False, colours[name], 1 if name == "truth" else 2, cv2.LINE_AA)
y = 18
for name, c in colours.items():
    cv2.putText(img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    y += 22
Image.fromarray(img).save("figures/grid_seed3_all_target.png")
