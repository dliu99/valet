"""How well a localize backend boxes the white box and the blue square, against the true projected outline.

    uv run python -m aspire.localize_check --backend h2o --seeds 100-104

Uses the wrist frame at the start pose. Truth is the bounding box of the projected true corners (grading only)."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image
from valet_sim.sim import CAMERAS, Sim, SimSpec, develop, with_box

from . import ledger, perceive, tasks
from .camera import Cameras


def _bbox(view, pts):
    px = np.array([p for p in (view.project(q) for q in pts) if p is not None])
    return [px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m aspire.localize_check")
    p.add_argument("--backend", default="claude", choices=["claude", "jev", "h2o"])
    p.add_argument("--task", default="benchmark")
    p.add_argument("--seeds", default="100-104")
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    ledger.load_env()
    rows = []
    for seed in range(lo, hi + 1):
        ep = tasks.make(a.task, seed)
        b = ep.box
        sim = Sim(with_box(SimSpec(), b, ep.target_xy))
        raw = sim.renderer.render(CAMERAS["wrist"])
        key = perceive.frame_key(raw)
        img = develop(Image.fromarray(raw), sim.spec.optics, np.random.default_rng(int(key[:8], 16)))
        view = Cameras(sim.model).view("wrist", sim.joints())
        yaw = np.radians(b.yaw_deg)
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        box_pts = [[*(np.array([b.x, b.y]) + rot @ [sx * b.w / 2, sy * b.d / 2]), z] for z in (0, b.h) for sx in (-1, 1) for sy in (-1, 1)]
        tx, ty = ep.target_xy
        sq_pts = [[tx + sx * 0.035, ty + sy * 0.035, 0] for sx in (-1, 1) for sy in (-1, 1)]
        for name, text, pts in (("box", "the white box", box_pts), ("square", "the blue square", sq_pts)):
            truth = _bbox(view, pts)
            t0 = time.time()
            f = perceive.locate(img, text, key, a.backend)
            row = {"seed": seed, "object": name, "found": f.found, "s": round(time.time() - t0, 2), "cached": f.cached}
            if f.bbox:
                c_err = np.hypot((f.bbox[0] + f.bbox[2] - truth[0] - truth[2]) / 2, (f.bbox[1] + f.bbox[3] - truth[1] - truth[3]) / 2)
                row.update(iou=round(_iou(f.bbox, truth), 3), centre_err_px=round(float(c_err), 1), confidence=f.confidence)
            rows.append(row)
            print(json.dumps(row), flush=True)
    for name in ("box", "square"):
        r = [x for x in rows if x["object"] == name and "iou" in x]
        if r:
            print(f"{a.backend} {name}: IoU median {np.median([x['iou'] for x in r]):.2f} min {min(x['iou'] for x in r):.2f}, "
                  f"centre error median {np.median([x['centre_err_px'] for x in r]):.1f} px max {max(x['centre_err_px'] for x in r):.1f} px, "
                  f"found {len(r)}/{sum(1 for x in rows if x['object'] == name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
