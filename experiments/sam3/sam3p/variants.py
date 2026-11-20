"""Start + above views on seeds 20-39 (504 px): box fit errors for prompt, merge and two-view variants.

    uv run python -m sam3p.variants
"""

import numpy as np
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode
from valet_sim.ik import IK
from .camera import Camera
from .render import JawMask, SegRenderer, true_masks
from .reach import look_from_above, move
from .seg import Segmenter
from .perceive import Perception, _dilate
from .fit import fit_box, fit_box_views


def main() -> None:
    cam = Camera(); ik = IK(cam.model); jaw = JawMask()
    seg = Segmenter(resolution=504)
    variants = {"white box": Perception(seg, cam, jaw, "white box"), "white box+merge": Perception(seg, cam, jaw, "white box", True),
                "white cube": Perception(seg, cam, jaw, "white cube"), "white cube+merge": Perception(seg, cam, jaw, "white cube", True)}
    res = {k: {"start": [], "start_h": [], "above_hw": [], "two": [], "two_h": [], "iou_s": []} for k in variants}
    rng = np.random.default_rng(1)
    for seed in range(20, 40):
        ep = episode(seed); sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy)); sr = SegRenderer(sim.model)
        sim.step(0.3)
        f1, j1 = sim.frame("wrist"), sim.joints(); t1 = true_masks(sr, sim)
        move(sim, look_from_above(ik, cam, np.array([ep.box.x, ep.box.y]) + rng.uniform(-0.015, 0.015, 2), 0.22)); sim.step(0.3)
        f2, j2 = sim.frame("wrist"), sim.joints(); t2 = true_masks(sr, sim)
        tb = sim.truth()["box"]
        err = lambda f: None if f is None else round(100 * float(np.hypot(f.x - tb["x"], f.y - tb["y"])), 2)
        line = [str(seed)]
        for k, p in variants.items():
            b1, _, jm1, _ = p.pick(f1, j1); b2, _, jm2, _ = p.pick(f2, j2)
            s = None if b1 is None else fit_box(cam, j1, b1.mask, jm1)
            a = None if (b2 is None or s is None) else fit_box(cam, j2, b2.mask, jm2, (s.side, s.h))
            views = [(j, b.mask, m) for j, b, m in ((j1, b1, jm1), (j2, b2, jm2)) if b is not None]
            t = fit_box_views(cam, views) if views else None
            r = res[k]
            r["start"].append(err(s)); r["start_h"].append(None if s is None else abs(100 * (s.h - tb["h"])))
            r["above_hw"].append(err(a)); r["two"].append(err(t)); r["two_h"].append(None if t is None else abs(100 * (t.h - tb["h"])))
            r["iou_s"].append(0 if b1 is None else float((b1.mask & t1["box"]).sum() / max((b1.mask | t1["box"]).sum(), 1)))
            line.append(f"{k}: s{err(s)} a{err(a)} 2v{err(t)} h{None if t is None else round(100*(t.h-tb['h']),1)}")
        print(" | ".join(line), flush=True)
    def st(v):
        v = [x if x is not None else 99 for x in v]; v = np.array(v, float)
        return f"med {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} max {v.max():.2f} >1cm {(v > 1).sum()}"
    for k, r in res.items():
        print(f"{k:18s} start-view mask IoU<0.8 {sum(i < 0.8 for i in r['iou_s'])}/20 | start {st(r['start'])} | above w/ start size {st(r['above_hw'])} | two-view {st(r['two'])} | two-view |h err| {st(r['two_h'])} | start |h err| {st(r['start_h'])}")


if __name__ == "__main__":
    main()
