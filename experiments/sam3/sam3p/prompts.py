"""Box mask IoU on start and above views (seeds 20-39, 504 px) for several wordings of the box prompt.

    uv run python -m sam3p.prompts
"""

import numpy as np
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode
from valet_sim.ik import IK
from .camera import Camera
from .render import JawMask, SegRenderer, true_masks
from .reach import look_from_above, move
from .seg import Segmenter


def main() -> None:
    cam = Camera(); ik = IK(cam.model); jaw = JawMask()
    seg = Segmenter(resolution=504)
    P = ["white box", "box", "cube", "white cube", "block", "white block"]
    tot = {p: [] for p in P}
    for seed in range(20, 40):
        ep = episode(seed); sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy)); sr = SegRenderer(sim.model)
        for view in ("start", "above"):
            if view == "above":
                move(sim, look_from_above(ik, cam, [ep.box.x, ep.box.y], 0.22))
            sim.step(0.3)
            fr = sim.frame("wrist"); tm = true_masks(sr, sim); jm = jaw(sim.joints()["gripper"])
            r = seg.segment(fr, P)
            row = []
            for p in P:
                best = next((i for i in r[p] if (i.mask & jm).sum() / max(i.area, 1) <= 0.3), None)
                iou = 0.0 if best is None else float((best.mask & tm["box"]).sum() / max((best.mask | tm["box"]).sum(), 1))
                tot[p].append(iou)
                row.append(f"{p}:{iou:.2f}({0 if best is None else best.score:.2f})")
            print(seed, view, " ".join(row), flush=True)
    for p in P:
        v = np.array(tot[p]); print(f"{p:12s} median IoU {np.median(v):.3f}  mean {v.mean():.3f}  <0.8: {(v < 0.8).sum()}/{len(v)}")


if __name__ == "__main__":
    main()
