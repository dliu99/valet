"""Perception quality on wrist frames against the sim's truth: SAM 3 at several input sizes, the same fits on the
true masks (the fit's own floor), and Jev's box answers. Frames come from non-benchmark seeds at the views the
policy uses. Writes results/perception.jsonl (one line per frame and method) and results/perception.md.

    uv run python -m sam3p.measure --seeds 20-39 --res 336 504 672 1008 --jev
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mujoco
import numpy as np
from valet_sim.ik import IK
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode

from .camera import Camera
from .fit import fit_box, fit_square, pixel_on_plane
from .perceive import BOX_PROMPT, SQUARE_PROMPT, TARGET_SIZE_M, Perception, _dilate
from .reach import look_from_above, move, set_gripper
from .render import JawMask, SegRenderer, true_masks

RESULTS = Path(__file__).resolve().parents[1] / "results"
VIEWS = ("start", "above-box", "pre-grasp", "above-square", "placed")
BOX_MID_Z = 0.0375  # a box centre's height when only a pixel is known: half the middle height


def box_yaw(sim: Sim) -> float:
    r = sim.data.xmat[sim.box_body].reshape(3, 3)
    return float((np.degrees(np.arctan2(r[1, 0], r[0, 0])) + 45) % 90 - 45)


def bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    return None if len(xs) == 0 else (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def mask_iou(a: np.ndarray | None, b: np.ndarray) -> float:
    if a is None:
        return 0.0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def yaw_err(a: float, b: float) -> float:
    return abs((a - b + 45) % 90 - 45)


def frames(seeds: list[int], cam: Camera, ik: IK, rng: np.random.Generator):
    """(seed, view, frame, joints, truth) at each view, with the sim's true masks for grading."""
    for seed in seeds:
        ep = episode(seed)
        sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
        seg = SegRenderer(sim.model)
        sim.step(0.3)
        for view in VIEWS:
            jitter = rng.uniform(-0.015, 0.015, 2)
            if view == "above-box":
                move(sim, look_from_above(ik, cam, np.array([ep.box.x, ep.box.y]) + jitter, 0.22))
            elif view == "pre-grasp":
                move(sim, look_from_above(ik, cam, np.array([ep.box.x, ep.box.y]) + jitter, 0.15))
            elif view == "above-square":
                move(sim, look_from_above(ik, cam, np.array(ep.target_xy) + jitter, 0.22))
            elif view == "placed":  # the box set down near the square's centre, seen from above the square
                off = rng.uniform(-0.02, 0.02, 2)
                yaw = np.radians(rng.uniform(-45, 45))
                a = sim.box_qadr
                with sim.lock:
                    sim.data.qpos[a:a + 7] = [ep.target_xy[0] + off[0], ep.target_xy[1] + off[1], ep.box.h / 2 + 0.002,
                                              np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                    sim.data.qvel[sim.model.jnt_dofadr[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "box")]:][:6] = 0
                    mujoco.mj_forward(sim.model, sim.data)
                move(sim, look_from_above(ik, cam, np.array(ep.target_xy) + jitter, 0.22))
            sim.step(0.3)
            frame = sim.frame("wrist")
            joints = sim.joints()
            masks = true_masks(seg, sim)
            t = sim.truth()["box"]
            yield {"seed": seed, "view": view, "frame": frame, "joints": joints, "masks": masks,
                   "box": {"x": t["x"], "y": t["y"], "yaw": box_yaw(sim), "side": t["w"], "h": t["h"], "tilt": t["tilt_deg"]},
                   "square": list(ep.target_xy)}


def grade_box(rec: dict, fr: dict, fitted, prefix: str = "") -> None:
    b = fr["box"]
    if fitted is None:
        rec[prefix + "box_cm"] = None
        return
    rec[prefix + "box_cm"] = round(100 * float(np.hypot(fitted.x - b["x"], fitted.y - b["y"])), 3)
    rec[prefix + "yaw_deg"] = round(yaw_err(fitted.yaw_deg, b["yaw"]), 2)
    rec[prefix + "h_cm"] = round(100 * (fitted.h - b["h"]), 3)
    rec[prefix + "side_cm"] = round(100 * (fitted.side - b["side"]), 3)


def evaluate(fr: dict, method: str, cam: Camera, jaw_mask: np.ndarray, box_mask, square_mask, box_bb, square_bb,
             hw: tuple[float, float] | None, extra: dict) -> tuple[dict, object]:
    """One method's answer on one frame, graded. Masks may be None (method gives boxes only, or found nothing)."""
    tm, j = fr["masks"], fr["joints"]
    true_box = tm["box"] & ~jaw_mask
    edge = lambda m: bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    rec = {"seed": fr["seed"], "view": fr["view"], "method": method, **extra,
           "box_visible": int(true_box.sum()) if not edge(true_box) else 0,
           "square_visible": int(tm["square"].sum()) if not edge(tm["square"]) else 0}
    rec["box_found"] = box_bb is not None
    rec["square_found"] = square_bb is not None
    rec["box_bbox_iou"] = round(bbox_iou(box_bb, bbox(true_box)), 3)
    rec["square_bbox_iou"] = round(bbox_iou(square_bb, bbox(tm["square"])), 3)
    if box_mask is not None or method.startswith("sam") or method == "truth-mask":
        rec["box_mask_iou"] = round(mask_iou(box_mask, true_box), 3)
        rec["square_mask_iou"] = round(mask_iou(square_mask, tm["square"]), 3)
    # the detector alone: the box's pixel centre on a plane at half a middle box's height; the square's on the table
    for name, bb, z, truth_xy in (("box", box_bb, BOX_MID_Z, (fr["box"]["x"], fr["box"]["y"])), ("square", square_bb, 0.0, fr["square"])):
        p = None if bb is None else pixel_on_plane(cam, j, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), z)
        rec[f"{name}_bbcentre_cm"] = None if p is None else round(100 * float(np.hypot(*(p - truth_xy))), 3)
    fitted = None
    if box_mask is not None:
        fitted = fit_box(cam, j, box_mask, jaw_mask)
        grade_box(rec, fr, fitted)
        if hw is not None and fr["view"] != "start":
            grade_box(rec, fr, fit_box(cam, j, box_mask, jaw_mask, hw), "hw_")
    if square_mask is not None:
        hide = jaw_mask | (_dilate(box_mask, 2) if box_mask is not None else False)
        s = fit_square(cam, j, square_mask & ~hide, TARGET_SIZE_M, hide)
        rec["square_cm"] = None if s is None else round(100 * float(np.hypot(*(s.xy - fr["square"]))), 3)
    return rec, fitted


def summarize(rows: list[dict]) -> str:
    def med(vals):
        v = [x for x in vals if x is not None]
        return "-" if not v else f"{statistics.median(v):.2f}"

    def p90(vals):
        v = sorted(x for x in vals if x is not None)
        return "-" if not v else f"{v[min(len(v) - 1, int(0.9 * len(v)))]:.2f}"

    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = ["| method | view | n | box found | box mask IoU | box bbox IoU | box centre cm (fit) med / p90 | box centre cm, bbox centre | "
           "box centre cm, fit w/ start-view size | yaw err deg | height err cm (abs) | square found | square mask IoU | square cm (fit) med / p90 | square cm, bbox centre | s/frame |",
           "|" + "---|" * 16]
    for m in methods:
        for view in ("all", *VIEWS):
            rs = [r for r in rows if r["method"] == m and (view == "all" or r["view"] == view)]
            vis = [r for r in rs if r["box_visible"] > 200]
            sq = [r for r in rs if r["square_visible"] > 200]
            if not rs:
                continue
            found = f"{sum(r['box_found'] for r in vis)}/{len(vis)}"
            sqfound = f"{sum(r['square_found'] for r in sq)}/{len(sq)}"
            hs = [abs(r["h_cm"]) for r in vis if r.get("h_cm") is not None and r["view"] == "start"]
            out.append(" | ".join([
                f"| {m}", view, str(len(rs)), found, med(r.get("box_mask_iou") for r in vis), med(r["box_bbox_iou"] for r in vis),
                f"{med(r.get('box_cm') for r in vis)} / {p90(r.get('box_cm') for r in vis)}", med(r["box_bbcentre_cm"] for r in vis),
                med(r.get("hw_box_cm") for r in vis), med(r.get("yaw_deg") for r in vis), med(hs) if view in ("all", "start") else "-",
                sqfound, med(r.get("square_mask_iou") for r in sq), f"{med(r.get('square_cm') for r in sq)} / {p90(r.get('square_cm') for r in sq)}",
                med(r["square_bbcentre_cm"] for r in sq), med(r.get("s") for r in rs)]) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sam3p.measure", description=__doc__)
    p.add_argument("--seeds", default="20-39")
    p.add_argument("--res", type=int, nargs="*", default=[336, 504, 672, 1008])
    p.add_argument("--jev", action="store_true")
    p.add_argument("--sam", default="3", choices=["3", "3.1"], help="SAM version")
    p.add_argument("--no-truth", action="store_true", help="skip the fits on true masks (already measured)")
    p.add_argument("--out", default=str(RESULTS / "perception.jsonl"))
    p.add_argument("--overlays", default=None, help="directory for a few mask overlays")
    a = p.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    cam = Camera()
    ik = IK(cam.model)
    jaw = JawMask()
    rng = np.random.default_rng(0)
    segmenter = None
    if a.res:
        from .seg import Segmenter

        segmenter = Segmenter(resolution=a.res[0], version=a.sam)
        perception = Perception(segmenter, cam, jaw)
    jev = None
    if a.jev:
        from .jev import Jev

        jev = Jev()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    hw_by = {}
    with out.open("w") as f:
        for fr in frames(seeds, cam, ik, rng):
            jm = _dilate(jaw(fr["joints"]["gripper"]), 3)
            name = "sam3" if a.sam == "3" else "sam3.1"
            todo = ([] if a.no_truth else [("truth-mask", None)]) + [(f"{name}-{r}", r) for r in a.res] + ([("jev", None)] if jev else [])
            for method, res in todo:
                key = (fr["seed"], method)
                extra = {}
                if method == "truth-mask":
                    tm = fr["masks"]
                    bm, sm = tm["box"] & ~jm, tm["square"]
                    rec, fitted = evaluate(fr, method, cam, jm, bm if bm.sum() else None, sm if sm.sum() else None,
                                           bbox(bm), bbox(sm), hw_by.get(key), extra)
                elif method == "jev":
                    t0 = time.time()
                    ans = jev.boxes(fr["frame"], {"box": "the white box", "square": "the blue square on the table"})
                    extra = {"s": round(time.time() - t0, 3), "box_conf": ans["box"].confidence, "square_conf": ans["square"].confidence}
                    bb = lambda b: (b.left, b.top, b.right, b.bottom)
                    rec, fitted = evaluate(fr, method, cam, jm, None, None, bb(ans["box"]), bb(ans["square"]), None, extra)
                else:
                    segmenter.processor.image_size = res
                    t0 = time.time()
                    bi, si, _, _ = perception.pick(fr["frame"], fr["joints"])
                    extra = {"s": round(time.time() - t0, 3), "box_score": None if bi is None else round(bi.score, 3),
                             "square_score": None if si is None else round(si.score, 3)}
                    rec, fitted = evaluate(fr, method, cam, jm, None if bi is None else bi.mask, None if si is None else si.mask,
                                           None if bi is None else bbox(bi.mask), None if si is None else bbox(si.mask), hw_by.get(key), extra)
                    if a.overlays and fr["seed"] < seeds[0] + 3 and res == a.res[0]:
                        _overlay(fr, bi, si, Path(a.overlays) / f"s{fr['seed']}-{fr['view']}-{method}.png")
                if fr["view"] == "start" and fitted is not None:
                    hw_by[key] = (fitted.side, fitted.h)
                rows.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(fr["seed"], fr["view"], {r["method"]: (r.get("box_cm"), r.get("square_cm")) for r in rows[-len(todo):]}, flush=True)
    table = summarize(rows)
    (out.parent / out.name.replace(".jsonl", ".md")).write_text(table + "\n")
    print(table)
    if jev:
        print("jev usd", round(sum(c["usd"] for c in jev.calls), 4), "calls", len(jev.calls))
    return 0


def _overlay(fr: dict, bi, si, path: Path) -> None:
    from PIL import Image

    a = np.asarray(fr["frame"]).copy()
    for inst, colour in ((bi, (255, 60, 60)), (si, (60, 255, 60))):
        if inst is not None:
            a[inst.mask] = (a[inst.mask] * 0.45 + np.array(colour) * 0.55).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
