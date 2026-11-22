"""Main's approach against this one, in main's own sim (`valet.armd.sim`: the same arm, wrist camera, lens and
optics as valet_sim, which this experiment was developed in).

    PYTHONPATH=.:../../src uv run python -m sam3p.versus frames --poses 30        # same frames, both locators
    PYTHONPATH=.:../../src uv run python -m sam3p.versus task --episodes 10       # this policy, main's task
    PYTHONPATH=.:../../src uv run python -m sam3p.versus main --episodes 10 --armd http://127.0.0.1:8771

`frames`: both ways of putting the box on the table, run on the same SAM 3 output for the same wrist frames,
taken from the poses main's `find` looks from (the start pose, then `lookAt` its first answer).
- main: `samAxis` in `brain/skills.ts`. The best "white box" mask (`describe`, score >= 0.5, not cut by the
  edge). Looking steeply down, its centroid on a plane 5 cm up. Otherwise its lowest pixel on the table, plus one
  3.2 cm radius away from the arm. `find` takes two views that agree within 2 cm and averages them.
- this: the jaw-filtered mask with an upright box fitted to its silhouette through the camera model, one view
  (`fit_box`) or two (`fit_box_views`).

`task`: the white box carried 10 cm to the arm's left and set down upright, the goal of main's
`tools/batchbox.sh`. Pose set by seed.
- `task` runs this experiment's policy in-process on armd's app (its Mover and Guard, the code behind
  POST /act). The clock runs as fast as the physics computes.
- `main` runs main's brain as it ships, `valet do --no-claude` (Jev picks every skill), against a separate
  `valet-armd serve --sim` at --armd, which loads its own SAM 3.
Both are graded the same way from the sim's truth.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from valet.armd.app import open_sim  # noqa: E402
from valet.armd.errors import ArmdError  # noqa: E402
from valet.armd.frames import turned  # noqa: E402
from valet.armd.kinematics import GRIPPER, MOTORS, ROLL, Kinematics  # noqa: E402
from valet.armd.locate import Locator  # noqa: E402
from valet.armd.segment import Instance as MainInstance  # noqa: E402
from valet.armd.segment import describe, edge_band  # noqa: E402
from valet.armd.settings import Settings  # noqa: E402
from valet.armd.simscene import BoxSpec  # noqa: E402
from valet_sim.ik import Unreachable  # noqa: E402

from . import policy as pol  # noqa: E402
from .fit import fit_box, fit_box_views  # noqa: E402
from .perceive import BOX_PROMPT, Perception  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "versus"
GOAL = "Pick up the white box, move it about 10 cm to the left, and set it down standing upright."
WHAT = "white box"
MOVE_CM = 10.0
BOX = dict(w=0.06, d=0.06, h=0.09)  # armd sim's own white box
PLACED_WITHIN_M = 0.02
UPRIGHT_WITHIN_M = 0.005  # lying on a side puts the centre 1.5 cm lower
LIFTED_ABOVE_M = 0.01

# brain/skills.ts GRASP and SCAN, and find's lookAt
MID_HEIGHT_M, RADIUS_M, STEEP_COS, AGREE_M = 0.05, 0.032, 0.9, 0.02
MIN_REACH_M, MAX_REACH_M, HOVER_M = 0.14, 0.42, 0.13
SCAN0 = dict(pan_deg=0.0, reach_m=0.16, height_m=0.22, pitch_deg=40.0)


def look_at(pan_deg: float, reach_m: float) -> dict:
    r = reach_m * 100
    if r >= 24:
        return dict(pan_deg=pan_deg, reach_m=(r - 10) / 100, height_m=0.17, pitch_deg=35.0 if r >= 30 else 50.0)
    return dict(pan_deg=pan_deg, reach_m=max(14, r - 4) / 100, height_m=0.16, pitch_deg=60.0)


# --- geometry in main's terms: pan and reach about the pan axis --------------------------------------------------

def polar(kin: Kinematics, xy) -> tuple[float, float]:
    rel = np.asarray(xy, float)[:2] - kin.pan_axis_xy
    return math.degrees(math.atan2(float(rel @ kin.forward(90.0)), float(rel @ kin.forward(0.0)))), float(np.linalg.norm(rel))


def cartesian(kin: Kinematics, pan_deg: float, reach_m: float) -> np.ndarray:
    a = math.radians(pan_deg)
    return kin.pan_axis_xy + reach_m * (math.cos(a) * kin.forward(0.0) + math.sin(a) * kin.forward(90.0))


def sideways(kin: Kinematics, xy, cm: float) -> np.ndarray:
    """`sideways` in brain/skills.ts: `cm` to the arm's left along a chord, at the same reach."""
    pan, r = polar(kin, xy)
    return cartesian(kin, pan - 2 * math.degrees(math.asin(cm / 100 / 2 / r)), r)


def poses(seed: int, n: int, pan=(-25.0, 25.0), reach=(0.18, 0.30), yaw=(-45.0, 45.0)) -> list[BoxSpec]:
    rng, kin = np.random.default_rng(seed), Kinematics()
    out = []
    for _ in range(n):
        xy = cartesian(kin, float(rng.uniform(*pan)), float(rng.uniform(*reach)))
        out.append(BoxSpec(x=float(xy[0]), y=float(xy[1]), yaw_deg=float(rng.uniform(*yaw)), **BOX))
    return out


# --- armd's sim app, in-process ------------------------------------------------------------------------------------

class ManualClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


class World:
    """armd's sim app (`open_sim`) on a clock that only moves when it is slept, so physics runs as fast as it computes."""

    def __init__(self):
        self.clock = ManualClock()
        self.app = open_sim(Settings(allow_motion=True), clock=self.clock, poses_path=Path(tempfile.mkdtemp()) / "poses.json")
        self.app.hold()
        self.top = 0.0

    def reset(self, box: BoxSpec) -> None:
        a = self.app
        a.guard.relax("reset")  # clears the guard's goal; /fake/reset alone leaves it at the last episode's pose
        a.fake.world.reset(box)
        a.guard.hold()
        a.mover.forget_grip()
        self.top = 0.0

    def q(self) -> np.ndarray:
        return self.app.guard.read().q

    def frame(self):
        return turned("wrist", self.app.cameras.capture("wrist").image, None, float(self.q()[ROLL]))

    def box(self) -> tuple[np.ndarray, bool]:
        t = self.app.fake.truth()
        return np.array([t["box"]["x"], t["box"]["y"], t["box"]["z"]]), bool(t["held"])

    def track(self) -> None:
        self.top = max(self.top, float(self.box()[0][2]))

    def to(self, **pose) -> dict:
        return self.app.act({"kind": "to", **pose})


class ArmdSim:
    """What SamPolicy reads from valet_sim's Sim, served by armd's app."""

    def __init__(self, world: World):
        self.w, self.app, self.data = world, world.app, self

    @property
    def time(self) -> float:
        return self.w.clock.now()

    def joints(self) -> dict[str, float]:
        return dict(zip(MOTORS, map(float, self.w.q())))

    def step(self, seconds: float) -> None:
        self.w.clock.sleep(seconds)
        self.w.track()

    def frame(self, cam: str = "wrist"):
        return self.w.frame()


def _move(sim: ArmdSim, goals: dict[str, float], timeout_s: float = 4.0) -> None:
    sim.app.mover.move_joints(np.array([goals[m] for m in MOTORS], float), sim.app.settings.speed_dps)
    sim.step(0.3)


def _set_gripper(sim: ArmdSim, pct: float, settle_s: float = 0.8) -> None:
    sim.app.mover.grip("open", pct)
    sim.step(0.2)


def _close(sim: ArmdSim) -> float:
    """This policy's close (`reach.close`): until the jaw stops moving, then 3 units inside, through armd's guard.
    armd's own grip also stops on 0.9 A of gripper current; in its sim the jaw draws 0.78 A just moving, so that
    trips at the first touch, before a box gripped in the middle of the jaws has been pushed against the fixed one."""
    m, g = sim.app.mover, sim.app.guard
    f = g.follow(m._gripper_path(g.goal.copy(), float(g.read().q[GRIPPER]), 0.0), stall=lambda r, goal: r.q[GRIPPER] - goal[GRIPPER] > 8.0)
    squeeze = g.goal.copy()
    squeeze[GRIPPER] = max(0.0, float((f.stalled or f.final).q[GRIPPER]) - 3.0)
    g.follow([squeeze])
    sim.step(0.3)
    return sim.joints()["gripper"]


@dataclass
class Spot:
    x: float
    y: float
    iou: float = 1.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


class MoveAside(pol.SamPolicy):
    """The SAM 3 policy on main's task: no target square; the spot is 10 cm left of the box's fitted position."""

    def __init__(self, *a, cm: float = MOVE_CM, **kw):
        super().__init__(*a, **kw)
        self.kin, self.cm = Kinematics(), cm

    def look(self, tag, fit=("box",)):
        return super().look(tag, fit=("box",))

    def survey(self) -> bool:
        v = self.look("survey")
        if self.take_box(v, oblique=True):
            self.survey_view = v
        for xy in pol.SURVEY_SPOTS:
            if self.est.box is not None:
                break
            try:
                self.look_above(xy, 0.24)
            except Unreachable:
                continue
            self.take_box(self.look("survey-aside"), oblique=False)
        return self.est.box is not None

    def grasp(self) -> bool:
        ok = super().grasp()
        if ok and self.est.square is None:
            self.picked_from = self.est.box.xy.copy()
            self.est.square = Spot(*sideways(self.kin, self.picked_from, self.cm))
        return ok


def grade(kin: Kinematics, box: BoxSpec, final: np.ndarray, held: bool, top: float) -> dict:
    target = sideways(kin, (box.x, box.y), MOVE_CM)
    off = float(np.hypot(*(final[:2] - target)))
    upright = abs(final[2] - box.h / 2) < UPRIGHT_WITHIN_M
    return {"lifted": top > box.h / 2 + LIFTED_ABOVE_M, "upright": bool(upright), "held": held, "off_cm": round(off * 100, 2),
            "placed": bool(upright and not held and off <= PLACED_WITHIN_M)}


# --- frames: both locators on the same masks -------------------------------------------------------------------------

def main_axis(loc: Locator, q: np.ndarray, found: list, aspect: float, band: np.ndarray) -> tuple[np.ndarray | None, str]:
    """samAxis from brain/skills.ts, on armd's describe() of the same SAM 3 output."""
    d = describe([MainInstance(i.mask, i.score) for i in found], band)
    if not d["found"]:
        return None, "not found"
    if not d["whole"]:
        return None, "cut by the edge"
    c = loc.locate(q, d["centroid"]["u"], d["centroid"]["v"], aspect, MID_HEIGHT_M)
    if c["ray"]["dir"][2] < -STEEP_COS and c["hit"]:
        pan, reach = c["pan_deg"], c["reach_m"]
        how = "centre"
    else:
        f = loc.locate(q, d["bottom"]["u"], d["bottom"]["v"], aspect, 0.0)
        if not f["hit"]:
            return None, "foot misses the table"
        here = loc.kin.pose(q).reach_m
        pan, reach = f["pan_deg"], f["reach_m"] + (1 if f["reach_m"] >= here else -1) * RADIUS_M
        how = "foot"
    if not MIN_REACH_M <= reach <= MAX_REACH_M:
        return None, "out of range"
    return cartesian(loc.kin, pan, reach), how


def frames(args) -> int:
    from .seg import Segmenter

    seg = Segmenter(resolution=args.res)
    p = Perception(seg, fallback="white cube")
    w, kin = World(), Kinematics()
    loc = Locator(kin, 0.0, None)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"frames-seed{args.seed}.jsonl"
    rows = []
    for i, box in enumerate(poses(args.seed, args.poses)):
        w.reset(box)
        if kin.pose(w.q()).xyz[2] < HOVER_M:
            w.to(height_m=0.16)
        truth = np.array([box.x, box.y])
        views, est = [], None
        second = None
        for k in range(2):
            if k == 1:
                tries = ([("lookAt", look_at(*polar(kin, est)))] if est is not None else []) + [("scan", SCAN0)]
                for second, pose in tries:
                    try:
                        if w.to(**pose)["ok"]:
                            break
                    except ArmdError:
                        continue
                else:
                    break
            frame, q = w.frame(), w.q().copy()
            joints = dict(zip(MOTORS, map(float, q)))
            found = seg.segment(frame, [BOX_PROMPT])[BOX_PROMPT]
            band = edge_band(np.ones((frame.height, frame.width), bool))
            m_xy, how = main_axis(loc, q, found, frame.width / frame.height, band)
            v = p.look(frame, joints, fit=())
            hw = None if k == 0 or not views or views[0]["fit"] is None else views[0]["hw"]
            f = None if v.box_inst is None else fit_box(p.cam, joints, v.box_inst.mask, v.ignore, hw)
            views.append({"q": joints, "main": m_xy, "how": how, "fit": f, "hw": None if f is None else (f.side, f.h),
                          "mask": None if v.box_inst is None else v.box_inst.mask, "ignore": v.ignore})
            if k == 0:
                est = m_xy
        err = lambda xy: None if xy is None else round(float(np.hypot(*(np.asarray(xy)[:2] - truth))) * 100, 2)  # noqa: E731
        row = {"pose": i, "box": [round(box.x, 4), round(box.y, 4), round(box.yaw_deg, 1)], "views": len(views), "second": second}
        for k, v in enumerate(views):
            row[f"main_{k}"], row[f"how_{k}"] = err(v["main"]), v["how"]
            row[f"fit_{k}"] = err(None if v["fit"] is None else v["fit"].xy)
        if len(views) == 2:
            a, b = views[0]["main"], views[1]["main"]
            row["main_find"] = err((a + b) / 2) if a is not None and b is not None and np.hypot(*(a - b)) <= AGREE_M else None
            both = [(v["q"], v["mask"], v["ignore"]) for v in views if v["mask"] is not None]
            two = fit_box_views(p.cam, both) if len(both) == 2 else None
            row["fit_two"] = err(None if two is None else two.xy)
        rows.append(row)
        print(json.dumps(row), flush=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {path}")
    return 0


# --- task: this policy in-process, and main's brain over HTTP --------------------------------------------------------

def task(args) -> int:
    from .seg import Segmenter

    pol.move, pol.set_gripper, pol.close = _move, _set_gripper, _close
    seg = Segmenter(resolution=args.res)
    p = Perception(seg, fallback="white cube")
    w, kin = World(), Kinematics()
    cfg = pol.Config(name="versus", two_view=True)
    rows = []
    for i, box in enumerate(poses(args.seed, args.episodes, yaw=(0.0, 0.0))):
        w.reset(box)
        n0, t0, s0 = len(seg.calls), time.time(), w.clock.now()
        pi = MoveAside(ArmdSim(w), p, cfg)
        pi.picked_from = None
        r = pi.run()
        final, held = w.box()
        got = pi.picked_from
        row = {"episode": i, "box": [round(box.x, 4), round(box.y, 4)], **r, "sim_s": round(r["sim_s"] - s0, 1),
               **grade(kin, box, final, held, w.top), "sam_calls": len(seg.calls) - n0, "llm_calls": 0, "wall_s": round(time.time() - t0, 1),
               "grasp_est_err_cm": None if got is None else round(float(np.hypot(got[0] - box.x, got[1] - box.y)) * 100, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write(rows, f"task-sam3-seed{args.seed}")
    return 0


def _http(url: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url + path, None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json"}, method=method)
    return json.load(urllib.request.urlopen(req, timeout=30))


def main_brain(args) -> int:
    if args.armd.rstrip("/").endswith(":8750"):
        raise SystemExit("8750 is the real arm's armd; start `valet-armd serve --sim` on another port")
    kin = Kinematics()
    st = _http(args.armd, "GET", "/state")
    if st.get("kind") != "sim":
        raise SystemExit(f"armd at {args.armd} is {st.get('kind')!r}, not the sim")
    rows = []
    logs = OUT / f"main-logs{args.tag}"
    logs.mkdir(parents=True, exist_ok=True)
    for i, box in enumerate(poses(args.seed, args.episodes, yaw=(0.0, 0.0))):
        _http(args.armd, "POST", "/relax", {})  # the guard's goal goes with it, so the reset pose is where motion starts
        _http(args.armd, "POST", "/fake/reset", {"box": {"x": box.x, "y": box.y, **BOX}})
        top, stop = [0.0], threading.Event()

        def watch():
            while not stop.is_set():
                try:
                    top[0] = max(top[0], _http(args.armd, "GET", "/fake/truth")["box"]["z"])
                except Exception:
                    pass
                time.sleep(0.25)

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        t0 = time.time()
        cmd = ["npm", "run", "--silent", "valet", "--", "do", GOAL, "--what", WHAT, "--steps", str(args.steps), "--no-claude", "--armd", args.armd]
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout).stdout
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            out += "\nTIMEOUT"
        wall = time.time() - t0
        time.sleep(1.0)
        stop.set()
        th.join()
        t = _http(args.armd, "GET", "/fake/truth")
        final = np.array([t["box"]["x"], t["box"]["y"], t["box"]["z"]])
        (logs / f"episode-{i:02d}.txt").write_text(out)
        ends = [ln for ln in out.splitlines() if ln.startswith("run #") and ":" in ln[5:]]
        row = {"episode": i, "box": [round(box.x, 4), round(box.y, 4)], "exit": ends[-1] if ends else out.strip().splitlines()[-1:],
               **grade(kin, box, final, bool(t["held"]), top[0]), "wall_s": round(wall, 1), "steps": out.count("\n  -> "),
               "decider_calls": sum(ln.endswith(" ms]") and not ln.lstrip().startswith("->") for ln in out.splitlines()), "llm_calls": 0,
               "skills": [ln.strip()[3:].split(":")[0] for ln in out.splitlines() if ln.lstrip().startswith("-> ")]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write(rows, f"task-main{args.tag}-seed{args.seed}")
    return 0


def _write(rows: list[dict], name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    print(f"{name}: placed {sum(r['placed'] for r in rows)}/{n}, lifted {sum(r['lifted'] for r in rows)}/{n}, wrote {path}")


def _stats(v: list) -> str:
    v = sorted(x for x in v if x is not None)
    if not v:
        return "-"
    return f"{np.median(v):.2f} / {v[min(len(v) - 1, int(0.9 * len(v)))]:.2f} / {v[-1]:.2f}"


def summary(args) -> int:
    out = []
    fr = [json.loads(ln) for f in sorted(OUT.glob("frames-seed*.jsonl")) for ln in f.read_text().splitlines() if ln.strip()]
    if fr:
        n = len(fr)
        out += [f"Box centre error on the table, cm (median / p90 / max), {n} poses; a miss is a view where the method gave no position.", "",
                "| method | views | positions | error cm |", "|---|---|---|---|"]
        for key, label in (("main_0", "main `samAxis`, start view"), ("main_1", "main `samAxis`, `lookAt` view"),
                           ("main_find", "main `find`: both views, agreeing within 2 cm"), ("fit_0", "mask fit, start view"),
                           ("fit_1", "mask fit, `lookAt` view (size from the start view)"), ("fit_two", "mask fit, both views together")):
            vals = [r.get(key) for r in fr]
            out.append(f"| {label} | {sum(key in r for r in fr)} | {sum(x is not None for x in vals)} | {_stats(vals)} |")
        hows = [r.get(k) for r in fr for k in ("how_0", "how_1") if r.get(k)]
        out += ["", "main's reading per view: " + ", ".join(f"{h} {hows.count(h)}" for h in dict.fromkeys(hows)), ""]
    for name, label in (("task-sam3", "this policy (SAM 3 mask fit, scripted)"), ("task-main", "main's brain as it ships (`valet do --no-claude`)"),
                        ("task-main-grip1.5", "main's brain, armd's gripper stop raised from 0.9 to 1.5 A")):
        rows = [json.loads(ln) for f in sorted(OUT.glob(f"{name}-seed*.jsonl")) for ln in f.read_text().splitlines() if ln.strip()]
        if not rows:
            continue
        n = len(rows)
        med = lambda k: np.median([r[k] for r in rows if r.get(k) is not None]) if any(r.get(k) is not None for r in rows) else float("nan")  # noqa: E731
        placed = [r["off_cm"] for r in rows if r["placed"]]
        out += [f"**{label}**: placed {sum(r['placed'] for r in rows)}/{n}, lifted {sum(r['lifted'] for r in rows)}/{n}, "
                f"set down upright {sum(r['upright'] and not r['held'] for r in rows)}/{n}; off the spot when placed "
                f"{_stats(placed)} cm; per episode (median): wall {med('wall_s'):.0f} s"
                + (f", SAM 3 frames {med('sam_calls'):.0f}" if name == "task-sam3" else "")
                + (f", Jev calls {med('decider_calls'):.0f}, skills run {med('steps'):.0f}" if name.startswith("task-main") else "")
                + ", LLM calls 0", ""]
        out += ["| ep | outcome | lifted | upright | off cm |", "|---|---|---|---|---|"]
        for r in rows:
            oc = r.get("outcome") or (r.get("exit") if isinstance(r.get("exit"), str) else str(r.get("exit")))[:90]
            out.append(f"| {r['episode']} | {oc} | {r['lifted']} | {r['upright']} | {r['off_cm']} |")
        out.append("")
    text = "\n".join(out)
    (OUT / "summary.md").write_text(text + "\n")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sam3p.versus", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    for name in ("frames", "task", "main"):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--res", type=int, default=504)
    sub.choices["frames"].add_argument("--poses", type=int, default=30)
    for name in ("task", "main"):
        sub.choices[name].add_argument("--episodes", type=int, default=10)
    sub.choices["main"].add_argument("--armd", default="http://127.0.0.1:8771")
    sub.choices["main"].add_argument("--steps", type=int, default=30)
    sub.choices["main"].add_argument("--timeout", type=float, default=900.0)
    sub.choices["main"].add_argument("--tag", default="", help="suffix for the result file, e.g. -grip1.5 for a patched armd")
    a = ap.parse_args(argv)
    return {"frames": frames, "task": task, "main": main_brain, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
