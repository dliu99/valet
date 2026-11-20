"""One benchmark episode: measure the scene, let Claude plan, run the plan with Jev's checks, replan on failure,
then grade. The policy reads only joint readings and wrist frames; the grade reads the simulator's truth after the
policy has finished."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from valet_sim.ik import Unreachable
from valet_sim.sim import START_JOINTS, Sim, SimSpec, with_box
from valet_sim.task import PAN_AXIS_X, episode, score

from .body import Body
from .jev import Jev, LocalSystemOne
from .planner import Planner
from .see import Estimate, Eyes

MAX_REPLANS = 2
MAX_STEPS = 40
FINAL_QUESTIONS = {
    "on_square": {"type": "noul", "instructions": "Is the white box standing on the blue square in `wrist`?"},
    "upright": {"type": "noul", "instructions": "Is the white box standing upright on its base in `wrist`, not tipped over?"},
}


@dataclass
class Config:
    effort: str = "medium"
    model: str = "claude-opus-5-5"
    fast: str = "jev"  # the per-frame model: "jev" (TypeSafe API) or "h2o" (H2O-Lightning-4B, local)
    lessons: Path | None = None  # a file of lessons carried between episodes; None: no lessons
    keep_lessons: int = 8


class Recorder:
    def __init__(self, out: Path):
        self.out = out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        self.events = (out / "events.jsonl").open("w")
        self.n = 0
        self.t0 = time.monotonic()

    def event(self, kind: str, **data) -> None:
        self.events.write(json.dumps({"t": round(time.monotonic() - self.t0, 2), "kind": kind, **data}, default=_plain) + "\n")
        self.events.flush()

    def frame(self, name: str, im: Image.Image) -> str:
        self.n += 1
        path = self.out / "frames" / f"{self.n:02d}-{name}.jpg"
        im.convert("RGB").save(path, quality=85)
        return path.name

    def close(self) -> None:
        self.events.close()


def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _cm(v) -> list[float]:
    return [round(float(x) * 100, 1) for x in v]


def describe(name: str, est: Estimate) -> str:
    x, y = est.xyz[:2]
    r = np.hypot(x - PAN_AXIS_X, y)
    bearing = np.degrees(np.arctan2(y, x - PAN_AXIS_X))
    if name == "box":
        return (f"box: footprint centre ({x*100:.1f}, {y*100:.1f}), bearing {bearing:.0f}, {r*100:.1f} from the base axis; "
                f"side about {est.size*100:.1f}, height about {est.height*100:.1f}")
    return f"square: centre ({x*100:.1f}, {y*100:.1f}), bearing {bearing:.0f}, {r*100:.1f} from the base axis; side about {est.size*100:.1f}"


class Run:
    def __init__(self, seed: int, cfg: Config, out: Path, jev=None, planner=None):
        self.seed, self.cfg = seed, cfg
        ep = episode(seed)
        self.sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
        self.sim.step(0.3)
        self.body = Body(self.sim)
        self.jev = jev or (LocalSystemOne() if cfg.fast == "h2o" else Jev())
        self.eyes = Eyes(self.jev)
        self.planner = planner or Planner(cfg.effort, cfg.model)
        self.rec = Recorder(out)
        self.anchor: dict[str, np.ndarray] = {"table": np.zeros(3)}
        self.est: dict[str, Estimate] = {}
        self.log: list[dict] = []

    # --- perception ---------------------------------------------------------------------------------------------
    def measure(self) -> bool:
        f, v = self.body.frame()
        self.eyes.look(f, v, kind="start")
        self.rec.frame("start", f)
        square_guess = self.eyes._cast("square", 0.0)  # from the overview, before the box looks take over
        box = self.eyes.locate_box(self.body)
        square = self.eyes.locate_square(self.body, guess=square_guess)
        self.first_box_height = box.height if box is not None else 0.075
        for name, est in (("box", box), ("square", square)):
            if est is not None:
                self.est[name], self.anchor[name] = est, est.xyz.copy()
                self.rec.event("estimate", what=name, **est.summary())
        return box is not None and square is not None

    # --- execution ----------------------------------------------------------------------------------------------
    def run_plan(self, plan: dict) -> tuple[bool, str]:
        for i, step in enumerate(plan["steps"][:MAX_STEPS]):
            ok, result = self.do(step)
            entry = {"step": i, "do": step["do"], "why": step["why"], **result}
            self.log.append(entry)
            self.rec.event("step", plan=len(self.planner.meter.log), spec=step, ok=ok, **result)
            if not ok:
                return False, f"step {i} ({step['do']}: {step['why']}) failed: {result.get('failure', '')}"
        return True, ""

    def do(self, s: dict) -> tuple[bool, dict]:
        kind = s["do"]
        if kind == "move":
            if s["G"] is None or len(s["G"]) != 3:
                return False, {"failure": "move needs G = [x, y, z]"}
            rel = s["rel"] or "table"
            target = self.anchor.get(rel, np.zeros(3)) + np.asarray(s["G"], float) / 100
            now = self.body.pose()
            pitch = now["pitch_deg"] if s["pitch"] is None else float(s["pitch"])
            roll = now["roll_deg"] if s["roll"] is None else float(s["roll"])
            try:
                used = self.body.move_path(target, pitch, roll, s["gripper"], pitch_slack=15)
            except Unreachable as e:
                return False, {"failure": str(e), "target_cm": _cm(target)}
            reached = np.asarray(self.body.ik.pose(self.body.joints())[0])
            off = float(np.linalg.norm(reached - target))
            out = {"target_cm": _cm(target), "pitch_used": round(used, 0), "G_now_cm": _cm(reached),
                   "gripper": round(self.body.joints()["gripper"], 1)}
            if off > 0.008:
                out["note"] = f"G stopped {off*100:.1f} cm from the target (something in the way?)"
            return True, out
        if kind == "grip":
            if s["gripper"] is None:
                return False, {"failure": "grip needs gripper"}
            reading = self.body.grip(float(s["gripper"]))
            out = {"reading": round(reading, 1)}
            if s["expect"] and len(s["expect"]) == 2 and not (s["expect"][0] <= reading <= s["expect"][1]):
                out["failure"] = f"gripper reads {reading:.1f}, expected {s['expect'][0]}-{s['expect'][1]}"
                return False, out
            return True, out
        if kind == "check":
            if not s["question"]:
                return False, {"failure": "check needs a question"}
            f, _ = self.body.frame()
            name = self.rec.frame("check", f)
            p = self.jev.ask({"wrist": f}, {"q": {"type": "noul", "instructions": s["question"]}})["q"]["p"]
            want_yes = (s["want"] or "yes") == "yes"
            ok = (p >= 0.5) == want_yes
            out = {"question": s["question"], "p_yes": p, "frame": name}
            if not ok:
                out["failure"] = f"Jev answered {'yes' if p >= 0.5 else 'no'} (p_yes {p:.2f}), wanted {s['want']}"
            return ok, out
        if kind == "look":
            rel = s["rel"] if s["rel"] in ("box", "square") else "box"
            guess = self.anchor.get(rel)
            est = self.eyes.locate_box(self.body, guess=None if guess is None else guess + [0, 0, 0.075]) if rel == "box" \
                else self.eyes.locate_square(self.body, guess=guess)
            if est is None:
                return False, {"failure": f"could not find the {rel}"}
            self.est[rel], self.anchor[rel] = est, est.xyz.copy()
            return True, {"estimate": describe(rel, est)}
        return False, {"failure": f"unknown step {kind}"}

    # --- the episode --------------------------------------------------------------------------------------------
    def brief(self) -> str:
        lines = ["Measured:", describe("box", self.est["box"]), describe("square", self.est["square"]),
                 f"Arm now: {json.dumps(self.body.pose())}"]
        lessons = self.lessons()
        if lessons:
            lines += ["", "Lessons you wrote after earlier episodes (each from one episode; weigh them):"] + [f"- {l}" for l in lessons]
        lines += ["", "Write the plan."]
        return "\n".join(lines)

    def images(self) -> list[tuple[str, Image.Image]]:
        out = []
        for kind, caption in (("box_over", "Wrist camera looking straight down at the box (image up is away from the base, image right is the arm's right):"),
                              ("box_across", "Wrist camera looking at the box from the base side:"),
                              ("square_over", "Wrist camera looking straight down at the square:")):
            s = self.eyes.last(kind)
            if s is not None and s.frame is not None:
                out.append((caption, s.frame))
        return out

    def lessons(self) -> list[str]:
        p = self.cfg.lessons
        if not p or not p.exists():
            return []
        return [l[2:].strip() for l in p.read_text().splitlines() if l.startswith("- ")][-self.cfg.keep_lessons:]

    def assess(self) -> dict:
        """The policy's own verdict, measured the same way as the scene: lift the hand clear, go back to the home
        view, find the box and the square again, and compare their centres. Jev's yes/no answers are kept as notes."""
        holding = self.body.joints()["gripper"] > self.body.gripper_goal() + 3
        now = np.asarray(self.body.ik.pose(self.body.joints())[0])
        try:
            self.body.move_path(now + [0, 0, 0.06], self.body.pose()["pitch_deg"], self.body.pose()["roll_deg"], None, pitch_slack=40)
        except Unreachable:
            pass
        self.body.go({**START_JOINTS, "gripper": self.body.gripper_goal()})
        f, v = self.body.frame()
        a = self.eyes.look(f, v, extra=FINAL_QUESTIONS, kind="final_home")
        first_h = getattr(self, "first_box_height", 0.075)
        square = self.eyes.locate_square(self.body, guess=self.anchor.get("square", self.eyes._cast("square", 0.0)))
        box = self.eyes.locate_box(self.body)
        verdict: dict = {"holding": holding, "jev_on_square": a["on_square"]["p"], "jev_upright": a["upright"]["p"]}
        if box is not None and square is not None:
            off = float(np.linalg.norm(box.xyz[:2] - square.xyz[:2]))
            verdict.update(offset_cm=round(off * 100, 1), height_cm=round(box.height * 100, 1), first_height_cm=round(first_h * 100, 1))
            verdict["tipped"] = box.height < 0.6 * first_h  # a box on its side looks much lower than it stood
            verdict["placed_self"] = (not holding) and off < 0.03 and not verdict["tipped"]
        else:
            verdict["placed_self"] = False
        s = self.eyes.last("box_over")
        self.final_frame = s.frame if s is not None and s.frame is not None else f
        verdict["frame"] = self.rec.frame("final", self.final_frame)
        return verdict

    def run(self) -> dict:
        t0 = time.monotonic()
        result: dict = {"seed": self.seed}
        try:
            if not self.measure():
                result["note"] = "could not find the box or the square"
                plan_ok, why = False, result["note"]
            else:
                plan = self.planner.plan(self.brief(), self.images())
                self.rec.event("plan", **plan)
                plan_ok, why = self.run_plan(plan)
                replans = 0
                while not plan_ok and replans < MAX_REPLANS:
                    replans += 1
                    f, _ = self.body.frame()
                    report = (f"The plan stopped: {why}\nLog of this episode so far:\n" + "\n".join(json.dumps(e) for e in self.log)
                              + f"\nEstimates now: {describe('box', self.est['box'])}; {describe('square', self.est['square'])}"
                              + f"\nArm now: {json.dumps(self.body.pose())}\nWrite a new plan from here.")
                    plan = self.planner.replan(report, [("Wrist camera now:", f)])
                    self.rec.event("replan", why=why, **plan)
                    plan_ok, why = self.run_plan(plan)
                result["replans"] = replans
            result["plan_completed"] = plan_ok
            result["self_check"] = self.assess()
            result.update(score(self.sim.truth()))
            if self.cfg.lessons is not None and "box" in self.est:
                report = (f"The episode is over. The plan {'ran to the end' if plan_ok else 'stopped: ' + why}.\n"
                          f"Log:\n" + "\n".join(json.dumps(e) for e in self.log)
                          + f"\nAfterwards the box and the square were measured again: {json.dumps({k: v for k, v in result['self_check'].items() if k != 'frame'})} "
                            "(offset_cm: box centre to square centre; the box counts as placed within about 3 cm, upright, released).\n"
                            "Write one or two sentences of advice for planning the next episode (a different box and square): "
                            "what to keep doing or change. Be concrete (numbers).")
                lesson = self.planner.lesson(report, [("Last look straight down at the box:", self.final_frame)])
                result["lesson"] = lesson
                with self.cfg.lessons.open("a") as fh:
                    fh.write(f"- {lesson}\n")
        except Exception as e:  # a crash is a failed episode, recorded
            import traceback
            result["error"] = f"{type(e).__name__}: {e}"
            self.rec.event("error", trace=traceback.format_exc())
            result.update(score(self.sim.truth()))
        result["seconds"] = round(time.monotonic() - t0, 1)
        result["sim_seconds"] = round(float(self.sim.data.time), 1)
        result["fast_model"] = self.cfg.fast
        result["model"] = self.cfg.model
        result["jev"] = {"calls": self.jev.meter.calls, "usd": round(self.jev.meter.usd, 5), "tokens": self.jev.meter.input_tokens}
        m = self.planner.meter
        result["claude"] = {"calls": m.calls, "usd": round(m.usd, 4), "in": m.input_tokens, "out": m.output_tokens,
                            "cache_read": m.cache_read, "s": round(m.seconds, 1)}
        result["truth_box"] = self.sim.truth()["box"]
        self.rec.event("result", **result)
        self.rec.event("jev_log", log=self.jev.meter.log)
        self.rec.event("claude_log", log=m.log)
        self.rec.close()
        (self.rec.out / "result.json").write_text(json.dumps(result, indent=2, default=_plain))
        return result
