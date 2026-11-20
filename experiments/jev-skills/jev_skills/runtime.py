"""The runtime loop. Each decision, Jev sees the wrist frame and the arm's state and picks the next skill (its
parameter folded into the option). After a skill runs, Jev judges that skill's done criterion on the new frame.
Claude only writes the goal at the start and re-plans after two failures in a row, within the slow-call budget.

Grading reads `sim.truth()` only after a decision is made, to label whether Jev's choice and judgement were right;
nothing the policy sees comes from it."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode, score

from . import skills as S
from .arm import Arm
from .models import Claude, Ledger, SystemOne

TASK = "Pick up the white box and set it down upright on the blue square, then let go of it."
SEE = ("`wrist` is the camera on the arm's jaws. `goal` is the task. `arm` is what the arm's own sensors say. "
       "`memory` is where earlier looks found things. `history` lists the skills run so far, oldest first, with what "
       "each reported and whether its done test held.")
YES, NO = 0.7, 0.3


@dataclass
class Config:
    name: str = "base"
    decider: str = "jev"
    claude: bool = True
    max_decisions: int = 14
    max_slow: int = 5
    max_sim_s: float = 150.0
    judge_sees_report: bool = True
    final_unsure_uses_code: bool = True
    skills: tuple[str, ...] = S.DEFAULT_SKILLS
    params: S.Params = field(default_factory=S.Params)


GOAL_SYSTEM = ("You set the goal for a robot arm with one camera on its jaws. A fast vision model picks one skill at a "
               "time from a fixed menu and judges when each is done; it reads your goal sentence on every decision. "
               "Reply with one or two short sentences that state the goal and what finished looks like.")
GOAL_SCHEMA = {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"], "additionalProperties": False}
REPLAN_SYSTEM = ("You supervise a robot arm. A fast vision model picks skills from a menu; the last two skills failed. "
                 "Read the history and pick the skill that should run next, with one sentence of advice the fast model "
                 "will read on its next decisions.")


def replan_schema(keys: list[str]) -> dict:
    return {"type": "object", "properties": {"next": {"type": "string", "enum": keys}, "advice": {"type": "string"}},
            "required": ["next", "advice"], "additionalProperties": False}


class Episode:
    def __init__(self, seed: int, cfg: Config, out: Path, jev=None, claude=None):
        self.seed, self.cfg, self.out = seed, cfg, out
        out.mkdir(parents=True, exist_ok=True)
        self.events = (out / "events.jsonl").open("w")
        ep = episode(seed)
        self.sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
        self.arm = Arm(self.sim)
        self.ledger = Ledger()
        self.jev = jev or SystemOne(self.ledger, cfg.decider)
        if jev is not None:
            jev.ledger = self.ledger
        self.claude = claude if claude is not None else (Claude(self.ledger) if cfg.claude else None)
        if claude is not None:
            claude.ledger = self.ledger
        self.mem = S.Memory()
        self.ctx = S.Ctx(self.arm, self.jev, self.mem, cfg.params, self.log)
        self.history: list[str] = []
        self.advice = ""
        self.goal = TASK
        self.menu = {k: S.SKILLS[k] for k in cfg.skills}

    def log(self, kind: str, data: dict) -> None:
        self.events.write(json.dumps({"t": round(self.arm.t(), 2), "kind": kind, **data}, default=_jsonable) + "\n")
        self.events.flush()

    # ---- what Jev sees -------------------------------------------------------------------------------------------

    def state(self, extra: dict | None = None) -> dict:
        gp, pitch = self.arm.pose()
        over = "neither"
        for name, xy in (("the white box", self.mem.box_xy), ("the blue square", self.mem.square_xy)):
            if xy is not None and np.linalg.norm(gp[:2] - xy) < 0.03:
                over = name
        st = {"goal": self.goal,
              "arm": {"jaws": S.jaws_state(self.arm), "jaw_height_cm": round(100 * float(gp[2]), 1), "jaws_over": over},
              "memory": self.mem.facts(),
              "history": self.history[-6:] or ["nothing has run yet"]}
        if self.advice:
            st["advice"] = self.advice
        return {**st, **(extra or {})}

    def choose(self, frame: np.ndarray) -> tuple[str, dict]:
        options = {k: s.use_when for k, s in self.menu.items()}
        options[S.DONE_KEY] = S.DONE_USE
        q = {"skill:next": {"type": "choice", "instructions": f"{SEE} Which skill should run next to get closer to `goal`?",
                            "criteria": options}}
        a = self.jev.ask(self.state(), frame, q, "choose")["skill:next"]
        return a["choice"], a

    def judge(self, frame: np.ndarray, key: str, criterion: str, report: S.Report | None) -> float:
        extra = {"report": report.text} if (report is not None and self.cfg.judge_sees_report) else {}
        q = {f"done:{key}": {"type": "noul", "instructions": f"{SEE} The skill `{key}` just ran. Is this true now: {criterion}"}}
        a = self.jev.ask(self.state(extra), frame, q, "judge")[f"done:{key}"]
        return float(a["noul"])

    # ---- the loop ------------------------------------------------------------------------------------------------

    def run(self) -> dict:
        t_wall = time.monotonic()
        frame, _ = self.arm.frame()
        if self.claude is not None and self.ledger.slow_calls < self.cfg.max_slow:
            self.goal = self.claude.ask(GOAL_SYSTEM, f"The task: {TASK}\nThe image is the wrist camera now.", frame,
                                        GOAL_SCHEMA, "goal")["goal"]
            self.log("goal", {"goal": self.goal})
        fails, done_rejected, finished = 0, 0, False
        last: tuple[str, S.Report] | None = None
        for step in range(self.cfg.max_decisions):
            if self.arm.t() > self.cfg.max_sim_s:
                break
            frame, _ = self.arm.frame()
            Image.fromarray(frame).save(self.out / f"step{step:02d}.jpg", quality=70)
            if last is not None:
                key, rep = last
                p = self.judge(frame, key, self.menu[key].done, rep)
                verdict = "yes" if p >= YES else "no" if p <= NO else "unsure"
                label = self.label_done(key)
                self.log("judged", {"step": step, "skill": key, "p": round(p, 3), "verdict": verdict, "code_ok": rep.ok,
                                    "truth": label})
                good = verdict == "yes" if verdict != "unsure" else rep.ok
                self.history.append(f"{key}: {rep.text}. Done test: {'held' if good else 'NOT held'}"
                                    + (" (judge unsure; went by the skill's own check)" if verdict == "unsure" else ""))
                fails = 0 if good else fails + 1
            choice, a = self.choose(frame)
            right = self.label_choice()
            self.log("chose", {"step": step, "choice": choice, "p": round(float(a.get("confidence") or 0), 3),
                               "probs": {k: round(v, 3) for k, v in (a.get("probabilities") or {}).items()},
                               "right": sorted(right) if right is not None else None})
            if fails >= 2 and self.claude is not None and self.ledger.slow_calls < self.cfg.max_slow:
                keys = [*self.menu, S.DONE_KEY]
                r = self.claude.ask(REPLAN_SYSTEM, json.dumps({"goal": self.goal, "state": self.state(),
                                                              "menu": {k: s.use_when for k, s in self.menu.items()}}),
                                    frame, replan_schema(keys), "replan")
                self.log("replan", {"step": step, "jev_choice": choice, **r})
                choice, self.advice, fails = r["next"], r["advice"], 0
            if choice == S.DONE_KEY:
                p = self.judge(frame, "done", S.FINAL_CRITERION, None)
                placed = score(self.sim.truth())["placed"]
                code_ok = self.mem.placed_check.startswith("released on the blue square")
                self.log("final_check", {"step": step, "p": round(p, 3), "truth": placed, "code_ok": code_ok})
                if p >= YES or (self.cfg.final_unsure_uses_code and p > NO and code_ok):
                    finished = True
                    break
                done_rejected += 1
                self.history.append(f"done: rejected, because this is not true yet: {S.FINAL_CRITERION}")
                last = None
                if done_rejected >= 2:
                    break
                continue
            if choice not in self.menu:
                self.history.append(f"{choice}: not a skill")
                last = None
                continue
            rep = self.menu[choice].run(self.ctx)
            self.log("ran", {"step": step, "skill": choice, "ok": rep.ok, "report": rep.text, **rep.data})
            last = (choice, rep)
        self.arm.wait(1.0)
        result = score(self.sim.truth())
        led = self.ledger
        summary = {"seed": self.seed, "config": self.cfg.name, **result, "finished": finished,
                   "sim_s": round(self.arm.t(), 1), "fast_s": round(led.fast_s, 1), "slow_s": round(led.slow_s, 1),
                   "seconds": round(self.arm.t() + led.fast_s + led.slow_s, 1), "wall_s": round(time.monotonic() - t_wall, 1),
                   "fast_calls": led.fast_calls, "slow_calls": led.slow_calls,
                   "usd": round(led.fast_usd + led.slow_usd, 5), "calls": led.by_use}
        self.log("summary", summary)
        self.events.close()
        return summary

    # ---- grading labels (truth; never shown to the policy) --------------------------------------------------------

    def label_choice(self) -> set[str] | None:
        """The skills a sensible operator would accept now; None when the box is lying tipped (nothing helps)."""
        tr = self.sim.truth()
        box, m = tr["box"], self.mem
        sc = score(tr)
        if sc["placed"]:
            return {S.DONE_KEY}
        on_target = sc["off_target_m"] <= 0.03 and sc["resting"]
        if on_target and tr["touching_jaws"] and box["tilt_deg"] <= 15:
            return {"back_off"}
        if box["tilt_deg"] > 15 and not tr["held"]:
            return None
        if tr["held"]:
            if m.square_xy is None:
                return {"look_square"}
            gp, _ = self.arm.pose()
            if m.held and np.linalg.norm(gp[:2] - S._drop_point(m)) < 0.03:
                return {"place"}
            return {"carry"}
        if tr["touching_jaws"]:
            return {"back_off", "look_box"}
        ok = set()
        if m.square_xy is None:
            ok |= {"look_workspace", "look_square"}
        true_xy = np.array([box["x"], box["y"]])
        measured = m.box_xy is not None and m.box_h is not None and m.box_rect is not None
        if not measured or np.linalg.norm(m.box_xy - true_xy) > 0.02:
            ok |= {"look_box"} | ({"look_workspace"} if m.box_xy is None else set())
        else:
            ok |= {"grasp_near_top"}
        return ok

    def label_done(self, key: str) -> bool:
        tr = self.sim.truth()
        box = tr["box"]
        view = self.arm.view(self.arm.joints())
        w, h = view.width, view.height

        def in_view(xyz) -> bool:
            p = view.project(xyz)
            return p is not None and 0 <= p[0] < w and 0 <= p[1] < h

        corners = [[tr["target"][0] + dx, tr["target"][1] + dy, 0.0] for dx in (-0.035, 0.035) for dy in (-0.035, 0.035)]
        top = [box["x"], box["y"], box["z"] + box["h"] / 2]
        gp, _ = self.arm.pose()
        if key == "look_workspace":
            return in_view(top) and in_view([*tr["target"], 0.0])
        if key == "look_box":
            return all(in_view([top[0] + dx, top[1] + dy, top[2]]) for dx in (-0.025, 0.025) for dy in (-0.025, 0.025)) \
                and self.arm.pose()[1] >= 50
        if key == "look_square":
            return all(in_view(c) for c in corners)
        if key.startswith("grasp"):
            return tr["held"] and box["z"] - box["h"] / 2 > 0.015
        if key == "carry":
            return tr["held"] and float(np.hypot(box["x"] - tr["target"][0], box["y"] - tr["target"][1])) < 0.03
        if key in ("place", "done"):
            return score(tr)["placed"]
        if key == "back_off":
            return not tr["touching_jaws"] and self.arm.gripper() > 60
        return False


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.round(4).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def config_dict(cfg: Config) -> dict:
    return {**asdict(cfg), "skills": list(cfg.skills)}
