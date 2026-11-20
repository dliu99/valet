"""Jev runs the arm. Each tick Jev sees the upright wrist frame, the joints and a one-sentence goal, and picks the
next move and how far to go. Claude only writes the goal sentence: at the start, when Jev says it is met, and when
progress stalls. Simulated time; the arm holds still while a model answers, as the real arm would."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import mujoco
import numpy as np

from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS, WRIST_CAMERA
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode, score

HERE = Path(__file__).resolve().parent
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_USD_PER_TOKEN = 0.042 / 1e6
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_USD_PER_TOKEN = (4.0 / 1e6, 20.0 / 1e6)

TASKS = {
    "plain": "Pick up the white box and set it down on the blue square, then let go of it.",
    "upright": "Pick up the white box and set it down upright on the blue square, then let go of it.",
}
TASK = TASKS["plain"]

SEE = "`wrist` is the camera on the arm's jaws; the red dot in `wrist` is on the table straight below the jaws."
MOVES = {
    "to_box": "Move the jaws across the table toward the white box.",
    "to_square": "Move the jaws across the table toward the blue square.",
    "lower": "Lower the jaws toward the table.",
    "raise": "Raise the jaws away from the table.",
    "turn_cw": "Turn the jaws so the scene in `wrist` rotates clockwise.",
    "turn_ccw": "Turn the jaws so the scene in `wrist` rotates counterclockwise.",
    "open": "Open the jaws.",
    "close": "Close the jaws.",
    "done": "Nothing: the goal is met.",
}
WHERE = {
    "ahead": "Above the red dot, toward the top edge of `wrist`.",
    "back": "Below the red dot, toward the bottom edge of `wrist`.",
    "left": "Left of the red dot.",
    "right": "Right of the red dot.",
    "here": "Under the red dot.",
}
CM = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CM_LEVELS = [
    "Under 1 cm: right there.",
    "About 1 cm: a nudge.",
    "About 2 cm: half a box width.",
    "About 4 cm: one box width.",
    "About 8 cm: two box widths.",
    "16 cm or more: a large part of the view.",
]
DEG = (0.0, 5.0, 10.0, 20.0, 40.0)
DEG_LEVELS = [
    "Already square: the jaws line up with the sides of what they will grip.",
    "About 5 degrees off square.",
    "About 10 degrees off square.",
    "About 20 degrees off square.",
    "About 40 degrees off square.",
]
OBJECTS = {"box": "the white box", "square": "the blue square"}


SEE_SIDE = "`side` is a fixed camera beside the table; the red dot in `side` is on the table straight below the jaws."
WHERE_SIDE = {k: v.replace("`wrist`", "`side`") for k, v in WHERE.items()}


def questions_for(side: bool = False) -> dict:
    qs = {"move:next": {"type": "choice", "instructions": f"{SEE} Which move brings the arm closer to `goal`?", "criteria": MOVES}}
    for key, name in OBJECTS.items():
        qs[f"where:{key}"] = {"type": "choice", "instructions": f"{SEE} Where is {name} relative to the red dot?", "criteria": WHERE}
        qs[f"far:{key}"] = {"type": "score", "instructions": f"{SEE} How far across the table is {name} from the red dot?", "criteria": CM_LEVELS}
        if side:
            qs[f"where_side:{key}"] = {"type": "choice", "instructions": f"{SEE_SIDE} In `side`, where is {name} relative to the red dot?", "criteria": WHERE_SIDE}
            qs[f"far_side:{key}"] = {"type": "score", "instructions": f"{SEE_SIDE} How far across the table is {name} from the red dot?", "criteria": CM_LEVELS}
    qs["far:height"] = {"type": "score", "instructions": "How far up or down should the jaws move to get where `goal` needs them next?", "criteria": CM_LEVELS}
    qs["far:turn"] = {"type": "score", "instructions": "In `wrist`, how far must the jaws turn to be square with what `goal` needs them to grip?", "criteria": DEG_LEVELS}
    return qs


def step_vector(answers: dict, obj: str, axes: dict[str, np.ndarray], view: str = "") -> np.ndarray:
    """The expected displacement toward an object: Jev's direction probabilities times its distance, in metres."""
    p = answers[f"where{view}:{obj}"]["probabilities"]
    v = sum(p.get(d, 0.0) * axes[d] for d in ("ahead", "back", "left", "right"))
    return v * expected(answers[f"far{view}:{obj}"], CM) / 100


OPEN_DEG = 70.0
MIN_JAWS_Z_M = 0.012
SETTLE_DEG = 0.5
NEAR_CLIP_M = 0.004


def load_env() -> None:
    for d in [HERE, *HERE.parents]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() and not k.lstrip().startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def jpeg(image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@dataclass
class Spend:
    jev_calls: int = 0
    jev_tokens: int = 0
    jev_s: float = 0.0
    claude_calls: int = 0
    claude_in: int = 0
    claude_out: int = 0
    claude_s: float = 0.0

    def usd(self) -> dict:
        jev = self.jev_tokens * JEV_USD_PER_TOKEN
        claude = self.claude_in * CLAUDE_USD_PER_TOKEN[0] + self.claude_out * CLAUDE_USD_PER_TOKEN[1]
        return {"jev": round(jev, 5), "claude": round(claude, 4), "total": round(jev + claude, 4)}


class Jev:
    def __init__(self, spend: Spend):
        self.spend = spend
        self.http = httpx.Client(timeout=60.0, headers={
            "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            "TypeSafe-Danger-Zone-Images": "true",
        })

    def ask(self, state: dict, images: dict[str, bytes], questions: dict) -> dict:
        assets, refs = {}, {}
        for name, data in images.items():
            ref = f"{name}-{abs(hash(data)) % 10**12}"
            refs[name] = ref
            assets[ref] = {"type": "image", "format": "jpeg", "data_base64": base64.b64encode(data).decode()}
        body = {"model": JEV_MODEL, "state": {**state, **refs}, "questions": questions, "assets": assets}
        t0 = time.monotonic()
        for attempt in range(4):
            try:
                r = self.http.post(JEV_URL, json=body)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        out = r.json()
        self.spend.jev_calls += 1
        self.spend.jev_tokens += out.get("usage", {}).get("input_tokens", 0)
        self.spend.jev_s += time.monotonic() - t0
        return out["answers"]


class GoalWriter:
    """Claude writes the one-sentence goal Jev pursues. It never picks moves."""

    SYSTEM = (
        "You set the goal for a robot arm. A fast vision model looks through the camera on the arm's jaws and picks "
        "one small move at a time toward your goal; it sees nothing else. The task: {task} "
        "Reply with the goal only, one short sentence the vision model can check against the wrist camera. "
        "The red dot in the wrist camera is on the table straight below the jaws."
    )

    def __init__(self, spend: Spend, task: str = TASK):
        import anthropic

        self.system = self.SYSTEM.format(task=task)
        self.spend = spend
        self.client = anthropic.Anthropic()

    def write(self, ask: str, images: dict[str, bytes]) -> str:
        content = []
        for name, data in images.items():
            content.append({"type": "text", "text": f"{name} camera:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(data).decode()}})
        content.append({"type": "text", "text": ask})
        t0 = time.monotonic()
        r = self.client.messages.create(model=CLAUDE_MODEL, max_tokens=2000, system=self.system,
                                        output_config={"effort": "low"}, messages=[{"role": "user", "content": content}])
        self.spend.claude_calls += 1
        self.spend.claude_in += r.usage.input_tokens
        self.spend.claude_out += r.usage.output_tokens
        self.spend.claude_s += time.monotonic() - t0
        return " ".join(b.text for b in r.content if b.type == "text").strip()


class Arm:
    """Moves of the grasp point in the wrist camera's terms, through the sim's IK. Reads only joints and the model."""

    def __init__(self, sim: Sim):
        self.sim = sim
        self.ik = IK(sim.model)
        self.cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.closed = False
        self.side_cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_cam")

    def gripper_goal(self) -> float:
        return self.sim.state()["goals"]["gripper"]

    def axes(self) -> dict[str, np.ndarray]:
        self.ik.pose(self.sim.state()["goals"])
        mujoco.mj_camlight(self.ik.model, self.ik.data)
        rot = self.ik.data.cam_xmat[self.cam].reshape(3, 3)
        right, up, view = rot[:, 0], rot[:, 1], -rot[:, 2]
        flat = lambda v: (v * [1, 1, 0]) / max(np.linalg.norm(v[:2]), 1e-6)
        ahead = flat(up + view)
        across = flat(right)
        z = np.array([0.0, 0.0, 1.0])
        return {"ahead": ahead, "back": -ahead, "right": across, "left": -across, "raise": z, "lower": -z}

    def shift(self, delta_m: np.ndarray) -> str:
        joints = self.sim.state()["goals"]
        p, pitch = self.ik.pose(joints)
        target = p + delta_m
        target[2] = max(target[2], MIN_JAWS_Z_M)
        return self._go(target, pitch, joints["wrist_roll"], joints)

    def turn(self, deg: float) -> str:
        joints = self.sim.state()["goals"]
        p, pitch = self.ik.pose(joints)
        return self._go(p, pitch, joints["wrist_roll"] + deg, joints)

    def _go(self, xyz, pitch: float, roll: float, seed: dict) -> str:
        lo, hi = self.sim.range[MOTORS.index("wrist_roll")]
        roll = float(np.clip(roll, lo, hi))
        for p in (pitch, pitch - 10, pitch + 10, pitch - 20, pitch + 20, pitch - 30):
            try:
                q = self.ik.solve(xyz, float(np.clip(p, 0, 90)), roll, seed=seed, gripper=self.gripper_goal())
            except Unreachable:
                continue
            self.sim.set_goals(q)
            self.settle(q)
            return "moved"
        return "out of reach"

    def settle(self, q: dict, timeout_s: float = 3.0) -> None:
        t_end = self.sim.data.time + timeout_s
        while self.sim.data.time < t_end:
            self.sim.step(0.05)
            now = self.sim.joints()
            if all(abs(now[j] - q[j]) < SETTLE_DEG for j in MOTORS if j != "gripper"):
                return

    def grip(self, close: bool) -> str:
        self.closed = close
        self.sim.set_goals({"gripper": 0.0 if close else OPEN_DEG})
        last = self.sim.joints()["gripper"]
        for _ in range(60):
            self.sim.step(0.05)
            now = self.sim.joints()["gripper"]
            if abs(now - last) < 0.05:
                break
            last = now
        if close:
            self.sim.set_goals({"gripper": max(0.0, now - 3.0)})
            self.sim.step(0.3)
        return "closed" if close else "opened"

    def pixel(self, world, cam: str = "wrist", clip: bool = True) -> tuple[float, float] | None:
        """Where a world point lands in a frame: pinhole camera, then (wrist) the frame's barrel distortion."""
        cid = self.cam if cam == "wrist" else self.side_cam
        self.ik.pose(self.sim.joints())
        mujoco.mj_camlight(self.ik.model, self.ik.data)
        rot = self.ik.data.cam_xmat[cid].reshape(3, 3)
        pc = rot.T @ (np.asarray(world, float) - self.ik.data.cam_xpos[cid])
        if pc[2] > -1e-3:
            return None
        w, h = self.sim.spec.image_size
        f = (h / 2) / np.tan(np.radians(self.sim.model.cam_fovy[cid]) / 2)
        s = np.array([f * pc[0] / -pc[2], -f * pc[1] / -pc[2]]) / (w / 2)
        n = s.copy()
        for _ in range(20 if cam == "wrist" else 0):
            n = s / (1 + self.sim.spec.optics.barrel_k1 * float(n @ n))
        x, y = w / 2 + n[0] * w / 2, h / 2 + n[1] * w / 2
        return (x, y) if not clip or (0 <= x < w and 0 <= y < h) else None

    def side_axes(self) -> dict[str, np.ndarray]:
        """Table directions that move the red dot up, down, left and right in the side frame."""
        p = np.array([*self.sim.state()["grasp_point"][:2], 0.0])
        at = lambda d: np.array(self.pixel(p + d, "side", clip=False))
        jac = np.column_stack([(at([0.01, 0, 0]) - at([0, 0, 0])) / 0.01, (at([0, 0.01, 0]) - at([0, 0, 0])) / 0.01])
        inv = np.linalg.inv(jac)
        unit = lambda px: (lambda v: np.array([*v / np.linalg.norm(v), 0.0]))(inv @ px)
        return {"right": unit([1, 0]), "left": unit([-1, 0]), "ahead": unit([0, -1]), "back": unit([0, 1])}

    def marked_frame(self, cam: str = "wrist"):
        """The frame with a red dot on the table straight below the jaws."""
        image = self.sim.frame(cam)
        xy = self.pixel([*self.sim.state()["grasp_point"][:2], 0.0], cam)
        if xy:
            from PIL import ImageDraw

            x, y = xy
            ImageDraw.Draw(image).ellipse([x - 7, y - 7, x + 7, y + 7], fill=(230, 20, 20), outline=(40, 0, 0), width=2)
        return image

    def reading(self) -> dict:
        st = self.sim.state()
        g, goal = st["joints"]["gripper"], st["goals"]["gripper"]
        jaws = "closed on something" if self.closed and g > 5 else "closed on nothing" if self.closed else "open"
        return {"joints_deg": {j: round(v, 1) for j, v in st["joints"].items()},
                "jaws_height_cm": round(st["grasp_point"][2] * 100, 1), "jaws": jaws}


def expected(answer: dict, values: tuple[float, ...]) -> float:
    return float(sum(p * values[int(k)] for k, p in answer["probabilities"].items()))


@dataclass
class Config:
    max_ticks: int = 100
    max_goals: int = 12
    stall_ticks: int = 6
    side: bool = False
    verify_done: bool = False
    claude: bool = True
    gripper_loops: bool = True
    memory: bool = False
    task: str = "plain"


def run_episode(seed: int, cfg: Config, out: Path) -> dict:
    ep = episode(seed)
    sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
    sim.model.vis.map.znear = NEAR_CLIP_M / sim.model.stat.extent
    arm = Arm(sim)
    spend = Spend()
    jev = Jev(spend)
    writer = GoalWriter(spend, TASKS[cfg.task]) if cfg.claude else None
    memory = Memory() if cfg.memory else None
    questions = questions_for(cfg.side)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "ticks.jsonl").open("w")
    t_wall = time.monotonic()

    def frames() -> dict[str, bytes]:
        f = {"wrist": jpeg(arm.marked_frame())}
        if cfg.side:
            f["side"] = jpeg(arm.marked_frame("side"))
        return f

    f = frames()
    goal = writer.write(f"Arm: {json.dumps(arm.reading())}. Write the first goal.", f) if writer else TASKS[cfg.task]
    goals = [goal]
    history: list[dict] = []
    end = "max ticks"
    for tick in range(cfg.max_ticks):
        f = frames()
        (out / f"t{tick:03d}.jpg").write_bytes(f["wrist"])
        reading = arm.reading()
        last = history[-1] if history else None
        dot = np.array([*sim.state()["grasp_point"][:2], 0.0])
        state = {"goal": goal, "arm": reading, "last_move": last and {k: last[k] for k in ("move", "amount", "result")}}
        if memory and memory.at:
            state["last_seen_cm_from_red_dot"] = memory.cm_from(dot)
        images = f
        answers = jev.ask(state, images, questions)
        move = answers["move:next"]["choice"]
        amount, result = None, None
        if memory:
            memory.see(answers, dot, arm.axes())
        if move in ("to_box", "to_square"):
            obj = move[3:]
            view = "_side" if cfg.side and answers[f"where_side:{obj}"]["confidence"] > answers[f"where:{obj}"]["confidence"] else ""
            delta = step_vector(answers, obj, arm.side_axes() if view else arm.axes(), view)
            if memory and obj in memory.at and answers[f"where:{obj}"]["confidence"] < CONFIDENT and not view:
                delta = memory.at[obj] - dot
                delta *= min(1.0, 0.08 / max(float(np.linalg.norm(delta)), 1e-6))
            amount = round(float(np.linalg.norm(delta)) * 100, 1)
            result = arm.shift(delta)
        elif move in ("lower", "raise"):
            amount = round(max(CM[0], expected(answers["far:height"], CM)), 1)
            result = arm.shift(np.array([0, 0, amount / 100 if move == "raise" else -amount / 100]))
        elif move in ("turn_cw", "turn_ccw"):
            amount = round(max(3.0, expected(answers["far:turn"], DEG)), 1)
            result = arm.turn(-amount if move == "turn_cw" else amount)  # a larger wrist_roll turns the scene counterclockwise
        elif move in ("open", "close"):
            result = arm.grip(move == "close")
        rec = {"tick": tick, "t": sim.state()["t"], "goal": goal, "arm": reading, "move": move, "amount": amount,
               "result": result, "answers": answers, "grasp_point": sim.state()["grasp_point"], "grading": grading(sim)}
        history.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  seed {seed} t{tick:02d} {move:9} {amount or '':>5} {result or ''}  p={answers['move:next']['probabilities'][move]:.2f}", flush=True)

        why = None
        if move == "done":
            why = "done"
        elif stalled(history, cfg.stall_ticks, gripper_counts=not cfg.gripper_loops):
            why = "stall"
        if why and not writer:
            if why == "done":
                end = "done"
                break
            why = None
        if why:
            if len(goals) >= cfg.max_goals:
                end = "max goals"
                break
            f = frames()
            moves = ", ".join(f"{h['move']} {h['amount'] or ''}".strip() + ("" if h["result"] in (None, "moved", "opened", "closed") else f" ({h['result']})") for h in history[-cfg.stall_ticks:])
            if why == "done":
                ask = (f"The vision model says this goal is met: \"{goal}\". Arm: {json.dumps(arm.reading())}. "
                       + ("Check the cameras. If the box rests on the blue square and the jaws are off it, reply DONE. "
                          "If this goal is not met yet, write it again, sharper. Otherwise write the next goal." if cfg.verify_done else
                          "If the box is resting on the blue square and the jaws are off it, reply DONE. Otherwise write the next goal."))
            else:
                ask = (f"The vision model is stuck on the goal \"{goal}\". Its last moves: {moves}. Arm: {json.dumps(arm.reading())}. "
                       "Write a goal that gets it moving again.")
            new = writer.write(ask, f)
            log.write(json.dumps({"tick": tick, "rewrite": why, "goal": new}) + "\n")
            print(f"  seed {seed} goal ({why}): {new}", flush=True)
            if new.strip().rstrip(".").upper() == "DONE":
                end = "done"
                break
            goal = new
            goals.append(goal)
            history.append({"move": "new goal", "amount": None, "result": None, "grasp_point": sim.state()["grasp_point"], "arm": arm.reading()})
    sim.step(0.5)
    gaps = [h["grading"]["jaws_to_box_cm"] for h in history if "grading" in h]
    result = {"seed": seed, **score(sim.truth()), "closest_cm": min(gaps, default=None),
              "touched": any(h["grading"]["touching"] for h in history if "grading" in h), "end": end, "ticks": len([h for h in history if "tick" in h]),
              "sim_s": round(sim.state()["t"], 1), "model_s": round(spend.jev_s + spend.claude_s, 1),
              "wall_s": round(time.monotonic() - t_wall, 1), "goals": goals,
              "jev_calls": spend.jev_calls, "jev_tokens": spend.jev_tokens, "claude_calls": spend.claude_calls,
              "claude_tokens": [spend.claude_in, spend.claude_out], "usd": spend.usd()}
    result["episode_s"] = round(result["sim_s"] + result["model_s"], 1)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    log.close()
    return result


CONFIDENT = 0.5


class Memory:
    """Where the wrist camera last placed the box and the square on the table, from Jev's confident answers. It lets
    the arm carry the box toward a square that the box in its jaws now hides."""

    def __init__(self):
        self.at: dict[str, np.ndarray] = {}

    def see(self, answers: dict, dot: np.ndarray, axes: dict[str, np.ndarray]) -> None:
        for obj in OBJECTS:
            a = answers[f"where:{obj}"]
            if a["confidence"] < CONFIDENT:
                continue
            seen = dot + (0 if a["choice"] == "here" else axes[a["choice"]] * expected(answers[f"far:{obj}"], CM) / 100)
            self.at[obj] = seen if obj not in self.at else (self.at[obj] + seen) / 2

    def cm_from(self, dot: np.ndarray) -> dict:
        return {OBJECTS[k]: round(float(np.linalg.norm((v - dot)[:2])) * 100, 1) for k, v in self.at.items()}


def stalled(history: list[dict], n: int, gripper_counts: bool = True) -> bool:
    """No progress: the jaws are back within 1.5 cm of where they were n ticks ago, and in the same state."""
    recent = history[-n:]
    if len(recent) < n or any(h["move"] == "new goal" for h in recent):
        return False
    if gripper_counts and any(h["move"] in ("open", "close") for h in recent):
        return False
    if recent[-1]["arm"]["jaws"] != recent[0]["arm"]["jaws"]:
        return False
    return float(np.linalg.norm(np.subtract(recent[-1]["grasp_point"], recent[0]["grasp_point"]))) < 0.015


def grading(sim: Sim) -> dict:
    """For the log and the report only; the policy never sees it."""
    t = sim.truth()
    b = t["box"]
    gap = np.linalg.norm(np.subtract(sim.state()["grasp_point"], [b["x"], b["y"], b["h"] / 2]))
    return {"jaws_to_box_cm": round(float(gap) * 100, 1), "touching": t["touching_jaws"], "lifted_cm": round(t["lifted_m"] * 100, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-ticks", type=int, default=Config.max_ticks)
    p.add_argument("--side", action="store_true", help="also send Jev the side camera, with the same red dot")
    p.add_argument("--verify-done", action="store_true", help="Claude checks a done claim before moving on")
    p.add_argument("--no-claude", action="store_true", help="Jev alone: the goal is the task sentence, and Jev's done ends the episode")
    p.add_argument("--memory", action="store_true", help="remember where the wrist camera last saw the box and the square")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-goals", type=int, default=Config.max_goals)
    p.add_argument("--no-gripper-loops", action="store_true", help="gripper moves always count as progress")
    p.add_argument("--task", choices=sorted(TASKS), default="plain", help="the task sentence Claude (or Jev alone) gets")
    p.add_argument("--tag", default="run")
    p.add_argument("--resume", type=Path, help="finish the episodes of an earlier run directory, with its config")
    p.add_argument("--episode", type=int, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    load_env()
    if a.episode is not None:
        cfg = Config(**{k: v for k, v in json.loads((a.out.parent / "config.json").read_text()).items() if not k.startswith("_")})
        run_episode(a.episode, cfg, a.out)
        return 0
    if a.resume:
        run_dir = a.resume.resolve()
        plan = json.loads((run_dir / "config.json").read_text())
        a.seeds, a.repeats, a.tag = plan.pop("_seeds"), plan.pop("_repeats"), plan.pop("_tag")
        cfg = Config(**plan)
    else:
        cfg = Config(max_ticks=a.max_ticks, max_goals=a.max_goals, side=a.side, verify_done=a.verify_done, claude=not a.no_claude,
                     memory=a.memory, gripper_loops=not a.no_gripper_loops, task=a.task)
        run_dir = HERE / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.tag}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({**vars(cfg), "_seeds": a.seeds, "_repeats": a.repeats, "_tag": a.tag}))
    episodes = [run_dir / (f"seed{seed:02d}" + (f"-r{k}" if a.repeats > 1 else "")) for k in range(a.repeats) for seed in a.seeds]

    def one(out: Path) -> dict:
        """Each episode in its own process, so a crash in the renderer costs one retry, not the run."""
        for _ in range(3):
            if (out / "result.json").is_file():
                break
            out.mkdir(parents=True, exist_ok=True)
            with (out / "stdout.log").open("w") as log:
                subprocess.run([sys.executable, __file__, "--episode", out.name.split("-")[0][4:], "--out", str(out)], stdout=log, stderr=subprocess.STDOUT)
        return json.loads((out / "result.json").read_text())

    results = []
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(one, episodes))
    for r in results:
        seed = r["seed"]
        print(f"seed {seed}: placed={r['placed']} lifted={r['lifted']} touched={r['touched']} closest={r['closest_cm']}cm off={r['off_target_m']} end={r['end']} ticks={r['ticks']} "
              f"episode={r['episode_s']}s usd={r['usd']['total']}", flush=True)
    n = len(results)
    summary = {"tag": a.tag, "config": vars(cfg), "seeds": a.seeds, "repeats": a.repeats,
               "placed": sum(r["placed"] for r in results), "lifted": sum(r["lifted"] for r in results), "n": n,
               "median_episode_s": statistics.median(r["episode_s"] for r in results),
               "median_wall_s": statistics.median(r["wall_s"] for r in results),
               "jev_calls_per_ep": sum(r["jev_calls"] for r in results) / n,
               "claude_calls_per_ep": sum(r["claude_calls"] for r in results) / n,
               "usd_per_ep": round(sum(r["usd"]["total"] for r in results) / n, 4),
               "jev_usd_per_ep": round(sum(r["usd"]["jev"] for r in results) / n, 5),
               "claude_usd_per_ep": round(sum(r["usd"]["claude"] for r in results) / n, 4),
               "touched": sum(r["touched"] for r in results),
               "median_closest_cm": statistics.median(r["closest_cm"] for r in results)}
    (run_dir / "summary.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
