"""The runtime: code owns the loop, Jev checks the scene between skills, and a decider
(a person or Claude) settles what Jev is unsure about.

For each layer: find it (scan, then hover and re-measure up close), pick it, check it is
held, place it on the stack, lift, check it is on top and the stack is intact. Each check
reads Jev's probabilities against thresholds as yes, no or unsure; code acts on yes and
no, and only unsure (or a run of failures) goes to the decider.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .escalate import DECISIONS, Decision, Escalation, Escalator
from .geometry import box_offset, box_to_world
from .judge import Answer, Box, Judge, Picked, Question, Thresholds, plain
from .recipe import Item, Recipe, find, holding, placed
from .records import Records
from .robot import Observation, OutOfWorkspace, Robot, Vec3
from .scene import SceneMap, Seen
from .skills import Skill, SkillContext


@dataclass
class Settings:
    table_z: float = 0.0
    board_height_m: float = 0.015  # cutting board thickness
    scan_points: list[Vec3] = field(default_factory=list)  # grasp-point positions to look from
    hover_m: float = 0.12  # look height above an item before picking
    look_up_m: float = 0.10  # look height above the stack after placing
    converge_m: float = 0.01  # re-measuring up close stops once the estimate moves less than this
    max_approach_looks: int = 3
    max_attempts: int = 3  # per layer
    max_escalations: int = 6  # per run; then abort
    max_seconds: float = 900.0
    gripper_open: float = 45.0  # used to let go of an item still held over the stack
    thresholds: Thresholds = field(default_factory=Thresholds)


@dataclass
class Look:
    obs: Observation
    answers: dict[str, Answer]
    readings: dict[str, str]
    frame_file: str


class Abort(Exception):
    pass


class Runner:
    def __init__(self, robot: Robot, judge: Judge, skills: dict[str, Skill], decider: Escalator, records: Records, settings: Settings):
        self.robot, self.judge, self.skills, self.decider = robot, judge, skills, decider
        self.records, self.s = records, settings
        self.scene = SceneMap()
        self.escalations = 0
        self._t0 = time.monotonic()
        self._last_frames: list = []

    # ---- seeing

    def look(self, tag: str, questions: dict[str, Question], facts: dict | None = None) -> Look:
        self.robot.hold()
        obs = self.robot.observe()
        frame_file = self.records.frame(obs.frame, tag)
        answers = self.judge.ask(obs.frame, questions, facts)
        readings = {qid: _read(a, self.s.thresholds) for qid, a in answers.items()}
        self.records.checks(tag, frame_file, questions, answers, readings)
        self._last_frames = (self._last_frames + [obs.frame])[-2:]
        return Look(obs, answers, readings, frame_file)

    def sight(self, look: Look, item: Item, z: float) -> Seen | None:
        p = look.answers[f"present:{item.name}"]
        box = look.answers[f"where:{item.name}"]
        if self.s.thresholds.read(p) != "yes" or not isinstance(box, Box):
            return None
        if box.confidence is not None and box.confidence < self.s.thresholds.box_min_confidence:
            return None
        placed_ = box_to_world(self.robot.camera, look.obs.T_world_cam, box, z)
        if placed_ is None:
            return None
        xyz, span = placed_
        return Seen(xyz, span, p, box.confidence, box_offset(box), look.obs.t)

    def scan(self, items: list[Item]) -> None:
        best: dict[str, Seen] = {}
        questions = {qid: q for item in items for qid, q in find(item).items()}
        for point in self.s.scan_points:
            self._move(point, "scan")
            look = self.look("scan", questions)
            for item in items:
                seen = self.sight(look, item, self._surface_z(item))
                if seen and (item.name not in best or seen.offset < best[item.name].offset):
                    best[item.name] = seen
        for name, seen in best.items():
            self.scene.update(name, seen)
        self.records.event("scan", found=sorted(best), scene=self.scene.as_dict())

    def approach(self, item: Item) -> Seen | None:
        """Hover over the last sighting and re-measure up close until the estimate settles."""
        seen = self.scene.get(item.name)
        z = self._surface_z(item)
        for _ in range(self.s.max_approach_looks):
            if seen is None:
                return None
            self._move(_above(seen.xyz, self.s.hover_m), f"approach {item.name}")
            new = self.sight(self.look(f"approach:{item.name}", find(item)), item, z)
            if new is None:
                self.scene.forget(item.name)
                return None
            self.scene.update(item.name, new)
            settled = math.dist(new.xyz[:2], seen.xyz[:2]) < self.s.converge_m
            seen = new
            if settled:
                break
        return seen

    # ---- doing

    def build(self, recipe: Recipe) -> str:
        outcome, done, skipped = "error", [], []
        try:
            self.scan([recipe.board, *recipe.layers])
            board = self.scene.get(recipe.board.name)
            while board is None:
                self._decide(f"find {recipe.board.name}", "the board was not seen in the scan", {}, ["rescan", "abort"])
                self.scan([recipe.board])
                board = self.scene.get(recipe.board.name)
            stack_z = self.s.table_z + self.s.board_height_m
            for item in recipe.layers:
                self._check_time()
                result = self.layer(item, board.xyz, stack_z)
                if result == "placed":
                    stack_z += item.thickness_m
                    done.append(item.name)
                else:
                    skipped.append(item.name)
            outcome = "done"
        except (Abort, OutOfWorkspace) as e:
            outcome = f"aborted: {e}"
        except Exception as e:
            outcome = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            try:
                self.robot.home()
            except Exception as e:  # keep the original error; record this one
                self.records.event("home_failed", error=f"{type(e).__name__}: {e}")
            self.records.result(recipe=recipe.name, outcome=outcome, placed=done, skipped=skipped, escalations=self.escalations)
        return outcome

    def layer(self, item: Item, board_xyz: Vec3, stack_z: float) -> str:
        """Returns "placed" or "skipped"; raises Abort."""
        facts = {"layer": item.name, "stack_top_z": round(stack_z, 4)}
        attempt = 0
        while attempt < self.s.max_attempts:
            attempt += 1
            facts["attempt"] = attempt
            self.records.event("attempt", item=item.name, n=attempt)

            seen = self.approach(item)
            if seen is None:
                self.scan([item])
                seen = self.approach(item)
            if seen is None:
                d = self._decide(f"find {item.name}", "not found after a rescan", facts, ["retry", "skip", "abort"])
                if d.choice == "skip":
                    return "skipped"
                continue

            self._run_skill(item.pick, SkillContext(item, seen.xyz))
            look = self.look(f"holding:{item.name}", holding(item))
            held = look.readings[f"holding:{item.name}"]
            if held == "unsure":
                d = self._decide(f"pick {item.name}", "unsure whether it is held", facts, ["continue", "retry", "rescan", "skip", "abort"], look)
                if d.choice == "skip":
                    return "skipped"
                if d.choice != "continue":
                    self._redo(d.choice, item)
                    continue
            elif held == "no":
                self._redo("retry", item)  # a missed grasp may have moved it
                continue

            target = (board_xyz[0], board_xyz[1], stack_z)
            self._run_skill(item.place, SkillContext(item, target))
            self._move(_above(target, self.s.look_up_m), f"look at stack after {item.name}")
            look = self.look(f"placed:{item.name}", placed(item), facts)
            on_top, intact = look.readings[f"on_top:{item.name}"], look.readings["stack_ok"]
            if on_top == "yes" and intact == "yes":
                return "placed"
            if intact == "no":
                d = self._decide(f"place {item.name}", "a layer slid off or folded", facts, ["continue", "skip", "abort"], look)
                return "placed" if d.choice == "continue" else "skipped"
            if on_top == "no" and intact == "yes":
                where = look.answers[f"after_miss:{item.name}"]
                if isinstance(where, Picked) and where.choice == "still_held":
                    self.robot.set_gripper(self.s.gripper_open)  # let go over the stack, then check again
                    look = self.look(f"placed:{item.name}", placed(item), facts)
                    if look.readings[f"on_top:{item.name}"] == "yes" and look.readings["stack_ok"] == "yes":
                        return "placed"
                self._redo("retry", item)  # it fell somewhere; find it again
                continue
            d = self._decide(f"place {item.name}", f"on top: {on_top}, stack intact: {intact}", facts, ["continue", "retry", "rescan", "skip", "abort"], look)
            if d.choice == "continue":
                return "placed"
            if d.choice == "skip":
                return "skipped"
            self._redo(d.choice, item)

        self._decide(f"layer {item.name}", f"failed {self.s.max_attempts} attempts", facts, ["skip", "abort"])
        return "skipped"

    # ---- helpers

    def _decide(self, step: str, reason: str, facts: dict, allowed: list[str], look: Look | None = None) -> Decision:
        self.escalations += 1
        if self.escalations > self.s.max_escalations:
            raise Abort(f"more than {self.s.max_escalations} escalations")
        esc = Escalation(
            step=step,
            reason=reason,
            answers={k: plain(v) for k, v in (look.answers if look else {}).items()},
            frames=list(self._last_frames),
            allowed=[a for a in allowed if a in DECISIONS],
            facts=dict(facts),
        )
        self.robot.hold()
        d = self.decider.decide(esc)
        self.records.event("decision", step=step, reason=reason, choice=d.choice, why=d.reason, by=d.by, frame=look.frame_file if look else None)
        if d.choice == "abort":
            raise Abort(f"{d.by} chose abort at '{step}': {d.reason}")
        return d

    def _redo(self, choice: str, item: Item) -> None:
        """Before another attempt: the item may have moved, so its sighting is dropped;
        "rescan" also looks over the whole table for it now."""
        self.scene.forget(item.name)
        if choice == "rescan":
            self.scan([item])

    def _run_skill(self, name: str, ctx: SkillContext) -> None:
        skill = self.skills.get(name)
        if skill is None:
            raise Abort(f"no skill bound to '{name}'")
        self._check_time()
        result = skill.run(self.robot, ctx)
        self.records.event("skill", skill=name, item=ctx.item.name, target=ctx.target, ok=result.ok, note=result.note, gripper=result.gripper)
        if not result.ok:
            raise Abort(f"skill '{name}' could not run: {result.note}")

    def _move(self, xyz: Vec3, why: str) -> None:
        m = self.robot.move_to(xyz)
        self.records.event("move", to=xyz, why=why, ok=m.ok, error_mm=round(m.error_mm, 1), note=m.note)

    def _surface_z(self, item: Item) -> float:
        if item.thickness_m == 0.0:  # the board
            return self.s.table_z + self.s.board_height_m
        return self.s.table_z + item.thickness_m

    def _check_time(self) -> None:
        if time.monotonic() - self._t0 > self.s.max_seconds:
            raise Abort(f"over the time budget of {self.s.max_seconds:.0f} s")


def _above(p: Vec3, dz: float) -> Vec3:
    return (p[0], p[1], p[2] + dz)


def _read(a: Answer, th: Thresholds) -> str:
    if isinstance(a, Box):
        return "box" if a.confidence is None or a.confidence >= th.box_min_confidence else "weak box"
    if isinstance(a, Picked):
        return a.choice
    return th.read(a)
