"""In-context demonstrations: `valet_sim.expert` plans from the true box pose (allowed for demos, never for the
policy), and its waypoints run through the same tool the agent uses, so the demo is in the agent's own action
space. A demo is three observations, each followed by the targets executed from it.

`task` is the benchmark task on another episode. `distractor` is the paper's control: a different task (lift the
box and set it back where it was) from the same start as the task demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from valet_sim.expert import APPROACH_STEPS, LIFT_M, OPEN_PCT, plan_grasp, solve_near
from valet_sim.ik import Unreachable
from valet_sim.task import BENCHMARK_SEEDS

from .scene import Observation, grade, observation_blocks, observe, open_episode
from .tool import NAMES, Arm, to_gripper

INSTRUCTION = "Pick up the white box and set it down on the blue square."
DISTRACTOR_INSTRUCTION = "Pick up the white box and set it back down where it was."
GRIP_CLOSED = 0.0
DEMO_SEED = 104  # demo_seed(); 101 was used first (see REPORT.md)
LEVEL_WITHIN_DEG = 20.0


@dataclass
class Demo:
    kind: str
    seed: int
    instruction: str
    turns: list[tuple[Observation, list[dict]]] = field(default_factory=list)
    starts: list[np.ndarray] = field(default_factory=list)
    result: dict = field(default_factory=dict)

    def actions(self, k: int, frame: str = "base") -> list[dict]:
        """Call k's targets as the agent's tool takes them."""
        actions = self.turns[k][1]
        return actions if frame == "base" else [to_gripper(a, self.starts[k]) for a in actions]

    def blocks(self, views: str = "wrist", frame: str = "base") -> list[dict]:
        out: list[dict] = [{"type": "text", "text": f"Demonstration on another episode. Instruction: {self.instruction}"}]
        for k, (obs, _) in enumerate(self.turns):
            out += observation_blocks(obs, f"Demonstration observation {k + 1}:", views)
            out.append({"type": "text", "text": f"Demonstration call: {NAMES[frame]}("
                        + json.dumps({"actions": self.actions(k, frame)}) + ")"})
        out.append({"type": "text", "text": "End of demonstration."})
        return out


def target(arm: Arm, q: dict[str, float], steps: int) -> dict:
    p, pitch = arm.ik.pose(q)
    return {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "z": round(float(p[2]), 3),
            "roll_deg": round(q["wrist_roll"], 1), "pitch_deg": round(pitch, 1), "gripper": round(q["gripper"], 1),
            "control_steps": steps}


def record(seed: int, kind: str = "task") -> Demo:
    if seed in BENCHMARK_SEEDS:
        raise ValueError(f"seed {seed} is a benchmark seed; demos come from outside {BENCHMARK_SEEDS[0]}-{BENCHMARK_SEEDS[-1]}")
    return run_expert(seed, kind)


def run_expert(seed: int, kind: str = "task") -> Demo:
    """The expert's plan executed through the tool. On benchmark seeds this only checks the tool, never feeds a policy."""
    if kind not in ("task", "distractor"):
        raise ValueError(kind)
    sim, arm, ep = open_episode(seed)
    ik = arm.ik
    b = sim.truth()["box"]
    centre = np.array([b["x"], b["y"], min(b["h"] / 2, 0.04)])
    path, pitch = plan_grasp(ik, centre, ep.box)
    grasp = path[-1]
    roll = grasp["wrist_roll"]
    site, _ = ik.pose(grasp)
    held = site - centre
    dest = np.array([*ep.target_xy, centre[2]]) if kind == "task" else centre.copy()
    lift = solve_near(ik, site + [0, 0, LIFT_M], pitch, roll, grasp, GRIP_CLOSED)
    carry = solve_near(ik, dest + held + [0, 0, LIFT_M], pitch, roll, lift, GRIP_CLOSED)
    place = solve_near(ik, dest + held + [0, 0, 0.004], pitch, roll, carry, GRIP_CLOSED)
    plan = [
        [target(arm, q, 10 if k == 0 else 3) for k, q in enumerate(path[:: max(1, APPROACH_STEPS // 7)])],
        [target(arm, {**grasp, "gripper": GRIP_CLOSED}, 10), target(arm, lift, 10)],
        [target(arm, carry, 15), target(arm, place, 10), target(arm, {**place, "gripper": OPEN_PCT}, 5),
         target(arm, {**carry, "gripper": OPEN_PCT}, 10)],
    ]
    demo = Demo(kind, seed, INSTRUCTION if kind == "task" else DISTRACTOR_INSTRUCTION)
    for actions in plan:
        demo.turns.append((observe(arm), actions))
        demo.starts.append(arm.point())
        arm.execute(actions)
    sim.step(0.5)
    demo.result = grade(sim)
    if kind == "distractor":
        t = sim.truth()["box"]
        demo.result["moved_m"] = round(float(np.hypot(t["x"] - ep.box.x, t["y"] - ep.box.y)), 4)
    return demo


def demo_seed(seeds=range(100, 140), level_within_deg: float = LEVEL_WITHIN_DEG) -> int:
    """The first seed outside the benchmark where the expert places the box, the distractor lifts it and sets it
    back within 2 cm, and the grasp holds the wrist camera within `level_within_deg` of level. The level rule
    matters: at seed 101 the expert grasps at roll 80 (camera nearly upside down), and the agent copied that roll
    and then could not read its own wrist view."""
    for seed in seeds:
        try:
            task, distractor = record(seed, "task"), record(seed, "distractor")
        except Unreachable:
            continue
        r = distractor.result
        roll = task.turns[0][1][-1]["roll_deg"]
        if task.result["placed"] and r["lifted"] and r["released"] and r["moved_m"] < 0.02 and abs(roll + 90) <= level_within_deg:
            return seed
    raise RuntimeError(f"no demo seed in {seeds}")


def find(kind: str, seed: int = DEMO_SEED) -> Demo:
    return record(seed, kind)


def save(demo: Demo, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, (obs, actions) in enumerate(demo.turns, 1):
        obs.scene.save(out / f"obs{k}-scene.jpg")
        obs.wrist.save(out / f"obs{k}-wrist.jpg")
    (out / "demo.json").write_text(json.dumps({
        "kind": demo.kind, "seed": demo.seed, "instruction": demo.instruction, "result": demo.result,
        "turns": [{"state": obs.state, "actions": actions} for obs, actions in demo.turns]}, indent=1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m physical_agent.demo", description=__doc__)
    p.add_argument("--kind", choices=("task", "distractor"), default="task")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    demo = find(a.kind, DEMO_SEED if a.seed is None else a.seed)
    print(f"{demo.kind} demo, seed {demo.seed}: {demo.result}")
    for k, (_, actions) in enumerate(demo.turns, 1):
        print(f" call {k}: {json.dumps(actions)}")
    if a.out:
        save(demo, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
