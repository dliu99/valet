"""The executor end to end, offline: a stub Jev that answers from the simulator's own geometry (tests only), and a
scripted plan in place of Claude."""

import json

import numpy as np

from jc7.episode import Config, Run
from jc7.jev import Meter
from jc7.planner import ClaudeMeter
from jc7.see import _bbox, prism_corners


class TruthJev:
    """Answers Box questions with the true outline, Nouls with 0.9. Never used outside tests."""

    def __init__(self):
        self.meter = Meter()
        self.run = None

    def ask(self, frames, questions, facts=None):
        self.meter.calls += 1
        sim = self.run.sim
        v = self.run.body.arm.view(sim.joints())
        t = sim.truth()
        b, (tx, ty) = t["box"], t["target"]
        yaw = 0.0
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = {"p": 0.9}
                continue
            text = q["instructions"]
            if "square" in text:
                pts = np.array([[tx + sx * 0.035, ty + sy * 0.035, 0.0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            else:
                pts = prism_corners(b["x"], b["y"], b["w"], b["h"], yaw, top_only="top face" in text)
            e = _bbox(v, pts)
            e = np.clip(e, 0, 1)
            out[qid] = {"box": dict(zip(("top", "bottom", "left", "right"), map(float, e))), "confidence": 0.9}
        return out


class ScriptedPlanner:
    def __init__(self, plan):
        self.meter = ClaudeMeter()
        self._plan = plan

    def plan(self, brief, images):
        assert "box: footprint centre" in brief and images
        self.meter.calls += 1
        self.meter.log.append({"kind": "plan"})
        return self._plan

    def replan(self, report, images):
        self.meter.calls += 1
        return {"approach": "stop", "steps": []}


def step(do, rel="table", G=None, pitch=None, roll=None, gripper=None, expect=None, question="", want="yes"):
    return {"do": do, "rel": rel, "G": G, "pitch": pitch, "roll": roll, "gripper": gripper, "expect": expect,
            "question": question, "want": want, "why": do}


def test_a_scripted_plan_runs_and_is_graded(tmp_path):
    plan = {"approach": "test", "steps": [
        step("move", "box", [0.0, 1.6, 12], 90, -90, 70),
        step("move", "box", [0.0, 1.6, 4], 90, -90, 70),
        step("grip", gripper=0, expect=[10, 45]),
        step("move", "box", [0.0, 1.6, 12], 75, -90),
        step("check", question="Is the white box held in `wrist`?"),
        step("move", "square", [0.0, 1.6, 12], 75, -90),
        step("move", "square", [0.0, 1.6, 4.4], 85, -90),
        step("grip", gripper=60),
        step("move", "square", [0.0, 1.6, 12], 75, -90),
    ]}
    jev = TruthJev()
    run = Run(3, Config(), tmp_path / "ep", jev=jev, planner=ScriptedPlanner(plan))
    jev.run = run
    result = run.run()
    assert "error" not in result, result.get("error")
    assert result["lifted"], result
    assert result["placed"], result
    assert result["self_check"]["placed_self"]
    events = [json.loads(l)["kind"] for l in (tmp_path / "ep" / "events.jsonl").read_text().splitlines()]
    assert events.count("step") == len(plan["steps"]) and "result" in events


def test_a_failed_grip_expectation_stops_the_plan(tmp_path):
    plan = {"approach": "test", "steps": [
        step("move", "box", [0.0, 15.0, 12], 75, -90, 70),  # nowhere near the box
        step("grip", gripper=0, expect=[10, 45]),
        step("move", "box", [0.0, 0.0, 15], 60, -90),
    ]}
    jev = TruthJev()
    run = Run(3, Config(), tmp_path / "ep", jev=jev, planner=ScriptedPlanner(plan))
    jev.run = run
    result = run.run()
    assert result["replans"] >= 1 and not result["lifted"]
    assert run.log[1]["failure"].startswith("gripper reads")
