import json
from types import SimpleNamespace

import numpy as np
import pytest

from physical_agent.agent import Config, Usage, run_episode
from physical_agent.demo import DEMO_SEED, demo_seed, record
from physical_agent.scene import open_episode
from physical_agent.tool import MAX_TARGETS, to_base, to_gripper


def target(x, y, z, pitch=60.0, roll=-90.0, gripper=60.0, steps=10):
    return {"x": x, "y": y, "z": z, "roll_deg": roll, "pitch_deg": pitch, "gripper": gripper, "control_steps": steps}


def test_a_reachable_target_moves_the_grasp_point_there():
    sim, arm, _ = open_episode(0)
    done = arm.execute([target(0.22, 0.05, 0.12), target(0.20, -0.05, 0.10, pitch=80, roll=-60, gripper=30)])
    assert done.text == "executed 2 targets" and done.ran == 2
    e = arm.eef()
    assert np.hypot(e["x"] - 0.20, e["y"] + 0.05) < 0.01 and abs(e["z"] - 0.10) < 0.01
    assert abs(e["pitch_deg"] - 80) < 3 and abs(e["roll_deg"] + 60) < 2


def test_an_unreachable_target_stops_the_sequence_and_says_which():
    sim, arm, _ = open_episode(0)
    before = arm.eef()
    done = arm.execute([target(0.9, 0.0, 0.1), target(0.22, 0.0, 0.12)])
    assert done.ran == 0 and done.text.startswith("executed 0 targets; target 1 failed: no joints put")
    assert arm.eef() == before


def test_too_many_targets_or_missing_fields_are_refused():
    sim, arm, _ = open_episode(0)
    assert arm.execute([target(0.22, 0.0, 0.12)] * (MAX_TARGETS + 1)).ran == 0
    bad = target(0.22, 0.0, 0.12)
    del bad["gripper"]
    assert "missing gripper" in arm.execute([bad]).text


def test_demos_come_from_outside_the_benchmark_and_succeed():
    with pytest.raises(ValueError):
        record(3)
    assert demo_seed() == DEMO_SEED
    demo = record(DEMO_SEED)
    assert demo.result["placed"] and abs(demo.turns[0][1][-1]["roll_deg"] + 90) <= 20
    assert len(demo.turns) == 3 and all(len(actions) <= MAX_TARGETS for _, actions in demo.turns)
    assert sum(b["type"] == "image" for b in demo.blocks("both")) == 6
    assert sum(b["type"] == "image" for b in demo.blocks("wrist")) == 3
    relative = demo.actions(0, "gripper")
    assert set(relative[0]) == {"forward", "left", "up", "roll_deg", "pitch_deg", "gripper", "control_steps"}
    back = to_base(relative[-1], demo.starts[0])
    assert np.hypot(back["x"] - demo.turns[0][1][-1]["x"], back["y"] - demo.turns[0][1][-1]["y"]) < 0.002


def test_gripper_frame_targets_are_offsets_along_the_arms_heading():
    start = np.array([0.0388 + 0.2, 0.2, 0.1])  # 45 degrees left of straight ahead, from the pan axis
    b = to_base({"forward": 0.1, "left": 0.0, "up": -0.05, "roll_deg": -90, "pitch_deg": 90, "gripper": 50,
                 "control_steps": 5}, start)
    assert b["x"] == pytest.approx(start[0] + 0.1 / np.sqrt(2)) and b["y"] == pytest.approx(0.2 + 0.1 / np.sqrt(2))
    assert b["z"] == pytest.approx(0.05)
    assert to_gripper(b, start)["forward"] == pytest.approx(0.1, abs=1e-3)
    sim, arm, _ = open_episode(0, "gripper")
    p0 = arm.point()
    done = arm.execute([{"forward": 0.02, "left": 0.03, "up": -0.04, "roll_deg": -90, "pitch_deg": 70, "gripper": 50,
                         "control_steps": 5}])
    assert done.ran == 1
    moved = to_gripper({**arm.eef(), "control_steps": 5}, p0)
    assert moved["forward"] == pytest.approx(0.02, abs=0.01) and moved["left"] == pytest.approx(0.03, abs=0.01)
    assert moved["up"] == pytest.approx(-0.04, abs=0.01)


class FakeUsage:
    input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 1000, 100, 0, 0

    def model_dump(self, **_):
        return {"input_tokens": 1000, "output_tokens": 100}


class FakeClient:
    """Replays scripted responses and keeps every request it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=lambda o: getattr(o, "__dict__", str(o)))))
        content = self.script.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, usage=FakeUsage())


def call(actions, id="toolu_1"):
    return [SimpleNamespace(type="thinking", thinking="plan", signature="s"),
            SimpleNamespace(type="tool_use", id=id, name="execute_absolute_eef_sequence", input={"actions": actions})]


def test_the_loop_executes_one_call_per_turn_and_ends_when_the_model_stops_calling(tmp_path):
    client = FakeClient([call([target(0.22, 0.0, 0.12)]), [SimpleNamespace(type="text", text="Done.")]])
    ep = run_episode(client, 0, Config(turns=5), out=tmp_path)
    assert ep.turns == 2 and ep.ended == "done"
    second = client.requests[1]["messages"]
    result = second[-1]["content"][0]
    assert result["type"] == "tool_result" and result["content"] == "executed 1 target"
    assert sum(b["type"] == "image" for b in second[-1]["content"]) == 1  # wrist only by default
    assert "scene camera" not in client.requests[0]["system"]
    sent = json.dumps(client.requests)
    assert "truth" not in sent and "lifted_m" not in sent and "tilt_deg" not in sent
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_the_turn_budget_ends_the_episode(tmp_path):
    client = FakeClient([call([target(0.22, 0.0, 0.12)], id=f"toolu_{k}") for k in range(3)])
    ep = run_episode(client, 0, Config(turns=2))
    assert ep.turns == 2 and ep.ended == "turn budget"


def test_dollars_follow_the_price_table():
    u = Usage(input=1_000_000, output=100_000, cache_read=1_000_000, cache_write=0)
    assert u.dollars("claude-opus-5-5") == pytest.approx(4.0 + 2.0 + 0.2)


def test_the_scene_view_is_sent_only_when_asked(tmp_path):
    client = FakeClient([[SimpleNamespace(type="text", text="Done.")]])
    run_episode(client, 0, Config(turns=2, views="both"))
    first = client.requests[0]["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in first) == 2 and "scene camera" in client.requests[0]["system"]
