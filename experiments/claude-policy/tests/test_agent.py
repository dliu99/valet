"""The loop and tools against a stub Claude and a stub Jev: no API calls."""

from types import SimpleNamespace as NS

import pytest

from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode

from claude_policy.agent import Config, run_episode, tools
from claude_policy.arm import Arm


def usage(i=100, w=0, r=0, o=50):
    return NS(input_tokens=i, cache_creation_input_tokens=w, cache_read_input_tokens=r, output_tokens=o)


def call(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


class StubClient:
    """Answers with the scripted turns in order, and keeps every request."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **req):
        self.requests.append({**req, "messages": list(req["messages"])})  # the loop appends to one list
        content, stop = self.turns.pop(0)
        return NS(model=req["model"], content=content, stop_reason=stop, usage=usage())


class StubJev:
    def __init__(self, p):
        self.p = p
        self.questions = []

    def ask(self, frames, question):
        assert set(frames) == {"wrist"} and all(j[:2] == b"\xff\xd8" for j in frames.values())
        self.questions.append(question)
        return self.p, {"input_tokens": 10}


@pytest.fixture
def arm():
    ep = episode(0)
    return Arm(Sim(with_box(SimSpec(), ep.box, ep.target_xy)))


def run(arm, turns, cfg=Config(), checker=None):
    client = StubClient(turns)
    events = []
    res = run_episode(arm, client, cfg, events.append, lambda cam, jpeg: None, checker)
    return res, client, events


def test_calls_run_in_order_and_the_loop_ends_on_text(arm):
    res, client, events = run(arm, [
        ([call("a", "move", xyz=[0.25, 0.0, 0.12], pitch_deg=80), call("b", "gripper", action="close"), call("c", "look")], "tool_use"),
        ([NS(type="text", text="done")], "end_turn"),
    ])
    assert res["end"] == "end_turn" and res["final_text"] == "done"
    assert res["claude_calls"] == 2 and res["tool_calls"] == {"move": 1, "gripper": 1, "look": 1}
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert not any(r["is_error"] for r in results)
    assert arm.state()["joints"]["gripper"] < 5  # nothing between the jaws, so they shut
    first = client.requests[0]["messages"][0]["content"]
    assert [b["type"] for b in first] == ["text", "text", "image", "text"]  # goal, then the wrist frame and state


def test_a_failed_move_skips_the_rest_of_the_turn(arm):
    res, client, _ = run(arm, [
        ([call("a", "move", xyz=[0.9, 0.0, 0.1], pitch_deg=90), call("b", "gripper", action="close")], "tool_use"),
        ([NS(type="text", text="unreachable")], "end_turn"),
    ])
    results = client.requests[1]["messages"][-1]["content"]
    assert results[0]["is_error"] and "no joints" in results[0]["content"]
    assert results[1]["is_error"] and results[1]["content"].startswith("skipped")
    assert res["tool_calls"] == {"move": 1}


def test_history_is_append_only(arm):
    _, client, _ = run(arm, [
        ([call("a", "look")], "tool_use"),
        ([call("b", "gripper", action="open", percent=90)], "tool_use"),
        ([NS(type="text", text="ok")], "end_turn"),
    ])
    a, b = client.requests[1]["messages"], client.requests[2]["messages"]
    assert b[: len(a)] == a


def test_jev_check_says_no_and_stops_the_turn(arm):
    jev = StubJev(0.1)
    res, client, events = run(arm, [
        ([call("a", "gripper", action="close", check="Is the box between the jaws?"), call("b", "move", xyz=[0.25, 0, 0.15], pitch_deg=80)], "tool_use"),
        ([NS(type="text", text="missed")], "end_turn"),
    ], Config(jev_checks=True), jev)
    results = client.requests[1]["messages"][-1]["content"]
    assert "no (p_yes=0.10)" in results[0]["content"] and results[1]["content"].startswith("skipped")
    assert jev.questions == ["Is the box between the jaws?"] and res["jev_calls"] == 1
    assert any(e["type"] == "jev" for e in events)


def test_check_is_offered_only_with_jev():
    def props(cfg):
        return {t["name"]: set(t["input_schema"]["properties"]) for t in tools(cfg)}
    assert "check" not in props(Config())["move"]
    assert "check" in props(Config(jev_checks=True))["gripper"]


def test_moves_keep_the_gripper_setting(arm):
    arm.open(30)
    arm.move_to([0.22, 0.05, 0.12], 70)
    assert abs(arm.state()["joints"]["gripper"] - 30) < 2


def test_jev_verify_sends_claude_back_once_on_no(arm):
    jev = StubJev(0.2)
    res, client, events = run(arm, [
        ([NS(type="text", text="done")], "end_turn"),
        ([call("a", "look")], "tool_use"),
        ([NS(type="text", text="done now")], "end_turn"),
        ([NS(type="text", text="really done")], "end_turn"),
    ], Config(jev_verify=True), jev)
    assert res["final_text"] == "really done" and res["claude_calls"] == 4 and res["jev_calls"] == 2
    assert "answered no" in client.requests[1]["messages"][-1]["content"]
    assert [e["kind"] for e in events if e["type"] == "jev"] == ["verify", "verify"]


def test_jev_verify_yes_ends_the_episode(arm):
    res, _, _ = run(arm, [([NS(type="text", text="done")], "end_turn")], Config(jev_verify=True), StubJev(0.9))
    assert res["end"] == "end_turn" and res["claude_calls"] == 1 and res["jev_calls"] == 1
