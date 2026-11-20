"""The Jev and Claude adapters against stub clients: what is sent, and how answers are read.
No network: the stubs record the request and return SDK-shaped responses."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import typesafe_sdk as ts

from valet import config
from valet.escalate import AskClaude, Escalation
from valet.judge import Box, JevJudge, Locate, OneOf, Picked, Thresholds, YesNo

FRAME = np.zeros((48, 64, 3), np.uint8)


class StubTypeSafe:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(answers=self.answers)


def test_jev_judge_sends_typed_questions_with_the_image_header():
    stub = StubTypeSafe({
        "held": ts.NoulAnswer(noul=0.82),
        "where": ts.BoxAnswer(box=ts.BoxCoordinates(top=0.1, bottom=0.3, left=0.2, right=0.5), confidence=0.77),
        "which": ts.ChoiceAnswer(choice="beside", confidence=0.6, probabilities={"beside": 0.7, "on_top": 0.3}),
    })
    judge = JevJudge(model="jev-1.14", client=stub)
    answers = judge.ask(
        FRAME,
        {
            "held": YesNo("Is the gripper in `wrist` holding a slice of cheese?", yes="pinched", no="empty"),
            "where": Locate("the slice of cheese in `wrist`"),
            "which": OneOf("Where is it?", {"on_top": "on the sandwich", "beside": "off it"}),
        },
        facts={"layer": "cheese"},
    )
    assert answers == {
        "held": 0.82,
        "where": Box(0.1, 0.3, 0.2, 0.5, 0.77),
        "which": Picked("beside", 0.6, {"beside": 0.7, "on_top": 0.3}),
    }
    call = stub.calls[0]
    assert call["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
    assert call["model"] == "jev-1.14"
    assert isinstance(call["state"]["wrist"], ts.Image) and call["state"]["layer"] == "cheese"
    q = call["questions"]
    assert isinstance(q["held"], ts.Noul) and q["held"].criteria == {"true": "pinched", "false": "empty"}
    assert isinstance(q["where"], ts.Box) and isinstance(q["which"], ts.Choice)


def test_thresholds_read_three_ways():
    th = Thresholds()
    assert [th.read(p) for p in (0.9, 0.7, 0.5, 0.3, 0.1)] == ["yes", "yes", "unsure", "no", "no"]


class StubAnthropic:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _reply(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, model="claude-opus-5-5", content=[SimpleNamespace(type="text", text=text)])


def test_ask_claude_returns_an_allowed_decision():
    stub = StubAnthropic(_reply(json.dumps({"decision": "retry", "reason": "the ham is folded"})))
    esc = Escalation("place ham", "on top: unsure", {"on_top:ham": 0.5}, frames=[FRAME, FRAME], allowed=["continue", "retry", "abort"])
    d = AskClaude(client=stub).decide(esc)
    assert (d.choice, d.reason, d.by) == ("retry", "the ham is folded", "claude-opus-5-5")
    call = stub.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["schema"]["properties"]["decision"]["enum"] == ["continue", "retry", "abort"]
    content = call["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]
    assert json.loads(content[-1]["text"])["vision_answers"] == {"on_top:ham": 0.5}


def test_ask_claude_refusal_becomes_abort():
    stub = StubAnthropic(_reply("", stop="refusal"))
    d = AskClaude(client=stub).decide(Escalation("pick bread", "unsure", {}, frames=[FRAME]))
    assert d.choice == "abort"


def test_example_config_builds_settings_and_skills():
    cfg = config.load(Path(__file__).parent.parent / "valet.example.toml")
    s = config.settings(cfg)
    assert s.scan_points == [(0.22, 0.12, 0.30), (0.22, -0.12, 0.30)] and s.thresholds.yes == 0.7
    skills = config.skills(cfg)
    assert set(skills) == {"pick_flat", "place_on_stack"}
    assert skills["pick_flat"].gripper.open == 45.0
