import json

import pytest

from valet.escalate import Decision
from valet.fake import FakeJudge, FakeRobot, sandwich_world
from valet.recipe import SANDWICH
from valet.records import Records
from valet.runtime import Runner, Settings
from valet.skills import ScriptedPick, ScriptedPlace

SCAN = [(0.22, 0.12, 0.30), (0.22, -0.12, 0.30)]
LAYERS = [i.name for i in SANDWICH.layers]


class Scripted:
    """A decider that answers from a list and remembers what it was asked."""

    def __init__(self, *choices):
        self.choices, self.asked = list(choices), []

    def decide(self, esc):
        self.asked.append(esc)
        return Decision(self.choices.pop(0), "test", "test")


def make(tmp_path, *, script=None, drop_on_pick=None, decider=None, **settings):
    world = sandwich_world()
    robot = FakeRobot(world, drop_on_pick=drop_on_pick)
    judge = FakeJudge(world, robot, script=script)
    records = Records(tmp_path, "sandwich")
    runner = Runner(
        robot, judge, {"pick_flat": ScriptedPick(), "place_on_stack": ScriptedPlace()},
        decider or Scripted(), records, Settings(scan_points=SCAN, **settings),
    )
    return runner, world, robot, judge, records


def events(records, kind):
    lines = (records.dir / "events.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["kind"] == kind]


def test_builds_the_whole_sandwich_without_escalating(tmp_path):
    runner, world, robot, _, records = make(tmp_path)
    assert runner.build(SANDWICH) == "done"
    assert world.stack == LAYERS
    assert runner.escalations == 0
    assert robot.ee == (0.0, 0.0, 0.25)  # homed
    result = json.loads((records.dir / "result.json").read_text())
    assert result["placed"] == LAYERS and result["outcome"] == "done"
    assert (records.dir / "checks.jsonl").read_text().count('"id": "on_top:cheese"') == 1


def test_a_missed_grasp_is_retried_by_code(tmp_path):
    runner, world, _, _, records = make(tmp_path, drop_on_pick={"cheese"})
    assert runner.build(SANDWICH) == "done"
    assert world.stack == LAYERS
    assert runner.escalations == 0
    assert [e["n"] for e in events(records, "attempt") if e["item"] == "cheese"] == [1, 2]


def test_an_unsure_check_goes_to_the_decider(tmp_path):
    decider = Scripted("continue")
    runner, world, _, _, records = make(tmp_path, script={"on_top:ham": [0.5]}, decider=decider)
    assert runner.build(SANDWICH) == "done"
    assert runner.escalations == 1
    esc = decider.asked[0]
    assert esc.step == "place ham"
    assert esc.answers["on_top:ham"] == 0.5
    assert "continue" in esc.allowed and esc.frames
    assert events(records, "decision")[0]["choice"] == "continue"


def test_retry_after_an_unsure_place_finds_the_item_again(tmp_path):
    # Jev cannot tell whether the lettuce is on top; the decider asks for a retry. The lettuce is
    # in fact on the stack, so the retry cannot find it loose and the decider skips it.
    decider = Scripted("retry", "skip")
    runner, world, _, _, _ = make(tmp_path, script={"on_top:lettuce": [0.5]}, decider=decider)
    assert runner.build(SANDWICH) == "done"
    assert [e.step for e in decider.asked] == ["place lettuce", "find lettuce"]
    assert world.stack == LAYERS


def test_abort_stops_and_homes(tmp_path):
    runner, world, robot, _, records = make(tmp_path, script={"holding:bread": [0.5]}, decider=Scripted("abort"))
    outcome = runner.build(SANDWICH)
    assert outcome.startswith("aborted: test chose abort at 'pick bread'")
    assert world.stack == []
    assert robot.ee == (0.0, 0.0, 0.25)
    assert json.loads((records.dir / "result.json").read_text())["outcome"] == outcome


def test_too_many_escalations_abort(tmp_path):
    unsure = [0.5] * 10
    runner, _, _, _, _ = make(tmp_path, script={"holding:bread": unsure}, decider=Scripted(*["retry"] * 10), max_escalations=2)
    assert runner.build(SANDWICH) == "aborted: more than 2 escalations"


def test_a_target_outside_the_workspace_aborts(tmp_path):
    runner, _, _, _, _ = make(tmp_path)
    runner.s.scan_points = [(0.9, 0.0, 0.3)]
    assert runner.build(SANDWICH).startswith("aborted: x=0.900 m is outside")


def test_a_crash_is_recorded_as_an_error_not_done(tmp_path):
    runner, _, robot, judge, records = make(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("service down")

    judge.ask = broken
    with pytest.raises(RuntimeError, match="service down"):
        runner.build(SANDWICH)
    assert json.loads((records.dir / "result.json").read_text())["outcome"] == "error: RuntimeError: service down"
    assert robot.ee == (0.0, 0.0, 0.25)


def test_missing_skill_aborts_before_moving_the_item(tmp_path):
    runner, world, _, _, _ = make(tmp_path)
    runner.skills.pop("pick_flat")
    assert runner.build(SANDWICH) == "aborted: no skill bound to 'pick_flat'"
    assert world.stack == []
