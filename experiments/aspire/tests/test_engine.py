import json

import numpy as np
import pytest
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode

from aspire import engine, ledger, perceive, tasks
from aspire.camera import Cameras


def _box_corners(b):
    yaw = np.radians(b.yaw_deg)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return [[*(np.array([b.x, b.y]) + rot @ [sx * b.w / 2, sy * b.d / 2]), z] for z in (0, b.h) for sx in (-1, 1) for sy in (-1, 1)]


@pytest.mark.parametrize("cam", ["wrist", "side"])
def test_a_projected_point_back_projects_to_itself(cam):
    sim = Sim()
    view = Cameras(sim.model).view(cam, sim.joints())
    for xyz in ([0.25, 0.05, 0.0], [0.22, -0.08, 0.03], [0.3, 0.1, 0.07]):
        u, v = view.project(xyz)
        assert np.allclose(view.to_plane(u, v, xyz[2]), xyz, atol=1e-4)


def test_the_wrist_distortion_inverts():
    view = Cameras(Sim().model).view("wrist", Sim().joints())
    for u, v in ((10, 10), (320, 180), (600, 340)):
        assert np.allclose(view.distort(*view.undistort(u, v)), (u, v), atol=1e-3)


@pytest.mark.parametrize("code,why", [
    ("def run(robot):\n    robot._sim.truth()\n", "underscore"),
    ("import os\ndef run(robot):\n    pass\n", "import"),
    ("def run(robot):\n    getattr(robot, 'x')\n", "getattr"),
    ("def go(robot):\n    pass\n", r"run\(robot\)"),
    ("def run(robot):\n    robot.truth\n", "truth"),
])
def test_programs_cannot_reach_past_the_api(code, why):
    with pytest.raises(engine.ProgramRejected, match=why if why != "import" else "import"):
        engine.check_program(code)


MOVES = """
def run(robot):
    robot.set_gripper(70)
    r = robot.move_to_pose([0.25, 0.0, 0.15], 60)
    robot.log("move", r["status"])
    robot.move_to_pose([0.9, 0.0, 0.1], 60)
    s = robot.observe()
    assert s["frames"]["wrist"].shape == (360, 640, 3) and set(s["frames"]) == {"wrist"}
"""


def test_an_episode_traces_every_call_and_grades_it(tmp_path):
    r = engine.run_episode(engine.EpisodeSpec(MOVES, 101, out=str(tmp_path)))
    assert r["error"] is None
    assert [c["prim"] for c in r["calls"]] == ["set_gripper", "move_to_pose", "move_to_pose", "observe"]
    assert r["calls"][0]["result"]["status"] == "reached"
    assert r["calls"][1]["result"]["status"] in ("reached", "stopped")
    assert r["calls"][1]["result"]["error_m"] < 0.02
    assert r["calls"][2]["result"]["status"] == "unreachable"
    assert set(r["score"]) >= {"placed", "lifted", "off_target_m"}
    assert (tmp_path / "c02_after_side.jpg").exists() and (tmp_path / "trace.json").exists()
    assert "move" in r["log"][0]
    assert "move_to_pose([0.25, 0.0, 0.15]" in engine.summarize(r)


def test_a_program_cannot_swallow_the_call_limit():
    code = "def run(robot):\n    for _ in range(100):\n        try:\n            robot.observe()\n        except Exception:\n            pass\n"
    r = engine.run_episode(engine.EpisodeSpec(code, 101, keyframes=False))
    assert r["error"].startswith("limit") and len(r["calls"]) == 60


def test_a_program_error_is_recorded_with_its_line():
    r = engine.run_episode(engine.EpisodeSpec("def run(robot):\n    robot.observe()\n    1 / 0\n", 101, keyframes=False))
    assert "ZeroDivisionError" in r["error"] and "line 3" in r["error"]


def test_localize_back_projects_the_bounding_box_centre(monkeypatch):
    ep = episode(103)
    sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
    view = Cameras(sim.model).view("wrist", sim.joints())
    t = ep.target_xy
    px = np.array([view.project([t[0] + dx, t[1] + dy, 0]) for dx in (-0.035, 0.035) for dy in (-0.035, 0.035)])
    bbox = (px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max())
    monkeypatch.setattr(perceive, "locate", lambda image, text, key, backend: perceive.Found(True, bbox, None, "stub"))
    code = "def run(robot):\n    r = robot.localize('the blue square')\n    robot.log(r['table_xy'])\n"
    r = engine.run_episode(engine.EpisodeSpec(code, 103, keyframes=False))
    xy = r["calls"][0]["result"]["table_xy"]
    assert np.hypot(xy[0] - t[0], xy[1] - t[1]) < 0.01


def test_variant_scenes_put_the_box_right_and_the_target_left():
    for s in tasks.DEBUG_SEEDS["variant"] + tasks.HELDOUT_SEEDS["variant"]:
        ep = tasks.make("variant", s)
        assert ep.box.y <= 0.001 and ep.target_xy[1] > 0
        assert 0.032 <= ep.box.w <= 0.042 and ep.box.h >= 0.085
    assert not set(tasks.DEBUG_SEEDS["benchmark"]) & set(tasks.HELDOUT_SEEDS["benchmark"])


def test_the_budget_stops_calls_before_they_overspend(tmp_path, monkeypatch):
    monkeypatch.setenv("ASPIRE_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("ASPIRE_BUDGET_USD", "1")
    ledger.record("actor", "claude-opus-5-5", 0.9)
    ledger.check(0.05)
    with pytest.raises(ledger.BudgetExceeded):
        ledger.check(0.2)
    assert json.loads((tmp_path / "ledger.jsonl").read_text())["role"] == "actor"


def test_benchmark_episodes_go_on_the_scoreboard_with_claude_as_slow_calls(tmp_path):
    from valet_sim.scoreboard import load, table

    from aspire import board

    results = [{"seed": s, "score": {"placed": s % 2 == 0, "lifted": True}, "sim_s": 10.0, "localize_calls": 4} for s in range(4)]
    board.record_results(results, "claude", "c", tmp_path / "episodes.jsonl")
    board.record_results(results, "jev", "j", tmp_path / "episodes.jsonl")
    rows = {r["config"]: r for r in table(load([tmp_path]))}
    assert rows["c"]["slow_calls"] == 4 and rows["c"]["fast_calls"] == 0 and rows["c"]["placed"] == 2
    assert rows["j"]["slow_calls"] == 0 and rows["j"]["fast_calls"] == 4


def test_a_system_one_box_counts_as_found_whatever_its_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(perceive, "CACHE_DIR", tmp_path)
    path = tmp_path / "jev" / __import__("hashlib").sha256(b"the white box\nk").hexdigest()[:32]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"found": False, "bbox": [10, 10, 60, 50], "confidence": 0.27, "backend": "jev", "cached": False}))
    f = perceive.locate(None, "the white box", "k", "jev")
    assert f.found and f.cached and f.confidence == 0.27
