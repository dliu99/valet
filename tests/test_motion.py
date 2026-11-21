import numpy as np
import pytest

from valet.armd import fake
from valet.armd.errors import ArmdError
from valet.armd.kinematics import GRIPPER, Kinematics
from valet.armd.poses import PoseBook


def act(app, **motion):
    return app.act(motion)


def test_placo_reference_pose(kin):
    tip = kin.tip(np.array([10.0, -20.0, 30.0, 40.0, -90.0, 0.0]))
    assert np.allclose(tip[:3, 3], [0.290751, -0.036575, 0.091786], atol=1e-6)
    assert np.allclose(tip[:3, 2], [0.633028, -0.111613, -0.766041], atol=1e-6)


def test_reach_moves_the_grasp_point_radially_only(app):
    r = act(app, kind="delta", reach_m=0.02)
    b, a = r["before"], r["after"]
    assert r["ok"]
    assert a["reach_m"] - b["reach_m"] == pytest.approx(0.02, abs=0.002)
    assert a["height_m"] == pytest.approx(b["height_m"], abs=0.002)
    assert a["pitch_deg"] == pytest.approx(b["pitch_deg"], abs=1.0)
    assert a["pan_deg"] == pytest.approx(b["pan_deg"], abs=0.01)


def test_height_moves_the_grasp_point_vertically_only(app):
    r = act(app, kind="delta", height_m=-0.03)
    b, a = r["before"], r["after"]
    assert a["height_m"] - b["height_m"] == pytest.approx(-0.03, abs=0.002)
    assert a["reach_m"] == pytest.approx(b["reach_m"], abs=0.002)
    assert a["tip"]["x"] == pytest.approx(b["tip"]["x"], abs=0.002)


def test_positive_pan_turns_the_base_toward_minus_y(app):
    r = act(app, kind="delta", pan_deg=10)
    b, a = r["before"], r["after"]
    assert a["pan_deg"] - b["pan_deg"] == pytest.approx(10, abs=0.1)
    assert a["tip"]["y"] < b["tip"]["y"] - 0.02
    assert a["reach_m"] == pytest.approx(b["reach_m"], abs=0.001)
    assert a["height_m"] == pytest.approx(b["height_m"], abs=0.001)


def test_pitch_turns_the_gripper_about_the_grasp_point(app):
    r = act(app, kind="delta", pitch_deg=-10)
    b, a = r["before"], r["after"]
    assert a["pitch_deg"] - b["pitch_deg"] == pytest.approx(-10, abs=1.0)
    assert [a["tip"][k] for k in "xyz"] == pytest.approx([b["tip"][k] for k in "xyz"], abs=0.003)


def test_several_fields_are_one_move(app):
    r = act(app, kind="delta", pan_deg=-5, reach_m=0.01, height_m=-0.01)
    b, a = r["before"], r["after"]
    assert r["ok"]
    assert (a["pan_deg"] - b["pan_deg"], a["reach_m"] - b["reach_m"], a["height_m"] - b["height_m"]) == pytest.approx((-5, 0.01, -0.01), abs=0.002)


def test_a_big_move_and_a_move_down_to_the_table_go_all_the_way(app):
    r = act(app, kind="delta", pan_deg=50)
    assert r["ok"] and r["after"]["pan_deg"] - r["before"]["pan_deg"] == pytest.approx(50, abs=0.1)
    r = act(app, kind="delta", height_m=-r["after"]["height_m"])
    assert r["ok"] and r["after"]["height_m"] == pytest.approx(0.0, abs=0.002)


def test_an_unreachable_delta_is_refused_and_nothing_moves(app):
    before = app.state()["joints"]
    with pytest.raises(ArmdError) as e:
        act(app, kind="delta", reach_m=0.3)
    assert (e.value.status, e.value.kind) == (422, "unreachable")
    assert app.state()["joints"] == before


def test_a_lagging_joint_stops_the_move_and_holds_there_with_torque_on(app, clock, monkeypatch):
    monkeypatch.setattr(fake, "SERVO_SPEED_DPS", 5.0)
    r = act(app, kind="delta", pan_deg=30)
    assert not r["ok"] and r["stopped"].startswith("lag: shoulder_pan")
    assert r["after"]["torque"] is True
    stopped_at = r["after"]["joints"]
    clock.sleep(2.0)
    assert app.state()["joints"] == stopped_at
    monkeypatch.setattr(fake, "SERVO_SPEED_DPS", 360.0)
    assert act(app, kind="delta", pan_deg=-5)["ok"]


def test_speed_is_capped_by_config(app, clock):
    t0 = clock.now()
    act(app, kind="delta", pan_deg=20, speed_dps=500)
    assert clock.now() - t0 >= 1.5 * 20 / 25


def test_without_a_hold_motions_are_refused_not_errors(make_app):
    app = make_app()
    r = act(app, kind="delta", pan_deg=5)
    assert not r["ok"] and "POST /hold" in r["refused"]


def test_remembered_poses_persist_and_exclude_the_gripper(app, tmp_path):
    act(app, kind="remember", name="here")
    act(app, kind="delta", pan_deg=15)
    act(app, kind="grip", to="close")
    gripper = app.state()["gripper_pct"]
    r = act(app, kind="pose", name="here")
    assert r["ok"] and r["after"]["pan_deg"] == pytest.approx(0.0, abs=0.05)
    assert r["after"]["gripper_pct"] == pytest.approx(gripper, abs=0.5)
    assert PoseBook(tmp_path / "poses.json").get("here") is not None
    assert "here" in app.pose_book()


def test_an_unknown_pose_is_404(app):
    with pytest.raises(ArmdError) as e:
        act(app, kind="pose", name="nowhere")
    assert e.value.status == 404


def test_grip_facts_come_from_where_the_gripper_stopped(app):
    assert app.state()["grip"] == "unknown"
    act(app, kind="grip", to="close")
    assert app.state()["grip"] == "empty"
    act(app, kind="grip", to="open")
    s = app.state()
    assert s["grip"] == "open" and s["gripper_pct"] == pytest.approx(60, abs=0.5)


def test_grip_jammed_when_it_stops_wide(app):
    tip = app.state()["tip"]
    app.fake_reset({"x": tip["x"], "y": tip["y"], "w": 0.04, "d": 0.04, "h": 2 * tip["z"]})
    barely_open = app.guard.goal.copy()
    barely_open[GRIPPER] = 43.0
    app.guard.follow([barely_open])
    act(app, kind="grip", to="close")
    assert app.state()["grip"] == "jammed"


def test_grip_holding_remembers_the_grasp_pose(app):
    tip = app.state()["tip"]
    app.fake_reset({"x": tip["x"], "y": tip["y"], "w": 0.04, "d": 0.04, "h": 2 * tip["z"]})
    act(app, kind="grip", to="close")
    s = app.state()
    assert s["grip"] == "holding" and s["gripper_pct"] == pytest.approx(40, abs=1.0)
    assert np.allclose(app.poses.get("last_grasp"), list(s["joints"].values())[:5], atol=0.1)


def test_kinematics_is_shared_by_fake_and_state(app):
    s = app.state()
    q = np.array(list(s["joints"].values()))
    assert np.allclose(Kinematics().tip(q)[:3, 3], [s["tip"][k] for k in "xyz"], atol=1e-3)


def test_to_moves_the_grasp_point_to_an_absolute_pose(app):
    app.hold()
    result = act(app, kind="to", pan_deg=-20.0, reach_m=0.22, height_m=0.12, pitch_deg=40.0)
    assert result["ok"], result
    after = app.state()
    assert after["pan_deg"] == pytest.approx(-20.0, abs=0.5)
    assert after["reach_m"] == pytest.approx(0.22, abs=0.005)
    assert after["height_m"] == pytest.approx(0.12, abs=0.005)
    assert after["pitch_deg"] == pytest.approx(40.0, abs=1.5)
    kept = act(app, kind="to", height_m=0.15)
    assert kept["ok"], kept
    assert app.state()["reach_m"] == pytest.approx(0.22, abs=0.005), "fields left out keep their value"


def test_jaws_that_close_further_after_holding_report_a_drop(app):
    from valet.armd.motion import GripStop

    app.mover._grip = GripStop("close", 60.0, 24.0)
    assert app.mover.grip_fact(24.0) == "holding"
    assert app.mover.grip_fact(23.0) == "holding"
    assert app.mover.grip_fact(21.0) == "dropped"
