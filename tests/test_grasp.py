import pytest

from valet.armd.check import run_check
from valet.armd.frames import FileCameras


def delta(app, **d):
    r = app.act({"kind": "delta", **d})
    assert r["ok"], r
    return r["after"]


def test_a_fake_grasp_lifts_the_box(app):
    box = app.fake_truth()["box"]
    delta(app, pitch_deg=-30)
    s = delta(app, reach_m=box["x"] - app.state()["tip"]["x"])
    assert s["tip"]["x"] == pytest.approx(box["x"], abs=0.003)
    while s["height_m"] > box["z"] + 0.03:
        s = delta(app, height_m=-0.03)
    delta(app, height_m=box["z"] - s["height_m"])
    assert not app.fake_truth()["held"]

    r = app.act({"kind": "grip", "to": "close"})
    assert r["ok"] and r["after"]["grip"] == "holding"
    assert app.fake_truth()["held"]

    lifted = delta(app, height_m=0.03)
    truth = app.fake_truth()
    assert truth["box"]["z"] == pytest.approx(box["z"] + 0.03, abs=0.003)
    assert truth["box"]["z"] == pytest.approx(truth["tip"]["z"], abs=0.003)
    assert lifted["grip"] == "holding"

    delta(app, pan_deg=-10)
    moved = app.fake_truth()["box"]
    assert moved["y"] > box["y"] + 0.03

    app.act({"kind": "grip", "to": "open"})
    dropped = app.fake_truth()
    assert not dropped["held"] and dropped["box"]["z"] == pytest.approx(box["z"])
    assert dropped["box"]["y"] == pytest.approx(moved["y"])


def test_check_passes_on_the_fake_and_prints_a_line_per_step(make_app):
    lines = []
    assert run_check(make_app, lines.append)
    assert [ln.split(":")[0] for ln in lines] == ["1 bus", "2 state", "3 cameras", "4 hold", "5 nudge"]
    assert all(": ok" in ln for ln in lines)


def test_check_stops_at_the_first_failure(make_app):
    def blind():
        app = make_app()
        app.cameras = FileCameras({})
        return app

    lines = []
    assert not run_check(blind, lines.append)
    assert lines[-1].startswith("3 cameras: FAIL")
    assert len(lines) == 3
