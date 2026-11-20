import base64
import io
import json
import threading
import urllib.request

import numpy as np
import pytest
from PIL import Image

from valet_sim import expert
from valet_sim.ik import IK
from valet_sim.scene import BoxSpec
from valet_sim.server import App, make_server
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import BENCHMARK_SEEDS, episode, score


def test_joint_goals_are_reached_at_the_servo_speed_limit():
    sim = Sim()
    start = sim.joints()["shoulder_pan"]
    sim.set_goals({"shoulder_pan": start + 45})
    sim.step(0.25)
    assert sim.joints()["shoulder_pan"] - start < 0.25 * SimSpec().servo.max_speed_dps + 2
    sim.step(1.5)
    assert sim.joints()["shoulder_pan"] == pytest.approx(start + 45, abs=1.0)


def test_moving_freely_draws_what_the_real_arm_draws():
    sim = Sim()
    sim.set_goals({"shoulder_lift": -10, "elbow_flex": 20})
    peak = 0.0
    for _ in range(30):
        sim.step(0.05)
        peak = max(peak, max(sim.state()["amps"].values()))
    assert peak < 0.8


def test_ik_puts_the_grasp_point_where_asked():
    sim = Sim()
    q = IK(sim.model).solve([0.25, 0.05, 0.08], pitch_deg=60)
    sim.set_goals(q)
    sim.step(3.0)
    assert np.linalg.norm(sim.grasp_point() - [0.25, 0.05, 0.08]) < 0.01


def test_the_jaws_hold_a_box_they_close_on_and_it_lifts_with_them():
    ep = episode(2)
    sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
    path, pitch = expert.plan_grasp(IK(sim.model), np.array([ep.box.x, ep.box.y, ep.box.h / 2]), ep.box)
    for q in path:
        expert.move(sim, q, timeout_s=1.5)
    expert.close_gripper(sim)
    assert 5 < sim.joints()["gripper"] < 50, "the jaws stopped on the box"
    lifted = IK(sim.model).solve(sim.grasp_point() + [0, 0, 0.05], pitch, path[-1]["wrist_roll"], seed=sim.joints(), gripper=sim.joints()["gripper"])
    expert.move(sim, lifted)
    truth = sim.truth()
    assert truth["held"] and truth["lifted_m"] > 0.02


def test_pushing_the_box_moves_it():
    sim = Sim(with_box(SimSpec(), BoxSpec(x=0.26, y=0.0), (0.22, 0.15)))
    before = sim.truth()["box"]
    ik = IK(sim.model)
    sim.set_goals(ik.solve([0.17, 0.0, 0.04], 60))
    sim.step(3.0)
    for x in np.linspace(0.18, 0.29, 12):
        sim.set_goals(expert.solve_near(ik, [x, 0.0, 0.04], 60, -90.0, sim.joints(), 60.0))
        sim.step(0.4)
    after = sim.truth()["box"]
    assert after["x"] - before["x"] > 0.01 or after["tilt_deg"] > 30


def test_benchmark_episodes_are_fixed_by_their_seed_and_score_from_truth():
    assert episode(7) == episode(7)
    assert len({episode(s).box.x for s in BENCHMARK_SEEDS}) == len(BENCHMARK_SEEDS)
    sim = Sim(with_box(SimSpec(), episode(0).box, episode(0).target_xy))
    result = score(sim.truth())
    assert not result["placed"] and not result["lifted"] and result["resting"]


def test_the_expert_places_the_box_on_a_solvable_seed():
    ep = episode(2)
    assert expert.run(Sim(with_box(SimSpec(), ep.box, ep.target_xy)), ep)["placed"]


def test_frames_are_upright_and_not_blank():
    sim = Sim()
    wrist = np.asarray(sim.frame("wrist"))
    assert wrist.shape == (360, 640, 3) and wrist.std() > 10
    assert np.asarray(sim.frame("side")).std() > 10


def test_the_http_api(unused_port):
    server = make_server(App(seed=3), unused_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{unused_port}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    try:
        assert set(call("/state")) >= {"joints", "goals", "amps", "torque", "grasp_point"}
        assert call("/goals", {"joints": {"gripper": 30}})["goals"]["gripper"] == 30
        frame = call("/frames", {"cams": ["wrist"], "max_side": 320})["frames"]["wrist"]
        assert Image.open(io.BytesIO(base64.b64decode(frame["data_b64"]))).size == (320, 180)
        assert call("/reset", {"seed": 5}) == {"seed": 5}
        assert call("/score")["seed"] == 5
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/move", {"xyz": [0.9, 0, 0.1], "pitch_deg": 30})
        assert e.value.code == 422
    finally:
        server.shutdown()


@pytest.fixture
def unused_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
