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
from valet_sim.scene import BoxSpec, SceneSpec, TagSpec, mujoco, tag_corners
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


def test_a_held_box_blocks_the_wrist_view_as_a_real_one_does():
    from valet_sim.scene import NEAR_CLIP_M

    sim = Sim()
    assert sim.model.vis.map.znear * sim.model.stat.extent == pytest.approx(NEAR_CLIP_M)


def test_the_tag_and_the_errors_are_off_by_default():
    from valet_sim.sim import Errors

    sim = Sim()
    assert SceneSpec().tag is None and SimSpec().errors == Errors()
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "tag") == -1
    assert sim.joints() == pytest.approx(sim.truth()["joints"], abs=1e-6)


def test_the_tag_shows_where_tag_corners_says():
    cv2 = pytest.importorskip("cv2")
    tag = TagSpec(x=0.3, y=0.1, yaw_deg=30, id=3)
    sim = Sim(with_box(SimSpec(scene=SceneSpec(tag=tag)), BoxSpec(x=0.6, y=0.5), None))
    side = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_cam")
    sim.model.cam_pos[side], sim.model.cam_quat[side] = [0.3, 0.1, 0.3], [1, 0, 0, 0]  # straight down from 30 cm
    mujoco.mj_forward(sim.model, sim.data)
    image = np.asarray(sim.frame("side"))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    assert ids is not None and ids.ravel().tolist() == [3]
    h, w = image.shape[:2]
    f = h / 2 / np.tan(np.radians(sim.model.cam_fovy[side]) / 2)
    rel = tag_corners(tag) - [0.3, 0.1, 0.3]
    expected = np.column_stack([w / 2 + f * rel[:, 0] / -rel[:, 2], h / 2 - f * rel[:, 1] / -rel[:, 2]]) - 0.5
    assert np.abs(corners[0][0] - expected).max() < 2.0


def test_joint_offsets_shift_readings_and_goals_alike():
    from valet_sim.sim import Errors

    offsets = (3.0, -4.0, 5.0, -2.0, 0.0, 0.0)
    sim = Sim(with_box(SimSpec(errors=Errors(joint_offsets_deg=offsets)), BoxSpec(x=0.6, y=0.5), None))
    q = IK(sim.model).solve([0.25, 0.05, 0.12], pitch_deg=60)
    sim.set_goals(q)
    sim.step(3.0)
    read, true = sim.joints(), sim.truth()["joints"]
    for i, j in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")):
        assert read[j] == pytest.approx(q[j], abs=1.0)
        assert read[j] - true[j] == pytest.approx(offsets[i], abs=0.01)
    assert np.linalg.norm(sim.grasp_point() - [0.25, 0.05, 0.12]) < 0.01  # the arm believes it got there
    assert np.linalg.norm(np.array(sim.truth()["grasp_point"]) - [0.25, 0.05, 0.12]) > 0.01  # it did not


def test_compliance_sags_the_arm_but_not_the_grip():
    from valet_sim.sim import Errors

    sim = Sim(with_box(SimSpec(errors=Errors(compliance_deg_per_nm=4.0)), BoxSpec(x=0.6, y=0.5), None))
    sim.set_goals(IK(sim.model).solve([0.31, 0.0, 0.08], pitch_deg=45))
    sim.step(3.0)
    sag = np.array(sim.truth()["grasp_point"]) - sim.grasp_point()
    assert 0.008 < np.linalg.norm(sag) < 0.025 and sag[2] < -0.005
    assert sim.bend[-1] == 0.0


def test_a_mount_error_moves_the_camera_not_the_arm():
    from valet_sim.sim import Errors

    plain = Sim()
    moved = Sim(SimSpec(errors=Errors(mount_shift_m=(0.004, 0.0, 0.0), mount_tilt_deg=(2.0, 0.0, 0.0))))
    a, b = plain.truth(), moved.truth()
    assert np.linalg.norm(np.subtract(a["wrist_camera"]["pos"], b["wrist_camera"]["pos"])) == pytest.approx(0.004, abs=1e-4)
    assert a["grasp_point"] == b["grasp_point"]


def test_the_scoreboard_counts_benchmark_seeds_and_flags_slow_models(tmp_path):
    from valet_sim.scoreboard import Episode, load, record, table

    log = tmp_path / "experiments" / "x" / "results" / "episodes.jsonl"
    for seed in range(4):
        record(log, Episode("x", "a", seed, placed=seed < 2, lifted=True, seconds=10.0, slow_calls=2, fast_calls=50, usd=0.01))
        record(log, Episode("x", "b", seed, placed=True, lifted=True, seconds=60.0, slow_calls=20, fast_calls=0, usd=0.3))
    record(log, Episode("x", "a", 100, placed=True, lifted=True, seconds=1.0, slow_calls=0, fast_calls=1, usd=0.0))
    rows = {r["config"]: r for r in table(load([tmp_path]))}
    assert (rows["a"]["seeds"], rows["a"]["placed"], rows["a"]["scalable"]) == (4, 2, True)
    assert (rows["b"]["placed"], rows["b"]["scalable"]) == (4, False)
