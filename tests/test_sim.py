import base64
import io

import numpy as np
import pytest
from PIL import Image

mujoco = pytest.importorskip("mujoco")

from valet.armd.app import open_sim
from valet.armd.kinematics import MOTORS
from valet.armd.settings import Settings
from valet.armd.sim import SimSpec
from valet.armd.simscene import BoxSpec, SceneSpec, build

GRASP = (0.0, -2.65, 73.62, -40.97, -90.0, 60.0)


def sim_app(make_clock, tmp_path, **spec):
    return open_sim(Settings(allow_motion=True), clock=make_clock, poses_path=tmp_path / "poses.json", spec=SimSpec(**spec))


def test_the_mujoco_model_moves_like_the_urdf_armd_plans_with(kin):
    m = build()
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in MOTORS]
    rng = np.random.default_rng(1)
    offsets = []
    for _ in range(50):
        q = np.concatenate([rng.uniform(-60, 60, 5), [30.0]])
        d.qpos[adr] = np.radians(q)
        mujoco.mj_kinematics(m, d)
        tip = kin.tip(q)
        offsets.append(tip[:3, :3].T @ (d.site_xpos[site] - tip[:3, 3]))
    spread = np.ptp(np.array(offsets), axis=0)
    assert np.all(spread < 1e-4), f"the MJCF and URDF disagree by more than a fixed site offset: {spread}"


def test_a_delta_lands_where_armd_planned_it(clock, tmp_path):
    app = sim_app(clock, tmp_path)
    app.hold()
    clock.sleep(0.5)
    before = app.state()
    result = app.act({"kind": "delta", "reach_m": 0.03})
    assert result["ok"], result
    assert result["after"]["reach_m"] - before["reach_m"] == pytest.approx(0.03, abs=0.003)
    assert max(result["after"]["amps"].values()) < 1.0, "moving freely draws what the real arm draws"


def test_closing_on_the_box_stalls_holds_and_lifts_it(clock, tmp_path, kin):
    tip = kin.tip(np.array(GRASP))[:3, 3]
    box = BoxSpec(x=float(tip[0]), y=float(tip[1]) - 0.01, w=0.04, d=0.04, h=0.05)
    app = sim_app(clock, tmp_path, scene=SceneSpec(box=box), start_joints=GRASP)
    app.hold()
    clock.sleep(0.5)
    assert app.act({"kind": "grip", "to": "close"})["ok"]
    state = app.state()
    assert state["grip"] == "holding"
    assert 5.0 < state["gripper_pct"] < 40.0, "the jaws stopped on the box, not fully closed"
    z0 = app.fake_truth()["box"]["z"]
    app.act({"kind": "delta", "height_m": 0.04})
    truth = app.fake_truth()
    assert truth["held"]
    assert truth["box"]["z"] - z0 > 0.025


def test_an_empty_close_reads_empty(clock, tmp_path):
    app = sim_app(clock, tmp_path)
    app.hold()
    assert app.act({"kind": "grip", "to": "close"})["ok"]
    assert app.state()["grip"] == "empty"


def test_the_wrist_frame_is_rendered_and_turned_upright_like_a_real_one(clock, tmp_path):
    app = sim_app(clock, tmp_path)
    app.hold()
    out = app.frames(["wrist", "side"], 320)
    assert out["missing"] == {}
    wrist = out["frames"]["wrist"]
    assert wrist["upright"] and (wrist["width"], wrist["height"]) == (320, 180)
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(wrist["data_b64"]))))
    assert pixels.std() > 10, "a blank frame"
    assert pixels[: 60].mean() != pytest.approx(pixels[-60:].mean(), abs=1.0)
