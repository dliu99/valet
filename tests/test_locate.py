import base64
import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mujoco")

from valet.armd.app import open_sim
from valet.armd.settings import Settings
from valet.armd.sim import Optics, SimSpec
from valet.armd.simscene import BoxSpec, Look, SceneSpec

LOOK_DOWN = (25.0, -30.0, 20.0, 80.0, -90.0, 60.0)


def red_centroid(app) -> tuple[float, float, float]:
    frame = app.frames(["wrist"], None)["frames"]["wrist"]
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(frame["data_b64"]))).convert("RGB"), np.int32)
    red = (a[..., 0] > 150) & (a[..., 1] < 90) & (a[..., 2] < 90)
    assert red.sum() > 200, "the box is in view"
    ys, xs = np.nonzero(red)
    h, w = red.shape
    return xs.mean() / w, ys.mean() / h, w / h


@pytest.mark.parametrize("roll, ahead, left", [(-90.0, 0.03, -0.02), (-97.0, 0.03, -0.02), (-90.0, 0.07, 0.06), (-100.0, -0.02, -0.07)])
def test_a_point_of_the_upright_frame_lands_where_the_box_stands(clock, tmp_path, kin, roll, ahead, left):
    joints = (*LOOK_DOWN[:4], roll, LOOK_DOWN[5])
    tip = kin.tip(np.array(joints))[:3, 3]
    box = BoxSpec(x=float(tip[0]) + ahead, y=float(tip[1]) + left, w=0.03, d=0.03, h=0.03)
    scene = SceneSpec(box=box, look=replace(Look(), box_rgba=(0.9, 0.05, 0.05, 1.0)))
    app = open_sim(Settings(allow_motion=True), clock=clock, poses_path=tmp_path / "poses.json",
                   spec=SimSpec(scene=scene, start_joints=joints, optics=replace(Optics(), noise_sd=0.0, blur_px=0.0, vignette=0.0)))
    app.hold()
    clock.sleep(0.5)
    u, v, aspect = red_centroid(app)
    hit = app.locate({"u": u, "v": v, "aspect": aspect, "height_m": box.h})
    assert hit["hit"], hit
    assert hit["xyz_m"][0] == pytest.approx(box.x, abs=0.005)
    assert hit["xyz_m"][1] == pytest.approx(box.y, abs=0.005)


def test_two_views_triangulate_the_box_top(clock, tmp_path, kin):
    joints = (*LOOK_DOWN[:4], -90.0, LOOK_DOWN[5])
    tip = kin.tip(np.array(joints))[:3, 3]
    box = BoxSpec(x=float(tip[0]) + 0.03, y=float(tip[1]) - 0.02, w=0.03, d=0.03, h=0.06)
    scene = SceneSpec(box=box, look=replace(Look(), box_rgba=(0.9, 0.05, 0.05, 1.0)))
    app = open_sim(Settings(allow_motion=True), clock=clock, poses_path=tmp_path / "poses.json",
                   spec=SimSpec(scene=scene, start_joints=joints, optics=replace(Optics(), noise_sd=0.0, blur_px=0.0, vignette=0.0)))
    app.hold()
    clock.sleep(0.5)
    rays = []
    for move in ({"kind": "delta", "pan_deg": 0.0}, {"kind": "delta", "pan_deg": -12.0, "reach_m": -0.04}):
        assert app.act(move)["ok"]
        clock.sleep(0.5)
        u, v, aspect = red_centroid(app)
        rays.append(app.locate({"u": u, "v": v, "aspect": aspect})["ray"])
    hit = app.triangulate({"a": rays[0], "b": rays[1]})
    assert hit["ok"], hit
    assert hit["angle_deg"] > 5
    assert hit["xyz_m"][0] == pytest.approx(box.x, abs=0.01)
    assert hit["xyz_m"][1] == pytest.approx(box.y, abs=0.01)
    assert hit["height_m"] == pytest.approx(box.h, abs=0.025), "the visible red is the top and the sides, so a little under the top"
