import numpy as np

from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode

from jc7.body import Body
from jc7.see import fit_prism, fit_square, prism_corners, _bbox


def _sim(seed=3):
    ep = episode(seed)
    sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
    sim.step(0.3)
    return sim, ep


def test_ray_and_projection_invert():
    sim, _ = _sim()
    v = Body(sim).frame()[1]
    for p in ([0.2, 0.05, 0.0], [0.28, -0.1, 0.0], [0.25, 0.0, 0.07]):
        f = v.project(p)
        back = v.on_plane(*f, p[2])
        assert np.linalg.norm(back - p) < 1e-4


def test_look_at_centres_the_point():
    sim, _ = _sim()
    body = Body(sim)
    target = np.array([0.26, 0.08, 0.0])
    view = body.look_at(target, cam_height=0.2)
    assert view is not None
    f = view.project(target)
    assert abs(f[0] - 0.5) < 0.03 and abs(f[1] - 0.5) < 0.03
    assert view.rot[2, 2] > 0.99  # the camera looks along its -z: straight down


def test_prism_fit_recovers_a_box_from_its_image_boxes():
    sim, _ = _sim()
    body = Body(sim)
    truth = dict(x=0.25, y=-0.06, side=0.05, h=0.08, yaw=12.0)
    obs = []
    for cam in ([0.25, -0.06, 0.075, 0.13], [0.25, -0.06, 0.075, 0.19]):
        view = body.look_at(cam[:3], cam_height=cam[3])
        for part in ("all", "top"):
            e = _bbox(view, prism_corners(truth["x"], truth["y"], truth["side"], truth["h"], truth["yaw"], top_only=part == "top"))
            obs.append((view, part, dict(zip(("top", "bottom", "left", "right"), e))))
    view = body.look_across([0.25, -0.06, 0.04])
    e = _bbox(view, prism_corners(truth["x"], truth["y"], truth["side"], truth["h"], truth["yaw"]))
    obs.append((view, "all", dict(zip(("top", "bottom", "left", "right"), e))))
    est = fit_prism(obs)
    assert np.hypot(est.xyz[0] - truth["x"], est.xyz[1] - truth["y"]) < 0.004
    assert abs(est.height - truth["h"]) < 0.01


def test_square_fit():
    sim, _ = _sim()
    body = Body(sim)
    view = body.look_at([0.22, 0.1, 0.0], cam_height=0.2)
    pts = np.array([[0.22 + sx * 0.035, 0.1 + sy * 0.035, 0.0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    e = _bbox(view, pts)
    est = fit_square([(view, dict(zip(("top", "bottom", "left", "right"), e)))])
    assert np.hypot(est.xyz[0] - 0.22, est.xyz[1] - 0.1) < 0.003
