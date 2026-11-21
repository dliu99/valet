import numpy as np
from valet_sim.ik import IK
from valet_sim.scene import BoxSpec, SceneSpec, TagSpec, tag_corners
from valet_sim.sim import Errors, Sim, SimSpec, with_box

from apriltag_exp import calib, errors, tags
from apriltag_exp.camera import Arm, Lens, angle_deg, true_camera


def test_the_lens_model_undoes_its_own_distortion():
    lens = Lens()
    xy = np.array([[0.0, 0.0], [0.4, -0.2], [-0.9, 0.5]])
    assert np.allclose(lens.undistort(lens.distort(xy)), xy, atol=1e-9)
    assert np.isnan(lens.distort([[5.0, 0.0]])).all()  # beyond what the lens can show


def test_fk_and_the_nominal_mount_give_the_true_camera_when_nothing_is_wrong():
    sim = Sim()
    T, T_true = Arm().camera(sim.joints()), true_camera(sim.truth())
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 1e-6 and angle_deg(T[:3, :3].T @ T_true[:3, :3]) < 1e-4


def test_the_tag_gives_the_camera_pose_from_one_frame():
    tag = TagSpec()
    sim = Sim(with_box(SimSpec(scene=SceneSpec(tag=tag)), BoxSpec(x=0.6, y=0.5), None))
    frame = np.asarray(sim.frame("wrist"))
    s = tags.sight(frame, tag)
    T_true = true_camera(sim.truth())
    assert s is not None and s.reprojection_px < 1.0
    assert np.abs(s.corners_px - Lens().project(T_true, tag_corners(tag))).max() < 2.0
    assert np.linalg.norm(s.T_world_cam[:3, 3] - T_true[:3, 3]) < 0.015


def test_a_sweep_of_tag_views_recovers_a_perturbed_mount():
    tag = TagSpec()
    err = Errors(mount_shift_m=(0.004, -0.003, 0.002), mount_tilt_deg=(2.0, -1.5, 1.0))
    sim = Sim(SimSpec(scene=SceneSpec(tag=tag, box=BoxSpec(x=0.6, y=0.5), target_xy=None), errors=err))
    arm = Arm()
    views = calib.sweep(sim, tag, arm, IK(arm.model))
    t = sim.truth()
    true_mount = np.linalg.inv(Arm().gripper_pose(t["joints"])) @ true_camera(t)
    before = calib.mount_error(arm.mount, true_mount)
    after = calib.mount_error(calib.fit(views, tag, with_offsets=False).mount, true_mount)
    assert len(views) >= 6 and before[0] > 4 and before[1] > 2
    assert after[0] < 1.0 and after[1] < 0.3


def test_error_draws_are_fixed_by_seed_and_only_switch_on_what_is_asked():
    assert errors.draw(4, "all") == errors.draw(4, "all")
    e, _ = errors.draw(4, "mount")
    assert e.compliance_deg_per_nm == 0 and not any(e.joint_offsets_deg) and any(e.mount_tilt_deg)
    e, tag = errors.draw(4, "offsets")
    assert tag == TagSpec() and all(2 <= abs(v) <= 6 for v in e.joint_offsets_deg[:5])
