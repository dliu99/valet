"""Part 2 and 3: how well a wrist-only policy can place the box and the blue square in the world, and where the arm
then actually goes when told to go there, for each source of the camera's pose.

One trial is one benchmark scene with one draw of errors. The policy takes the same frames for every method: the
start view, a look straight down at the box from 20 cm (and a second, higher one offset toward the base, for
triangulation), and a look down at the square. Only the camera poses differ:

  fk        forward kinematics of the readings and the nominal mount: what the policy has today
  fk_tri    fk, with the box triangulated from the two box looks instead of assuming its height (tag-free best)
  tag       the tag's PnP pose when it is in the frame; otherwise the last tag view, carried by the FK change since
  mount_cal fk with the mount from a hand-eye fit to a sweep of tag views (no tag at run time)
  cal       fk with mount, joint zeros and sag all fitted to the sweep (no tag at run time); commands corrected too
  cal_tag   cal, plus the tag at run time as in `tag`

Errors are reported two ways: in the world (est - truth), and as reach: the arm is told to put its grasp point at
the estimate, through the same model, and the error is where the grasp point truly goes, less the truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS, BoxSpec, SceneSpec, TagSpec
from valet_sim.sim import START_JOINTS, Sim, SimSpec, with_box
from valet_sim.task import episode

from . import calib, detect, errors, tags
from .camera import Arm, Lens, angle_deg, on_plane, triangulate, true_camera
from .look import look_at

ASSUMED_BOX_H = 0.075  # the middle of the benchmark's 6-9 cm
TARGET_Z = 0.001
GRASP_Z = 0.04
PLACE_Z = 0.07
METHODS = ("fk", "fk_tri", "tag", "mount_cal", "cal", "cal_tag")


def make_sim(seed: int, kind: str, tag: TagSpec | None, box: BoxSpec | None = None, target_xy="episode") -> tuple[Sim, object]:
    """The trial's scene and errors; `tag` None leaves the tag off the table (the errors are the same either way)."""
    ep = episode(seed)
    err, true_tag = errors.draw(seed, kind, tag or TagSpec())
    spec = with_box(SimSpec(scene=SceneSpec(tag=true_tag if tag else None), errors=err), box or ep.box, ep.target_xy if target_xy == "episode" else target_xy)
    return Sim(spec), err


def calibrate(seed: int, kind: str, tag: TagSpec) -> dict[str, calib.Fit]:
    """The one-time calibration: the arm, with the same errors, sweeps the tag with nothing else on the table."""
    sim, _ = make_sim(seed, kind, tag, box=BoxSpec(x=0.6, y=0.5), target_xy=None)
    arm = Arm()
    views = calib.sweep(sim, tag, arm, IK(arm.model))
    return {"mount": calib.fit(views, tag, with_offsets=False), "full": calib.fit(views, tag, with_offsets=True)}


@dataclass
class Frame:
    joints: dict[str, float]
    image: np.ndarray
    true_cam: np.ndarray  # for measuring only
    tag: tags.Sighting | None = None


@dataclass
class Poser:
    """Camera poses for one method."""

    arm: Arm
    use_tag: bool
    tag: TagSpec
    anchor: tuple[np.ndarray, np.ndarray] | None = None  # (tag pose, model pose) of the last frame that saw the tag

    def __call__(self, f: Frame) -> np.ndarray:
        model = self.arm.camera(f.joints)
        if not self.use_tag:
            return model
        if f.tag is not None:
            self.anchor = (f.tag.T_world_cam, model)
            return f.tag.T_world_cam
        if self.anchor is None:
            return model
        T_tag, T_model = self.anchor
        return T_tag @ np.linalg.inv(T_model) @ model


@dataclass
class Trial:
    seed: int
    kind: str
    sim: Sim
    lens: Lens = field(default_factory=Lens)
    frames: dict[str, Frame] = field(default_factory=dict)

    def grab(self, name: str, tag: TagSpec | None) -> Frame:
        image = np.asarray(self.sim.frame("wrist"))
        s = tags.sight(image, tag, self.lens) if tag else None
        f = Frame(self.sim.joints(), image, true_camera(self.sim.truth()), s)
        self.frames[name] = f
        return f

    def go(self, goal: dict[str, float], settle_s: float = 2.5) -> None:
        self.sim.set_goals(goal)
        self.sim.step(settle_s)


def _exclude(jaw: np.ndarray, f: Frame, lens: Lens) -> np.ndarray:
    """The jaws, and the tag (with its white margin) if the frame shows it."""
    import cv2

    out = jaw.copy()
    if f.tag is not None:
        c = f.tag.corners_px
        quad = c.mean(0) + (c - c.mean(0)) * 1.6
        m = np.zeros_like(jaw, np.uint8)
        cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
        out |= m > 0
    return out


def run(seed: int, kind: str, tag: TagSpec = TagSpec(), fits: dict[str, calib.Fit] | None = None) -> dict:
    fits = fits or calibrate(seed, kind, tag)
    sim, err = make_sim(seed, kind, tag)
    arm, lens = Arm(), Lens()
    ik = IK(arm.model)
    jaw = detect.arm_mask(sim)
    t = Trial(seed, kind, sim)
    truth = sim.truth()
    box_true = np.array([truth["box"]["x"], truth["box"]["y"], truth["box"]["h"]])
    target_true = np.array([*truth["target"], TARGET_Z])

    # the frames: start, then looks driven by the plain FK estimate so every method sees the same images
    a = t.grab("start", tag)
    T_a = arm.camera(a.joints)
    wb = detect.white(a.image, _exclude(jaw, a, lens))
    bb = detect.blue(a.image, _exclude(jaw, a, lens))
    out = {"seed": seed, "kind": kind, "errors": {"offsets_deg": list(err.joint_offsets_deg), "compliance": err.compliance_deg_per_nm,
                                                  "mount_shift_m": list(err.mount_shift_m), "mount_tilt_deg": list(err.mount_tilt_deg)},
           "tag_seen": {}, "calibration": {k: {"rms_px": f.rms_px, "views": f.views} for k, f in fits.items()}}
    if wb is None or bb is None:
        out["failed"] = "start view: box or square not found"
        return out
    box_coarse = on_plane(*lens.ray(T_a, wb.centroid_px), ASSUMED_BOX_H)
    target_coarse = detect.on_table(bb, T_a, TARGET_Z, lens)
    if box_coarse is None or target_coarse is None:
        out["failed"] = "start view: estimate off the table"
        return out
    try:
        t.go(look_at(arm, ik, box_coarse[:2], 0.20))
        t.grab("box", tag)
        t.go(second_look(arm, ik, box_coarse[:2]))
        t.grab("box2", tag)
        t.go(look_at(arm, ik, target_coarse[:2], 0.20))
        t.grab("target", tag)
    except Unreachable as e:
        out["failed"] = f"look: {e}"
        return out
    out["tag_seen"] = {k: f.tag is not None for k, f in t.frames.items()}
    if out["tag_seen"]["start"] and t.frames["start"].tag is not None:
        s = t.frames["start"].tag
        T_true = t.frames["start"].true_cam
        out["tag_pose_error_start"] = {"mm": float(np.linalg.norm(s.T_world_cam[:3, 3] - T_true[:3, 3]) * 1000),
                                       "deg": angle_deg(s.T_world_cam[:3, :3].T @ T_true[:3, :3]), "side_px": s.side_px}

    # pixels: the detector's, and the truth's projection through the true camera (geometry alone)
    fb, fb2, ft = t.frames["box"], t.frames["box2"], t.frames["target"]
    pix = {"true": {}, "det": {}}
    for name, f in (("box", fb), ("box2", fb2)):
        pix["true"][name] = lens.project(f.true_cam, box_true)[0]
        blob = detect.white(f.image, _exclude(jaw, f, lens))
        pix["det"][name] = None if blob is None else blob.centroid_px
    pix["true"]["target"] = lens.project(ft.true_cam, target_true)[0]
    target_blob = detect.blue(ft.image, _exclude(jaw, ft, lens))

    models = {"fk": Arm(), "fk_tri": Arm(), "tag": Arm(), "mount_cal": fits["mount"].arm(), "cal": fits["full"].arm(), "cal_tag": fits["full"].arm()}
    estimates: dict[str, dict] = {}
    for method in METHODS:
        poser = Poser(models[method], method in ("tag", "cal_tag"), tag)
        poses = {k: poser(t.frames[k]) for k in ("start", "box", "box2", "target")}
        for src in ("true", "det"):
            e = {}
            ub, ub2 = pix[src]["box"], pix[src]["box2"]
            if ub is not None and lens.visible(ub):
                if method == "fk_tri" and ub2 is not None and lens.visible(ub2):
                    e["box"] = triangulate([lens.ray(poses["box"], ub), lens.ray(poses["box2"], ub2)])
                elif method != "fk_tri":
                    e["box"] = on_plane(*lens.ray(poses["box"], ub), ASSUMED_BOX_H)
            if src == "true":
                e["target"] = on_plane(*lens.ray(poses["target"], pix["true"]["target"]), TARGET_Z)
            elif target_blob is not None:
                e["target"] = detect.on_table(target_blob, poses["target"], TARGET_Z, lens)
            estimates[f"{method}/{src}"] = {k: v for k, v in e.items() if v is not None}

    # world error, then reach error through each method's own commanding model
    probe, _ = make_sim(seed, kind, tag, box=BoxSpec(x=0.6, y=0.5), target_xy=None)
    results = {}
    for key, e in estimates.items():
        method = key.split("/")[0]
        r = {}
        for what, est in e.items():
            true_xy = box_true[:2] if what == "box" else target_true[:2]
            r[f"{what}_world_mm"] = float(np.linalg.norm(est[:2] - true_xy) * 1000)
            if what == "box" and method == "fk_tri":
                r["box_h_err_mm"] = float((est[2] - box_true[2]) * 1000)
            site = reach(probe, models[method], est[:2], GRASP_Z if what == "box" else PLACE_Z)
            if site is not None:
                r[f"{what}_reach_mm"] = float(np.linalg.norm(site[:2] - true_xy) * 1000)
        results[key] = r
    out["results"] = results
    out["true_mount_vs_fit"] = mount_errors(sim, fits)
    out["fit"] = {k: {"offsets_deg": f.offsets_deg[:5].tolist(), "compliance": f.compliance_deg_per_nm} for k, f in fits.items()}
    return out


def second_look(arm: Arm, ik: IK, xy) -> dict[str, float]:
    """A second look at the box for triangulation: from 6 cm to one side or the other and a little higher."""
    radial = np.asarray(xy) / np.linalg.norm(xy)
    side = np.array([-radial[1], radial[0]])
    for d in (side, -side, -radial, radial):
        try:
            return look_at(arm, ik, np.asarray(xy) + 0.06 * d, 0.24)
        except Unreachable:
            continue
    raise Unreachable(f"no second look at {np.round(xy, 3).tolist()}")


def reach(probe: Sim, model: Arm, xy, z: float) -> np.ndarray | None:
    """Where the grasp point truly ends up when the arm is told, through `model`, to put it at (xy, z)."""
    q = solve(model, [xy[0], xy[1], z])
    if q is None:
        return None
    settle(probe, START_JOINTS)
    goal = model.command(q)
    settle(probe, goal)
    now = probe.joints()  # the servos stop short under load; nudge the goals by what the readings say is missing
    settle(probe, {j: goal[j] + (goal[j] - now[j]) if j != "gripper" else goal[j] for j in goal})
    return np.array(probe.truth()["grasp_point"])


def solve(model: Arm, xyz, pitches=(85.0, 75.0, 65.0, 55.0), roll_deg: float = -90.0, gripper: float = 60.0) -> dict | None:
    """IK through the arm's model, polished to 0.5 mm (the stock solver stops at 3)."""
    from scipy.optimize import least_squares

    ik = IK(model.model)
    for pitch in pitches:
        try:
            q = ik.solve(xyz, pitch, roll_deg, gripper=gripper)
        except Unreachable:
            continue
        free = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

        def r(x):
            p, pt = ik.pose({**q, **dict(zip(free, x))})
            return np.r_[(p - xyz) * 1000, (pt - pitch) * 0.2]

        sol = least_squares(r, [q[j] for j in free])
        q = {**q, **dict(zip(free, (float(v) for v in sol.x)))}
        goal = model.command(q)
        if all(lo + 1 < goal[j] < hi - 1 for j, (lo, hi) in zip(MOTORS[:5], ik.range[:5])):
            return q  # a goal at a joint limit would be clipped; take a shallower pitch instead
    return None


def settle(sim: Sim, goal: dict[str, float], timeout_s: float = 5.0) -> None:
    """Go to `goal` and wait until the readings stop moving (or `timeout_s`)."""
    sim.set_goals(goal)
    last = None
    for _ in range(int(timeout_s / 0.1)):
        sim.step(0.1)
        now = np.array(list(sim.joints().values()))
        if last is not None and np.abs(now - last).max() < 0.02:
            break
        last = now


def mount_errors(sim: Sim, fits: dict[str, calib.Fit]) -> dict:
    t = sim.truth()
    true_mount = np.linalg.inv(Arm().gripper_pose(t["joints"])) @ true_camera(t)
    out = {"nominal": calib.mount_error(Arm().mount, true_mount)}
    for k, f in fits.items():
        out[k] = calib.mount_error(f.mount, true_mount)
    return out
