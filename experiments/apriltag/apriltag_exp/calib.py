"""Calibrating the arm against the table tag: a short sweep of views, then a least-squares fit of the camera's
mount on the gripper (hand-eye) and, optionally, the joints' zero errors, to the tag corners seen in every view."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import TagSpec, tag_corners

from . import tags
from .camera import Arm, Lens, angle_deg, pose, rotvec
from .look import look_at, look_from

# Views from above the tag ("above": dx, dy from it, camera height, wrist roll) and from off to the side ("from":
# camera x, y, z, roll), rolled both ways and spread over heights, so that neither a joint's zero error nor the
# arm's sag can pass for a turn of the mount
SWEEP = (
    ("above", 0.00, 0.00, 0.20, -90.0), ("above", 0.04, 0.03, 0.15, -60.0), ("above", -0.03, 0.04, 0.24, -120.0),
    ("above", 0.03, -0.03, 0.24, -105.0), ("above", 0.00, 0.05, 0.27, -75.0),
    ("from", 0.16, -0.08, 0.18, -90.0), ("from", 0.20, -0.15, 0.20, -60.0), ("from", 0.12, -0.15, 0.20, -120.0),
    ("from", 0.20, -0.25, 0.12, -90.0), ("from", 0.12, -0.25, 0.20, -60.0),
)
PIXEL_SD = 0.5
OFFSET_PRIOR_DEG = 5.0
TILT_PRIOR_DEG = 5.0
SHIFT_PRIOR_M = 0.01


@dataclass
class View:
    joints: dict[str, float]
    corners_px: np.ndarray


@dataclass
class Fit:
    offsets_deg: np.ndarray
    mount: np.ndarray
    compliance_deg_per_nm: float
    rms_px: float
    views: int

    def arm(self) -> Arm:
        return Arm(offsets_deg=self.offsets_deg.copy(), mount=self.mount.copy(), compliance_deg_per_nm=self.compliance_deg_per_nm)


def sweep(sim, tag: TagSpec, arm: Arm, ik: IK, settle_s: float = 2.0) -> list[View]:
    """Drive through the sweep and keep every view that sees the tag (the policy's own model picks the poses)."""
    views = []
    for kind, a, b, c, roll in SWEEP:
        try:
            if kind == "above":
                goal = look_at(arm, ik, (tag.x + a, tag.y + b), cam_height=c, roll_deg=roll)
            else:
                goal = look_from(arm, ik, (a, b, c), (tag.x, tag.y, 0.0), roll_deg=roll)
        except Unreachable:
            continue
        sim.set_goals(goal)
        sim.step(settle_s)
        corners = tags.find(np.asarray(sim.frame("wrist")), tag)
        if corners is not None:
            views.append(View(sim.joints(), corners))
    return views


def fit(views: list[View], tag: TagSpec, with_offsets: bool = True, lens: Lens = Lens()) -> Fit:
    """Mount (and, `with_offsets`, the zero errors and the sag) that best reproject the tag corners. Mild priors
    keep what the views cannot see (the roll joint's zero trades exactly against a turn of the mount about the roll
    axis) near nominal."""
    nominal = Arm().mount
    world = tag_corners(tag)
    n_off = 6 if with_offsets else 0

    def unpack(p):
        offsets = np.zeros(6)
        offsets[:min(n_off, 5)] = p[:min(n_off, 5)]
        compliance = float(p[5]) if n_off else 0.0
        mount = nominal @ pose(rotvec(np.radians(p[n_off:n_off + 3])), p[n_off + 3:n_off + 6])
        return offsets, mount, compliance

    arm = Arm()

    def residuals(p):
        offsets, mount, arm.compliance_deg_per_nm = unpack(p)
        arm.offsets_deg, arm.mount = offsets, mount
        r = [((lens.project(arm.camera(v.joints), world) - v.corners_px) / PIXEL_SD).ravel() for v in views]
        r = np.concatenate(r)
        r = np.nan_to_num(r, nan=1e3)
        prior = np.concatenate([p[:n_off] / OFFSET_PRIOR_DEG, p[n_off:n_off + 3] / TILT_PRIOR_DEG, p[n_off + 3:] / SHIFT_PRIOR_M])
        return np.concatenate([r, prior])

    # lift, elbow and flex zeros trade against each other, so start from each corner of their likely range too
    starts = [np.zeros(n_off + 6)]
    if with_offsets:
        for signs in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            x0 = np.zeros(n_off + 6)
            x0[1:4] = 4.0 * signs
            starts.append(x0)
    scale = np.r_[np.ones(n_off), np.ones(3), np.full(3, 0.005)]
    sol = min((least_squares(residuals, x0, x_scale=scale, loss="soft_l1", f_scale=3.0) for x0 in starts), key=lambda r: r.cost)
    offsets, mount, compliance = unpack(sol.x)
    arm.offsets_deg, arm.mount, arm.compliance_deg_per_nm = offsets, mount, compliance
    px = np.concatenate([(lens.project(arm.camera(v.joints), world) - v.corners_px).ravel() for v in views])
    return Fit(offsets, mount, compliance, float(np.sqrt(np.nanmean(px ** 2))), len(views))


def mount_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(mm, deg) between two camera-in-gripper poses."""
    d = np.linalg.inv(truth) @ estimate
    return float(np.linalg.norm(d[:3, 3]) * 1000), angle_deg(d[:3, :3])
