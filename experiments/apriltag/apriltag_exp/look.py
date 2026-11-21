"""Where to hold the wrist camera: above a table point, looking down at it, from a chosen height."""

from __future__ import annotations

import numpy as np
from valet_sim.ik import IK, Unreachable

from .camera import Arm, on_plane

PITCHES_DEG = (90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 50.0, 40.0)


def look_at(arm: Arm, ik: IK, xy, cam_height: float = 0.20, gripper: float = 60.0, seed: dict | None = None,
            roll_deg: float = -90.0, pitches=PITCHES_DEG) -> dict[str, float]:
    """Joints (as readings) that point the camera's optical axis at table point `xy` from about `cam_height` above
    it, as close to straight down as the arm reaches. Uses the arm's model, so it is only as right as that is."""
    target = np.array([xy[0], xy[1], 0.0])
    best = None
    for pitch in pitches:
        site = np.array([xy[0] - 0.03, xy[1], cam_height - 0.06])
        q = None
        try:
            for _ in range(6):
                q = ik.solve(site, pitch, roll_deg, seed=q or seed, gripper=gripper)
                T = arm.camera(arm.command(q))
                hit = on_plane(T[:3, 3], T[:3, 2], 0.0)
                if hit is None:
                    raise Unreachable("camera looks above the horizon")
                miss = target - hit
                dz = cam_height - T[2, 3]
                if np.hypot(*miss[:2]) < 0.003 and abs(dz) < 0.005:
                    break
                site = site + [miss[0], miss[1], dz]
        except Unreachable:
            continue
        if q is None:
            continue
        T = arm.camera(arm.command(q))
        tilt = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
        if best is None or tilt < best[0] - 1:
            best = (tilt, q)
        if tilt < 10:
            break
    if best is None:
        raise Unreachable(f"no pose looks down at {np.round(xy, 3).tolist()} from {cam_height} m")
    return arm.command(best[1])


def look_from(arm: Arm, ik: IK, cam_xyz, at_xyz, roll_deg: float = -90.0, gripper: float = 60.0) -> dict[str, float]:
    """Joints (as readings) that put the camera at `cam_xyz` with its optical axis through `at_xyz`, the wrist
    rolled about `roll_deg`. Five joints, five conditions; solved on the arm's model."""
    from scipy.optimize import least_squares
    from valet_sim.scene import MOTORS

    cam_xyz, at_xyz = np.asarray(cam_xyz, float), np.asarray(at_xyz, float)
    want = (at_xyz - cam_xyz) / np.linalg.norm(at_xyz - cam_xyz)
    free = MOTORS[:5]
    lo, hi = ik.range[:5, 0], ik.range[:5, 1]

    def joints(x):
        return {**dict(zip(free, x)), "gripper": gripper}

    def residual(x):
        T = arm.camera(arm.command(joints(x)))
        return np.r_[(T[:3, 3] - cam_xyz) * 100, np.cross(T[:3, 2], want) * 10, (x[4] - roll_deg) * 1e-3]

    best = None
    for pitch in (80.0, 60.0, 40.0, 20.0):
        try:
            q = ik.solve(cam_xyz - [0.0, 0.0, 0.05], pitch, roll_deg, gripper=gripper)
        except Unreachable:
            continue
        x0 = np.clip([q[j] for j in free], lo + 1e-3, hi - 1e-3)
        sol = least_squares(residual, x0, bounds=(lo, hi))
        T = arm.camera(arm.command(joints(sol.x)))
        if T[:3, 2] @ want > 0.99 and np.linalg.norm(T[:3, 3] - cam_xyz) < 0.005:
            cost = float(sol.cost)
            if best is None or cost < best[0]:
                best = (cost, sol.x)
    if best is None:
        raise Unreachable(f"the camera cannot sit at {np.round(cam_xyz, 3).tolist()} looking at {np.round(at_xyz, 3).tolist()}")
    return arm.command(joints(best[1]))
