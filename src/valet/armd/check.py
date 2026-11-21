from __future__ import annotations

from typing import Callable

import numpy as np

from .app import Armd
from .kinematics import ARM, GRIPPER, ROLL

VOLTAGE_RANGE_V = (6.0, 13.0)
CALIBRATED_RANGE_MARGIN_DEG = 3.0
READS_AGREE_DEG = 2.0
PLACO_AGREES_M = 0.001
NUDGE_DEG = 2.0
NUDGE_SPEED_DPS = 5.0
NUDGE_TRACKS_DEG = 1.0
NUDGE_MAX_AMPS = 0.5
CHECK_MAX_SIDE = 320


class CheckFailed(Exception):
    pass


def bus_and_voltage(app: Armd) -> str:
    volts = app.guard.voltage()
    lo, hi = VOLTAGE_RANGE_V
    if volts.min() < lo or volts.max() > hi:
        raise CheckFailed(f"supply reads {volts.min():.1f}-{volts.max():.1f} V, outside {lo}-{hi} V")
    return f"{app.kind}, {volts.min():.1f}-{volts.max():.1f} V"


def state_matches_calibration(app: Armd) -> str:
    a, b = app.guard.read(), app.guard.read()
    if np.max(np.abs(a.q - b.q)) > READS_AGREE_DEG:
        raise CheckFailed(f"two reads disagree by {np.max(np.abs(a.q - b.q)):.1f}°")
    for m, v in zip(ARM, a.q):
        calibrated = app.guard.limit[m] + CALIBRATED_RANGE_MARGIN_DEG
        if abs(v) > calibrated:
            raise CheckFailed(f"{m} reads {v:.1f}°, outside its calibrated ±{calibrated:.0f}°")
    if not -5.0 <= a.q[GRIPPER] <= 105.0:
        raise CheckFailed(f"gripper reads {a.q[GRIPPER]:.1f}, outside 0-100")
    tip = app.kin.tip(a.q)[:3, 3]
    placo_fk = getattr(app.guard.servos, "fk", None)
    if placo_fk is not None and np.linalg.norm(placo_fk(a.q)[:3, 3] - tip) > PLACO_AGREES_M:
        raise CheckFailed("armd's kinematics and placo's disagree about the grasp point")
    return "joints " + " ".join(f"{v:.1f}" for v in a.q) + f", grasp point ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m"


def cameras_fresh_and_upright(app: Armd) -> str:
    got = app.frames(["wrist", "side"], CHECK_MAX_SIDE)
    if got["missing"]:
        raise CheckFailed("; ".join(f"{cam}: {why}" for cam, why in got["missing"].items()))
    if not got["frames"]["wrist"]["upright"]:
        raise CheckFailed(f"wrist roll {app.guard.read().q[ROLL]:.0f}° is not the roll at which upright was verified")
    return ", ".join(f"{cam} {f['age_ms']:.0f} ms old" for cam, f in got["frames"].items())


def hold_engages(app: Armd) -> str:
    if not app.hold()["torque"]:
        raise CheckFailed("torque reads off after hold")
    return "torque on, goal = present on every joint"


def nudge_each_joint(app: Armd) -> str:
    if not app.settings.allow_motion:
        return "skipped: allow_motion = false"
    worst_track, worst_amps = 0.0, 0.0
    for i, m in enumerate(ARM):
        home = app.guard.goal.copy()
        direction = 1.0 if home[i] + NUDGE_DEG <= app.guard.limit[m] else -1.0
        target = home.copy()
        target[i] += direction * NUDGE_DEG
        out = app.mover.move_joints(target, NUDGE_SPEED_DPS)
        track = abs(app.guard.read().q[i] - target[i])
        amps = float(app.guard.peak[i])
        if not out.ok:
            raise CheckFailed(f"{m}: {out.stopped}")
        if track > NUDGE_TRACKS_DEG:
            raise CheckFailed(f"{m} ended {track:.1f}° from its goal")
        if amps > NUDGE_MAX_AMPS:
            raise CheckFailed(f"{m} drew {amps:.2f} A")
        back = app.mover.move_joints(home, NUDGE_SPEED_DPS)
        if not back.ok:
            raise CheckFailed(f"{m} returning: {back.stopped}")
        worst_track, worst_amps = max(worst_track, track), max(worst_amps, amps)
    return f"every joint tracked within {worst_track:.2f}°, at most {worst_amps:.2f} A"


STEPS = [
    ("bus", bus_and_voltage),
    ("state", state_matches_calibration),
    ("cameras", cameras_fresh_and_upright),
    ("hold", hold_engages),
    ("nudge", nudge_each_joint),
]


def run_check(open_app: Callable[[], Armd], say: Callable[[str], None] = print) -> bool:
    try:
        app = open_app()
    except Exception as e:
        say(f"1 bus: FAIL {type(e).__name__}: {e}")
        return False
    try:
        for n, (name, step) in enumerate(STEPS, 1):
            try:
                detail = step(app)
            except CheckFailed as e:
                say(f"{n} {name}: FAIL {e}")
                return False
            say(f"{n} {name}: ok, {detail}")
        return True
    finally:
        app.close()
