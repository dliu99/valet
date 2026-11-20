"""Drive the SO-101 one guarded command at a time, straight on the Feetech bus.

    python -m valet.hardware.direct state
    python -m valet.hardware.direct hold                    # goal := present, torque on
    python -m valet.hardware.direct movej 0 -60 60 60 -90   # pan lift elbow wrist roll, degrees
    python -m valet.hardware.direct nudge elbow_flex -5
    python -m valet.hardware.direct xyz 0.25 0.0 0.10 --pitch 90 --roll 0
    python -m valet.hardware.direct grip 60
    python -m valet.hardware.direct relax                   # torque off: the arm drops

Every command prints one JSON line with the joints, the grasp point and the motor currents.
Servos keep holding their last goal between commands. A motion stops, and holds where it is,
when a motor draws stall-like current for a moment or falls far behind its goal.

Angles are LeRobot's degrees (zero at the middle of each joint's calibrated range), so the
Cartesian numbers are only as good as the calibration: check them against the camera.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:5]
AMPS_PER_UNIT = 0.0065
PORT = "/dev/tty.usbmodem5B901520711"
URDF = "SO101/so101_new_calib.urdf"
FRAME = Path("runs/live/wrist.jpg")


class Stopped(RuntimeError):
    pass


class Arm:
    def __init__(self, port: str = PORT, robot_id: str = "valet_follower", urdf: str = URDF):
        from lerobot.model import RobotKinematics
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.bus = SO101Follower(SO101FollowerConfig(port=port, id=robot_id)).bus
        self.kin = RobotKinematics(urdf_path=urdf, target_frame_name="gripper_frame_link", joint_names=MOTORS)
        self.bus.connect()
        cal = self.bus.read_calibration()
        # Half the calibrated travel, in degrees, less a margin: goals stay inside the hard stops.
        self.limit = {m: (cal[m].range_max - cal[m].range_min) / 2 * 360 / 4095 - 3.0 for m in ARM}

    def close(self) -> None:
        self.bus.disconnect(disable_torque=False)

    # ---- reading

    def q(self) -> np.ndarray:
        p = self.bus.sync_read("Present_Position")
        return np.array([p[m] for m in MOTORS], float)

    def amps(self) -> np.ndarray:
        c = self.bus.sync_read("Present_Current", normalize=False)
        return np.array([c[m] for m in MOTORS], float) * AMPS_PER_UNIT

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.kin.forward_kinematics(np.asarray(q, float))

    def state(self, note: str = "") -> dict:
        q = self.q()
        T = self.fk(q)
        torque = self.bus.sync_read("Torque_Enable", normalize=False)
        return {
            "joints": {m: round(float(v), 1) for m, v in zip(MOTORS, q)},
            "xyz": [round(float(v), 3) for v in T[:3, 3]],
            "pitch": round(float(np.degrees(np.arcsin(-T[2, 2]))), 1),  # 90 = pointing straight down
            "amps": [round(float(a), 2) for a in self.amps()],
            "torque": [int(torque[m]) for m in MOTORS],
            "note": note,
        }

    # ---- moving

    def hold(self) -> None:
        """Torque on with every goal at the present position, so nothing jumps."""
        q = self.q()
        self.bus.sync_write("Goal_Position", {m: float(v) for m, v in zip(MOTORS, q)})
        time.sleep(0.05)
        self.bus.enable_torque()

    def relax(self) -> None:
        self.bus.disable_torque()

    def movej(self, target: dict[str, float], speed: float = 25.0, max_amps: float = 1.6, lag_deg: float = 14.0) -> None:
        """Straight line in joint space at `speed` deg/s for the slowest joint."""
        q0 = self.q()
        q1 = q0.copy()
        for m, v in target.items():
            i = MOTORS.index(m)
            q1[i] = float(np.clip(v, 0.0, 100.0)) if m == "gripper" else float(np.clip(v, -self.limit[m], self.limit[m]))
        steps = max(1, int(np.max(np.abs(q1 - q0)[:5]) / speed * 30))
        moving = [i for i, m in enumerate(MOTORS) if m in target]
        over = 0
        for k in range(1, steps + 1):
            t0 = time.monotonic()
            s = k / steps
            s = s * s * (3 - 2 * s)  # ease in and out
            goal = q0 + (q1 - q0) * s
            self.bus.sync_write("Goal_Position", {MOTORS[i]: float(goal[i]) for i in moving})
            amps, now = np.abs(self.amps()), self.q()
            over = over + 1 if np.max(amps[:5]) > max_amps else 0
            lag = max((abs(now[i] - goal[i]) for i in moving if i < 5), default=0.0)
            if over >= 12 or lag > lag_deg:
                self.bus.sync_write("Goal_Position", {m: float(v) for m, v in zip(ARM, now[:5])})
                raise Stopped(f"stopped at step {k}/{steps}: amps {np.round(amps, 2).tolist()}, lag {np.round(now - goal, 1).tolist()}")
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t0)))
        time.sleep(0.4)

    def solve(self, xyz, pitch_deg: float, roll: float, seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Joints that put the grasp point at `xyz` with the gripper pitched `pitch_deg` below
        the horizon (90 = straight down), pointing away from the base. Returns (joints, mm error)."""
        xyz = np.asarray(xyz, float)
        yaw = np.arctan2(xyz[1], xyz[0])
        p = np.radians(pitch_deg)
        want_dir = np.array([np.cos(p) * np.cos(yaw), np.cos(p) * np.sin(yaw), -np.sin(p)])

        def resid(q4: np.ndarray) -> np.ndarray:
            T = self.fk(np.array([*q4, roll, 0.0]))
            return np.concatenate([(T[:3, 3] - xyz) * 1000.0, (T[:3, 2] - want_dir) * 60.0])

        best, best_cost = None, np.inf
        seeds = [np.array(seed[:4], float), np.array([np.degrees(yaw), -30.0, 30.0, 60.0]), np.array([np.degrees(yaw), 0.0, 0.0, 80.0])]
        lim = np.array([self.limit[m] for m in ARM[:4]])
        for q4 in seeds:
            q4 = q4.copy()
            for _ in range(80):
                r = resid(q4)
                J = np.stack([(resid(q4 + np.eye(4)[i] * 0.5) - r) / 0.5 for i in range(4)], axis=1)
                dq = np.linalg.solve(J.T @ J + 1e-2 * np.eye(4), -J.T @ r)
                q4 = np.clip(q4 + np.clip(dq, -15, 15), -lim, lim)
                if np.linalg.norm(dq) < 1e-3:
                    break
            cost = float(np.linalg.norm(resid(q4)))
            if cost < best_cost:
                best, best_cost = q4, cost
        q = np.array([*best, roll, 0.0])
        return q, float(np.linalg.norm(self.fk(q)[:3, 3] - xyz) * 1000.0)

    def move_xyz(self, xyz, pitch_deg: float = 90.0, roll: float | None = None, speed: float = 25.0, tol_mm: float = 6.0, z_min: float = 0.0) -> float:
        if xyz[2] < z_min:
            raise Stopped(f"z={xyz[2]:.3f} is below the floor {z_min:.3f}; nothing moved")
        q0 = self.q()
        roll = q0[4] if roll is None else roll
        q1, err = self.solve(xyz, pitch_deg, roll, q0)
        if err > tol_mm:
            raise Stopped(f"not reachable at pitch {pitch_deg}: closest is {err:.0f} mm away; nothing moved")
        self.movej(dict(zip(ARM, map(float, q1[:5]))), speed=speed)
        return err

    def grip(self, opening: float, max_amps: float = 0.9) -> float:
        """Open or close to `opening` (0..100). Closing stops early once the fingers squeeze."""
        g0 = self.q()[5]
        for g in np.linspace(g0, opening, max(2, int(abs(opening - g0) / 2.5))):
            self.bus.write("Goal_Position", "gripper", float(g))
            time.sleep(0.04)
            if opening < g0 and abs(self.amps()[5]) > max_amps:
                now = self.q()[5]
                self.bus.write("Goal_Position", "gripper", float(max(0.0, now - 6.0)))  # keep a light squeeze
                break
        time.sleep(0.4)
        return float(self.q()[5])


def snapshot(to: str | None) -> str | None:
    """Copy the newest camera frame (see camfeed.py) once one newer than now has landed."""
    if not to:
        return None
    t0 = time.time()
    while time.time() - t0 < 3.0 and (not FRAME.exists() or FRAME.stat().st_mtime < t0 + 0.25):
        time.sleep(0.05)
    Path(to).parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    # The camera sits on top of the gripper upside down: turned half a turn, with the wrist
    # roll at -90, picture-up is away from the base and picture-right is the arm's right.
    Image.open(FRAME).rotate(180).save(to, quality=90)
    return to


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="valet.hardware.direct")
    p.add_argument("--snap", help="save the wrist frame here afterwards")
    p.add_argument("--speed", type=float, default=25.0, help="deg/s")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("state")
    sub.add_parser("hold")
    sub.add_parser("relax")
    mj = sub.add_parser("movej")
    mj.add_argument("q", type=float, nargs=5)
    nd = sub.add_parser("nudge")
    nd.add_argument("pairs", nargs="+", help="joint delta [joint delta ...]")
    xy = sub.add_parser("xyz")
    xy.add_argument("xyz", type=float, nargs=3)
    xy.add_argument("--pitch", type=float, default=90.0)
    xy.add_argument("--roll", type=float)
    xy.add_argument("--plan", action="store_true", help="solve and print, do not move")
    gr = sub.add_parser("grip")
    gr.add_argument("opening", type=float)
    a = p.parse_args(argv)

    arm = Arm()
    note = ""
    try:
        if a.cmd == "hold":
            arm.hold()
        elif a.cmd == "relax":
            arm.relax()
        elif a.cmd == "movej":
            arm.movej(dict(zip(ARM, a.q)), speed=a.speed)
        elif a.cmd == "nudge":
            q = dict(zip(MOTORS, arm.q()))
            arm.movej({m: q[m] + float(d) for m, d in zip(a.pairs[::2], a.pairs[1::2])}, speed=a.speed)
        elif a.cmd == "xyz":
            if a.plan:
                q0 = arm.q()
                q1, err = arm.solve(a.xyz, a.pitch, q0[4] if a.roll is None else a.roll, q0)
                note = f"plan {np.round(q1[:5], 1).tolist()} err {err:.1f} mm"
            else:
                note = f"ik err {arm.move_xyz(a.xyz, a.pitch, a.roll, speed=a.speed):.1f} mm"
        elif a.cmd == "grip":
            note = f"gripper at {arm.grip(a.opening):.1f}"
    except Stopped as e:
        note = f"STOPPED: {e}"
    out = arm.state(note)
    arm.close()
    out["frame"] = snapshot(a.snap)
    print(json.dumps(out))
    return 1 if note.startswith("STOPPED") else 0


if __name__ == "__main__":
    sys.exit(main())
