"""The SO-101 follower arm with one wrist camera, through LeRobot.

Untested against hardware. Before the first run, on the real arm:
  1. Calibrate it with LeRobot (`lerobot-calibrate`) under the same `id`.
  2. Download so101_new_calib.urdf from TheRobotStudio/SO-ARM100 (Simulation/SO101) and set `urdf`.
  3. Pose the arm by hand with the gripper pointing straight down, run `valet pose`, and copy the
     joints into `down_joints`: every move keeps that orientation, turned by the requested yaw.
  4. Calibrate the camera intrinsics (OpenCV checkerboard) and measure where the camera sits on the
     gripper (`ee_to_cam_*`); check with `valet locate` against a ruler.
  5. Set `allow_motion = true`. Without it the arm reads and plans but refuses every motion.

LeRobot's `max_relative_target` clips each command's joint change, and every target is checked
against the workspace box before anything moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..robot import Camera, Move, Observation, Vec3, Workspace

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER = MOTORS.index("gripper")


class MotionNotAllowed(RuntimeError):
    pass


@dataclass
class SO101Config:
    port: str
    id: str = "valet_follower"
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    allow_motion: bool = False
    max_relative_target: float | None = 10.0  # degrees per command
    speed_m_s: float = 0.05
    fps: float = 30.0
    tolerance_mm: float = 8.0
    down_joints: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 90.0, 0.0, 0.0])
    home_joints: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    orientation_weight: float = 0.1  # soft: a 5-joint arm cannot reach every orientation everywhere
    camera_key: str = "wrist"  # must match the camera name of any policy you run
    camera_index: int | str = 0
    camera: Camera = field(default_factory=lambda: Camera(640, 480, 500.0, 500.0, 320.0, 240.0))
    camera_fps: int = 30
    ee_to_cam_xyz: Vec3 = (0.0, -0.05, 0.03)  # camera position in the grasp frame, metres
    ee_to_cam_rpy_deg: Vec3 = (0.0, 0.0, 0.0)
    workspace: Workspace = field(default_factory=lambda: Workspace((0.05, -0.30, -0.01), (0.40, 0.30, 0.35)))


class SO101Robot:
    def __init__(self, cfg: SO101Config):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.model import RobotKinematics
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.cfg = cfg
        self.camera = cfg.camera
        self.workspace = cfg.workspace
        cams = {
            cfg.camera_key: OpenCVCameraConfig(
                index_or_path=cfg.camera_index, width=cfg.camera.width, height=cfg.camera.height, fps=cfg.camera_fps
            )
        }
        self.lerobot = SO101Follower(
            SO101FollowerConfig(port=cfg.port, id=cfg.id, cameras=cams, max_relative_target=cfg.max_relative_target)
        )
        self.kin = RobotKinematics(urdf_path=cfg.urdf, target_frame_name=cfg.ee_frame, joint_names=MOTORS)
        self.T_ee_cam = _transform(cfg.ee_to_cam_xyz, cfg.ee_to_cam_rpy_deg)
        self.R_down = self.kin.forward_kinematics(np.array(cfg.down_joints, float))[:3, :3]

    def connect(self) -> None:
        self.lerobot.connect()

    def disconnect(self) -> None:
        self.lerobot.disconnect()

    # ---- reading

    def joints(self) -> np.ndarray:
        q = self.lerobot.bus.sync_read("Present_Position")
        return np.array([q[m] for m in MOTORS], float)

    def observe(self) -> Observation:
        obs = self.lerobot.get_observation()
        q = np.array([obs[f"{m}.pos"] for m in MOTORS], float)
        T_ee = self.kin.forward_kinematics(q)
        return Observation(
            t=time.monotonic(),
            frame=obs[self.cfg.camera_key],
            T_world_cam=T_ee @ self.T_ee_cam,
            ee_xyz=tuple(float(v) for v in T_ee[:3, 3]),
            gripper=float(q[GRIPPER]),
            joints=dict(zip(MOTORS, map(float, q))),
        )

    # ---- moving

    def move_to(self, xyz: Vec3, *, yaw_deg: float = 0.0) -> Move:
        self.workspace.check(xyz)
        q0 = self.joints()
        target = np.eye(4)
        target[:3, :3] = _rot_z(yaw_deg) @ self.R_down
        target[:3, 3] = xyz
        q1 = q0.copy()
        for _ in range(6):  # LeRobot's IK takes a few Newton steps per call; chain calls for long moves
            q1 = self.kin.inverse_kinematics(q1, target, position_weight=1.0, orientation_weight=self.cfg.orientation_weight)
            if _dist_mm(self.kin.forward_kinematics(q1)[:3, 3], xyz) < 2.0:
                break
        q1[GRIPPER] = q0[GRIPPER]
        plan_err = _dist_mm(self.kin.forward_kinematics(q1)[:3, 3], xyz)
        if plan_err > self.cfg.tolerance_mm:
            here = tuple(float(v) for v in self.kin.forward_kinematics(q0)[:3, 3])
            return Move(False, here, plan_err, f"not reachable: the closest pose found is {plan_err:.0f} mm away; nothing moved")
        start = self.kin.forward_kinematics(q0)[:3, 3]
        self._go(q0, q1, seconds=np.linalg.norm(start - np.array(xyz)) / self.cfg.speed_m_s)
        at = self.kin.forward_kinematics(self.joints())[:3, 3]
        err = _dist_mm(at, xyz)
        return Move(err <= self.cfg.tolerance_mm, tuple(float(v) for v in at), err, "" if err <= self.cfg.tolerance_mm else "stopped short")

    def set_gripper(self, opening: float) -> float:
        q = self.joints()
        q[GRIPPER] = float(np.clip(opening, 0.0, 100.0))
        self._send(q)
        time.sleep(0.6)
        return float(self.joints()[GRIPPER])

    def home(self) -> Move:
        q0 = self.joints()
        q1 = np.array(self.cfg.home_joints, float)
        q1[GRIPPER] = q0[GRIPPER]
        self._go(q0, q1, seconds=2.0)
        at = tuple(float(v) for v in self.kin.forward_kinematics(self.joints())[:3, 3])
        return Move(True, at, 0.0)

    def hold(self) -> None:
        pass  # the servos hold their last goal in position mode

    def _go(self, q0: np.ndarray, q1: np.ndarray, seconds: float) -> None:
        steps = max(1, int(seconds * self.cfg.fps))
        for k in range(1, steps + 1):
            self._send(q0 + (q1 - q0) * k / steps)
            time.sleep(1.0 / self.cfg.fps)
        time.sleep(0.3)

    def _send(self, q: np.ndarray) -> None:
        self.send_action({f"{m}.pos": float(v) for m, v in zip(MOTORS, q)})

    def send_action(self, action: dict[str, float]) -> None:
        """Send one LeRobot joint action. Learned skills use this too, so the motion gate covers
        them; the workspace box does not (a policy commands joints, not targets)."""
        if not self.cfg.allow_motion:
            raise MotionNotAllowed("allow_motion is false in the config; nothing moved")
        self.lerobot.send_action(action)


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _transform(xyz: Vec3, rpy_deg: Vec3) -> np.ndarray:
    r, p, y = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    T = np.eye(4)
    T[:3, :3] = _rot_z(np.degrees(y)) @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def _dist_mm(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) * 1000.0)
