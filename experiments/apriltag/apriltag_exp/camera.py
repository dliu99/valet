"""What a wrist-only policy knows about its camera: the arm's nominal kinematics (Menagerie's model, as the URDF
gives it), the nominal mount in `Look`, and the lens model in `Optics`. Never the sim's true pose.

Frames: world is the sim's base frame. Camera poses are OpenCV's (x right, y down, z out of the lens), as 4x4
camera-to-world transforms. Pixels are OpenCV index coordinates of the 640x360 upright frame.

The lens: `develop` resamples the pinhole render so that a frame pixel at centred offset p shows the pinhole point
p * (1 + k1 |p / (W/2)|^2). Undistorting is that formula; distorting inverts it by Newton's method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from valet_sim.scene import GRASP_SITE, MOTORS, WRIST_CAMERA, Look, SceneSpec, build
from valet_sim.sim import Optics

MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Lens:
    width: int = 640
    height: int = 360
    fovy_deg: float = Look().wrist_fovy_deg
    k1: float = Optics().barrel_k1

    @property
    def f(self) -> float:
        return self.height / 2 / math.tan(math.radians(self.fovy_deg) / 2)

    def undistort(self, uv) -> np.ndarray:
        """Frame pixels -> normalised pinhole coordinates (x/z, y/z)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        p = uv - [self.width / 2, self.height / 2]
        r2 = (p ** 2).sum(1, keepdims=True) / (self.width / 2) ** 2
        return p * (1 + self.k1 * r2) / self.f

    def distort(self, xy) -> np.ndarray:
        """Normalised pinhole coordinates -> frame pixels; NaN for points too far out for the lens to show (with
        k1 < 0, r (1 + k1 r^2) peaks at r = 1/sqrt(-3 k1), beyond the frame's corners)."""
        xy = np.atleast_2d(np.asarray(xy, float))
        target = np.linalg.norm(xy, axis=1) * self.f / (self.width / 2)
        r_peak = 1 / np.sqrt(-3 * self.k1) if self.k1 < 0 else np.inf
        r = np.minimum(target, r_peak)
        for _ in range(30):
            slope = np.maximum(1 + 3 * self.k1 * r * r, 1e-9)
            r = np.clip(r - (r * (1 + self.k1 * r * r) - target) / slope, 0, r_peak)
        scale = np.where(target > 1e-12, r / np.maximum(target, 1e-12), 1.0)
        out = xy * self.f * scale[:, None] + [self.width / 2, self.height / 2]
        out[np.abs(r * (1 + self.k1 * r * r) - target) > 1e-6] = np.nan
        return out

    def project(self, T_world_cam: np.ndarray, points) -> np.ndarray:
        """World points -> frame pixels (NaN behind the camera)."""
        P = np.atleast_2d(np.asarray(points, float))
        c = (P - T_world_cam[:3, 3]) @ T_world_cam[:3, :3]
        out = self.distort(c[:, :2] / c[:, 2:3])
        out[c[:, 2] <= 1e-6] = np.nan
        return out

    def ray(self, T_world_cam: np.ndarray, uv) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction in the world of the ray through a frame pixel."""
        n = self.undistort(uv)[0]
        d = T_world_cam[:3, :3] @ np.array([n[0], n[1], 1.0])
        return T_world_cam[:3, 3].copy(), d / np.linalg.norm(d)

    def visible(self, uv, margin: float = 0.0) -> bool:
        uv = np.atleast_2d(uv)
        return bool(np.all(np.isfinite(uv)) and np.all((uv[:, 0] >= margin) & (uv[:, 0] <= self.width - 1 - margin)
                                                       & (uv[:, 1] >= margin) & (uv[:, 1] <= self.height - 1 - margin)))


def on_plane(origin: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray | None:
    if direction[2] >= -1e-9:
        return None
    return origin + (z - origin[2]) / direction[2] * direction


def triangulate(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """The point closest, in least squares, to every ray."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def pose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rotvec(v) -> np.ndarray:
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


@dataclass
class Arm:
    """The arm as its model says it is. `offsets_deg`, `compliance_deg_per_nm` and `mount` are corrections a
    calibration may supply: the joint zero errors to subtract from readings, how far each joint bends past its
    reading per N m of gravity load, and the camera's pose in the gripper frame (OpenCV axes)."""

    offsets_deg: np.ndarray = field(default_factory=lambda: np.zeros(len(MOTORS)))
    mount: np.ndarray | None = None
    compliance_deg_per_nm: float = 0.0

    def __post_init__(self):
        self.model = build(SceneSpec())
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr, self.dofadr = np.array([m.jnt_qposadr[j] for j in jid]), np.array([m.jnt_dofadr[j] for j in jid])
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
        self.gripper = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        if self.mount is None:
            self.mount = self.nominal_mount()

    def nominal_mount(self) -> np.ndarray:
        """The camera in the gripper frame, from `Look`."""
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qadr] = 0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        d = self.data
        T_cam = pose(d.cam_xmat[self.cam].reshape(3, 3) @ MJ_TO_CV, d.cam_xpos[self.cam])
        T_grip = pose(d.xmat[self.gripper].reshape(3, 3), d.xpos[self.gripper])
        return np.linalg.inv(T_grip) @ T_cam

    def _fk(self, joints: dict[str, float]) -> None:
        q = np.radians(np.array([joints[j] for j in MOTORS]) - self.offsets_deg)
        if self.compliance_deg_per_nm:
            q = q - np.radians(self.compliance_deg_per_nm) * self._gravity(q)
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _gravity(self, q: np.ndarray) -> np.ndarray:
        """The torque each servo holds against gravity at joint angles q (radians)."""
        d = self.data
        d.qpos[self.qadr] = q
        d.qvel[:] = 0
        mujoco.mj_fwdPosition(self.model, d)
        mujoco.mj_fwdVelocity(self.model, d)
        return d.qfrc_bias[self.dofadr].copy()

    def gripper_pose(self, joints: dict[str, float]) -> np.ndarray:
        self._fk(joints)
        return pose(self.data.xmat[self.gripper].reshape(3, 3).copy(), self.data.xpos[self.gripper].copy())

    def camera(self, joints: dict[str, float]) -> np.ndarray:
        """Camera-to-world (OpenCV axes) for these joint readings."""
        return self.gripper_pose(joints) @ self.mount

    def grasp_point(self, joints: dict[str, float]) -> np.ndarray:
        self._fk(joints)
        return self.data.site_xpos[self.site].copy()

    def model_joints(self, readings: dict[str, float]) -> dict[str, float]:
        """Readings -> the joint angles the model should use (what `_fk` applies): the inverse of `command`."""
        q = np.radians(np.array([readings[j] for j in MOTORS]) - self.offsets_deg)
        if self.compliance_deg_per_nm:
            q = q - np.radians(self.compliance_deg_per_nm) * self._gravity(q)
        return {j: float(v) for j, v in zip(MOTORS, np.degrees(q))}

    def command(self, joints: dict[str, float]) -> dict[str, float]:
        """Joints from the nominal IK (where the links should be) -> the goals to send, given the zero errors and
        sag this arm was calibrated with. The inverse of what `_fk` does to readings."""
        q = np.array([joints[j] for j in MOTORS], float)
        if self.compliance_deg_per_nm:
            q = q + self.compliance_deg_per_nm * self._gravity(np.radians(q))
        return {j: float(q[i] + self.offsets_deg[i]) for i, j in enumerate(MOTORS)}


def true_camera(truth: dict) -> np.ndarray:
    """The sim's true wrist camera pose (for measuring error only)."""
    cam = truth["wrist_camera"]
    return pose(np.array(cam["xmat"]).reshape(3, 3) @ MJ_TO_CV, cam["pos"])
