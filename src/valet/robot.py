"""The robot interface the runtime drives, and the values it returns.

Positions are metres in the robot's base frame: +z up, the table at `table_z`.
The gripper always points down; `yaw_deg` turns it about the vertical axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics of the wrist camera, in pixels (OpenCV convention)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Observation:
    t: float  # time.monotonic() when the frame was read
    frame: np.ndarray  # H x W x 3, uint8, RGB
    T_world_cam: np.ndarray  # 4 x 4: camera pose in the base frame (x right, y down, z forward)
    ee_xyz: Vec3  # grasp point
    gripper: float  # measured opening, 0..100 as LeRobot normalizes it
    joints: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Move:
    ok: bool
    at: Vec3
    error_mm: float
    note: str = ""


class OutOfWorkspace(ValueError):
    """A target outside the configured workspace box. Nothing moved."""


@dataclass(frozen=True)
class Workspace:
    lo: Vec3
    hi: Vec3

    def check(self, xyz: Vec3) -> None:
        for axis, v, lo, hi in zip("xyz", xyz, self.lo, self.hi):
            if not lo <= v <= hi:
                raise OutOfWorkspace(f"{axis}={v:.3f} m is outside [{lo:.3f}, {hi:.3f}]; nothing moved")


class Robot(Protocol):
    camera: Camera
    workspace: Workspace

    def observe(self) -> Observation: ...

    def move_to(self, xyz: Vec3, *, yaw_deg: float = 0.0) -> Move:
        """Move the grasp point to `xyz`, gripper pointing down. Raises OutOfWorkspace."""
        ...

    def set_gripper(self, opening: float) -> float:
        """Command an opening (0..100) and return the measured one once it settles."""
        ...

    def home(self) -> Move: ...

    def hold(self) -> None:
        """Keep the current pose while the runtime waits (for Jev, Claude or a person)."""
        ...
