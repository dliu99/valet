"""The wrist frame comes out raw (upside down, as the real camera is mounted) so it goes through
the same upright turn as a real one."""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw

from .fake import Box, FakeArm, jaw_closing_axis_in_gripper
from .frames import Capture, NoFrame
from .kinematics import Kinematics

WIDTH, HEIGHT = 640, 480
FOCAL_PX = 400.0
CAMERA_BEHIND_GRASP_M = 0.08
GRASP_POINT_IN_UPRIGHT_IMAGE = (0.47, 0.62)
FIXED_JAW_IN_UPRIGHT_IMAGE = (0.36, 0.58, 0.42, 1.0)
NEAREST_DRAWN_M = 0.02
TABLE_RGB = (92, 78, 64)
BACKDROP_RGB = (196, 200, 204)
JAW_RGB = (128, 128, 128)
BOX_RGB = (250, 250, 250)
BOX_EDGE_RGB = (215, 215, 215)
BOX_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
SIDE_PX_PER_M = 1280.0
SIDE_VIEW_ORIGIN_XZ = (-0.05, 0.355)


def _wrist_upright(kin: Kinematics, closing_axis: np.ndarray, q: np.ndarray, box: Box) -> Image.Image:
    tip = kin.tip(q)
    ahead, right = tip[:3, 2], tip[:3, :3] @ closing_axis
    down = np.cross(ahead, right)
    u0, v0 = GRASP_POINT_IN_UPRIGHT_IMAGE
    to_grasp = CAMERA_BEHIND_GRASP_M * (ahead + (u0 - 0.5) * WIDTH / FOCAL_PX * right + (v0 - 0.5) * HEIGHT / FOCAL_PX * down)
    camera = tip[:3, 3] - to_grasp
    img = Image.new("RGB", (WIDTH, HEIGHT), TABLE_RGB)
    draw = ImageDraw.Draw(img)
    rel = box.corners() - camera
    depth = rel @ ahead
    u = WIDTH / 2 + FOCAL_PX * (rel @ right) / np.maximum(depth, NEAREST_DRAWN_M)
    v = HEIGHT / 2 + FOCAL_PX * (rel @ down) / np.maximum(depth, NEAREST_DRAWN_M)
    for face in BOX_FACES:
        if all(depth[i] > NEAREST_DRAWN_M for i in face):
            draw.polygon([(float(u[i]), float(v[i])) for i in face], fill=BOX_RGB, outline=BOX_EDGE_RGB)
    x0, y0, x1, y1 = FIXED_JAW_IN_UPRIGHT_IMAGE
    draw.rectangle([x0 * WIDTH, y0 * HEIGHT, x1 * WIDTH, y1 * HEIGHT], fill=JAW_RGB)
    return img


def _side(kin: Kinematics, q: np.ndarray, box: Box, table_z: float) -> Image.Image:
    ox, oz = SIDE_VIEW_ORIGIN_XZ

    def px(x: float, z: float) -> tuple[float, float]:
        return (x - ox) * SIDE_PX_PER_M, (oz - z) * SIDE_PX_PER_M

    img = Image.new("RGB", (WIDTH, HEIGHT), BACKDROP_RGB)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, px(0, table_z)[1], WIDTH, HEIGHT], fill=TABLE_RGB)
    frames = kin.frames(q)
    joints = [np.array([kin.pan_axis_xy[0], 0.0, table_z])] + [f[:3, 3] for f in frames]
    draw.line([px(p[0], p[2]) for p in joints], fill=JAW_RGB, width=8, joint="curve")
    tip = frames[-1]
    jaw_end = tip[:3, 3] + tip[:3, 2] * 0.02
    draw.line([px(tip[0, 3], tip[2, 3]), px(jaw_end[0], jaw_end[2])], fill=(60, 60, 60), width=4)
    draw.rectangle([*px(box.x - box.w / 2, box.z + box.h / 2), *px(box.x + box.w / 2, box.z - box.h / 2)], fill=BOX_RGB, outline=(40, 40, 40))
    return img


class FakeCameras:
    def __init__(self, kin: Kinematics, arm: FakeArm):
        self.kin = kin
        self.arm = arm
        self.closing_axis = jaw_closing_axis_in_gripper(kin)

    def capture(self, cam: str) -> Capture:
        q, box = self.arm.scene()
        if cam == "wrist":
            image = _wrist_upright(self.kin, self.closing_axis, q, box).transpose(Image.Transpose.ROTATE_180)
        elif cam == "side":
            image = _side(self.kin, q, box, self.arm.table_z)
        else:
            raise NoFrame(f"the fake has no camera {cam!r}")
        return Capture(image, time.time())
