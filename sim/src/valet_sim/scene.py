"""The SO-101 (MuJoCo Menagerie's model) on a table with a box and a target square, built with MjSpec."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl" if os.uname().sysname == "Darwin" else "egl")

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parent / "model" / "so101.xml"
WRIST_CAMERA = "wrist_cam"
SIDE_CAMERA = "side_cam"
GRASP_SITE = "grasp"
TAG_Z = 0.0003
CAMERA_DECOR = ("wrist_camera_lens", "wrist_camera_pcb")
GRASP_IN_GRIPPER = (0.012, 0.0, -0.085)
NEAR_CLIP_M = 0.004
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@dataclass(frozen=True)
class BoxSpec:
    x: float = 0.28
    y: float = 0.0
    yaw_deg: float = 0.0
    w: float = 0.05
    d: float = 0.05
    h: float = 0.08
    mass_kg: float = 0.03
    friction: float = 0.8
    rgba: tuple[float, float, float, float] = (0.96, 0.96, 0.97, 1.0)


@dataclass(frozen=True)
class Look:
    """The wrist camera is fitted to the real arm's: the fixed jaw's outline in real frames (jaw IoU 0.16 -> 0.64
    over Menagerie's mount), held level at wrist roll -90 and within 25 degrees of the jaw axis."""

    wrist_pos: tuple[float, float, float] = (0.00831, 0.05303, -0.03891)
    wrist_quat: tuple[float, float, float, float] = (0.977626, -0.203913, -0.044585, -0.026063)
    wrist_fovy_deg: float = 83.6
    arm_rgba: tuple[float, float, float, float] = (0.93, 0.93, 0.91, 1.0)
    arm_glow: float = 0.3
    table_rgb: tuple[float, float, float] = (0.6, 0.57, 0.52)
    target_rgba: tuple[float, float, float, float] = (0.15, 0.35, 0.85, 1.0)
    headlight: float = 0.35
    ambient: float = 0.4
    light: float = 0.45


@dataclass(frozen=True)
class TagSpec:
    """A printed AprilTag (family 36h11) lying flat on the table: opt-in, for experiments that want a metric
    reference in the wrist camera's view. `size_m` is the black square's edge, as AprilTag counts it; one white cell
    surrounds it, as on paper. At yaw 0 the printed image's top faces +x (away from the base). `tag_corners` gives
    its corners in the world, in the order OpenCV's detector reports them.

    The default place is beside the base on the arm's right, clear of every box and target the benchmark draws: the
    start pose sees it in 19 of the 20 benchmark scenes, a look straight down at the box or target in about half."""

    id: int = 0
    size_m: float = 0.05
    x: float = 0.14
    y: float = -0.23
    yaw_deg: float = 0.0
    px_per_cell: int = 32


@dataclass(frozen=True)
class SceneSpec:
    box: BoxSpec = field(default_factory=BoxSpec)
    target_xy: tuple[float, float] | None = (0.22, 0.15)
    target_size_m: float = 0.07
    look: Look = field(default_factory=Look)
    tag: TagSpec | None = None


def tag_corners(tag: TagSpec) -> np.ndarray:
    """The black square's corners in the world, in the order OpenCV's detector reports them: the printed image's
    top-left, top-right, bottom-right, bottom-left. The printed image's top faces +x (away from the base) at yaw 0."""
    s = tag.size_m / 2
    local = np.array([[-s, -s], [s, -s], [s, s], [-s, s]])  # tag x right, y down (OpenCV's tag frame)
    yaw = math.radians(tag.yaw_deg)
    # At yaw 0 the image's right is the world's -y and its down is the world's -x.
    right, down = np.array([math.sin(yaw), -math.cos(yaw)]), np.array([-math.cos(yaw), -math.sin(yaw)])
    xy = np.array([tag.x, tag.y]) + local[:, :1] * right + local[:, 1:] * down
    return np.column_stack([xy, np.full(4, TAG_Z)])


def tag_image(tag: TagSpec) -> np.ndarray:
    """The printed tag as an RGB image, white margin included."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, tag.id, 8 * tag.px_per_cell, borderBits=1)
    page = np.pad(marker, tag.px_per_cell, constant_values=255)
    return np.repeat(page[..., None], 3, axis=2)


def build(scene: SceneSpec = SceneSpec()) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL))
    look = scene.look
    for mat in spec.materials:
        if mat.rgba[0] > 0.9 and mat.rgba[2] < 0.2:
            mat.rgba = list(look.arm_rgba)
            mat.emission = look.arm_glow
    for g in spec.geoms:
        if g.name in CAMERA_DECOR:
            g.rgba = [0, 0, 0, 0]
    cam = next(c for c in spec.cameras if c.name == WRIST_CAMERA)
    cam.pos, cam.quat, cam.fovy = list(look.wrist_pos), list(look.wrist_quat), look.wrist_fovy_deg

    gripper = next(b for b in spec.bodies if b.name == "gripper")
    gripper.add_site(name=GRASP_SITE, pos=list(GRASP_IN_GRIPPER), size=[0.004, 0, 0], rgba=[1, 0, 0, 0], group=3)

    spec.visual.headlight.diffuse = [look.headlight] * 3
    spec.visual.headlight.ambient = [look.ambient] * 3
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.quality.shadowsize = 2048

    spec.add_texture(name="table", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.08, width=512, height=512,
                     rgb1=list(look.table_rgb), rgb2=list(look.table_rgb), markrgb=[c * 0.9 for c in look.table_rgb])
    table = spec.add_material(name="table", texrepeat=[8, 8], texuniform=True)
    table.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table"

    world = spec.worldbody
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], material="table")
    world.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 1.5, 0.6], pos=[0.9, 0, 0.6],
                   rgba=[0.75, 0.75, 0.73, 1], contype=0, conaffinity=0)
    world.add_light(name="ceiling", pos=[0.3, 0.1, 1.2], dir=[0, 0, -1], diffuse=[look.light] * 3, castshadow=True)
    world.add_light(name="window", pos=[-0.4, 0.6, 0.8], dir=[0.5, -0.6, -0.6], diffuse=[look.light * 0.5] * 3, castshadow=False)
    if scene.target_xy is not None:
        s = scene.target_size_m / 2
        world.add_geom(name="target", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, 0.0005],
                       pos=[*scene.target_xy, 0.0005], rgba=list(look.target_rgba), contype=0, conaffinity=0)

    if scene.tag is not None:
        _add_tag(spec, scene.tag)

    b = scene.box
    yaw = math.radians(b.yaw_deg)
    body = world.add_body(name="box", pos=[b.x, b.y, b.h / 2], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    body.add_freejoint(name="box")
    body.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b.w / 2, b.d / 2, b.h / 2], mass=b.mass_kg,
                  rgba=list(b.rgba), friction=[b.friction, 0.01, 0.001], condim=4, solref=[0.01, 1])
    world.add_camera(name=SIDE_CAMERA, pos=[0.25, -0.55, 0.3], xyaxes=[1, 0, 0, 0, 0.45, 1], fovy=55)
    model = spec.compile()
    model.vis.map.znear = NEAR_CLIP_M / model.stat.extent
    return model



def _add_tag(spec: mujoco.MjSpec, tag: TagSpec) -> None:
    image = tag_image(tag)
    h, w = image.shape[:2]
    tex = spec.add_texture(name="tag", type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3)
    tex.data = np.ascontiguousarray(image.transpose(1, 0, 2)[:, ::-1]).tobytes()  # MuJoCo maps it turned a quarter
    mat = spec.add_material(name="tag", texrepeat=[1, 1], texuniform=False, specular=0.0, shininess=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tag"
    half = tag.size_m / 2 * 10 / 8
    yaw = math.radians(tag.yaw_deg)
    spec.worldbody.add_geom(name="tag", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[half, half, 0.01], pos=[tag.x, tag.y, TAG_Z],
                            quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], material="tag", contype=0, conaffinity=0)
