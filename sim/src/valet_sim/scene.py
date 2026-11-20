"""The SO-101 (MuJoCo Menagerie's model) on a table with a box and a target square, built with MjSpec."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl" if os.uname().sysname == "Darwin" else "egl")

import mujoco

MODEL = Path(__file__).resolve().parent / "model" / "so101.xml"
WRIST_CAMERA = "wrist_cam"
SIDE_CAMERA = "side_cam"
GRASP_SITE = "grasp"
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
class SceneSpec:
    box: BoxSpec = field(default_factory=BoxSpec)
    target_xy: tuple[float, float] | None = (0.22, 0.15)
    target_size_m: float = 0.07
    look: Look = field(default_factory=Look)


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

