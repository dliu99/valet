"""The simulated SO-101 on a table with the white box, built with MjSpec so every quantity a sim-to-real
fit may change lives in one dataclass."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco

ROOT = Path(__file__).resolve().parents[3]
ARM_XML = ROOT / "sim" / "src" / "valet_sim" / "model" / "so101.xml"
DESK_TEXTURE = ROOT / "SO101" / "textures" / "desk.png"
WRIST_CAMERA = "wrist_cam"
SIDE_CAMERA = "side_cam"
CAMERA_PARTS = ("wrist_camera_lens", "wrist_camera_pcb")


@dataclass(frozen=True)
class BoxSpec:
    x: float = 0.28
    y: float = 0.0
    w: float = 0.06
    d: float = 0.06
    h: float = 0.09
    yaw_deg: float = 0.0
    mass_kg: float = 0.03
    friction: float = 0.8


@dataclass(frozen=True)
class Look:
    """Everything about how the scene renders that a sim-to-real fit may adjust. The wrist camera is
    `python -m valet.armd.simfit`'s fit to the real fixed jaw's outline, held level at roll -90° and looking
    along the jaws; Menagerie's mount looks 33° below them."""

    arm_rgba: tuple[float, float, float, float] = (0.93, 0.93, 0.91, 1.0)
    arm_glow: float = 0.3
    box_rgba: tuple[float, float, float, float] = (0.96, 0.96, 0.97, 1.0)
    desk_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    wrist_fovy_deg: float = 83.6
    wrist_pos: tuple[float, float, float] = (0.00831, 0.05303, -0.03891)
    wrist_quat: tuple[float, float, float, float] = (0.977626, -0.203913, -0.044585, -0.026063)
    headlight: float = 0.35
    ambient: float = 0.4
    ceiling_light: float = 0.45


@dataclass(frozen=True)
class SceneSpec:
    box: BoxSpec = field(default_factory=BoxSpec)
    look: Look = field(default_factory=Look)
    table_z: float = 0.0


def build(scene: SceneSpec = SceneSpec()) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    look = scene.look
    for mat in spec.materials:
        if mat.rgba[0] > 0.9 and mat.rgba[2] < 0.2:
            mat.rgba = list(look.arm_rgba)
            mat.emission = look.arm_glow
    for g in spec.geoms:
        if g.name in CAMERA_PARTS:
            g.rgba = [0, 0, 0, 0]
    cam = next(c for c in spec.cameras if c.name == WRIST_CAMERA)
    cam.fovy = look.wrist_fovy_deg
    cam.pos = list(look.wrist_pos)
    cam.quat = list(look.wrist_quat)

    spec.visual.headlight.diffuse = [look.headlight] * 3
    spec.visual.headlight.ambient = [look.ambient] * 3
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.quality.shadowsize = 2048

    world = spec.worldbody
    if DESK_TEXTURE.is_file():
        spec.add_texture(name="desk", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(DESK_TEXTURE))
        desk = spec.add_material(name="desk", rgba=list(look.desk_rgba), texrepeat=[6, 6], texuniform=True)
        desk.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "desk"
        world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], pos=[0, 0, scene.table_z], material="desk")
    else:
        world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], pos=[0, 0, scene.table_z], rgba=[0.59, 0.58, 0.54, 1])

    world.add_light(name="ceiling", pos=[0.3, 0.1, 1.2], dir=[0, 0, -1], diffuse=[look.ceiling_light] * 3, castshadow=True)
    world.add_light(name="window", pos=[-0.4, 0.6, 0.8], dir=[0.5, -0.6, -0.6], diffuse=[look.ceiling_light * 0.5] * 3, castshadow=False)

    b = scene.box
    yaw = math.radians(b.yaw_deg)
    body = world.add_body(name="box", pos=[b.x, b.y, scene.table_z + b.h / 2], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    body.add_freejoint(name="box")
    body.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b.w / 2, b.d / 2, b.h / 2], mass=b.mass_kg,
                  rgba=list(look.box_rgba), friction=[b.friction, 0.01, 0.001], condim=4, solref=[0.01, 1])

    world.add_camera(name=SIDE_CAMERA, pos=[0.28, -0.55, 0.18], xyaxes=[1, 0, 0, 0, 0.3, 1], fovy=55)
    return spec.compile()
