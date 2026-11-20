"""Load valet.toml (see valet.example.toml) into the objects the runtime needs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .judge import Thresholds
from .robot import Camera, Workspace
from .runtime import Settings
from .skills import Gripper, ScriptedPick, ScriptedPlace, Skill


def load(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def settings(cfg: dict[str, Any]) -> Settings:
    table, scan, rt, th, grip = (cfg.get(k, {}) for k in ("table", "scan", "runtime", "thresholds", "gripper"))
    return Settings(
        table_z=table.get("z", 0.0),
        board_height_m=table.get("board_height_m", 0.015),
        scan_points=[tuple(p) for p in scan.get("points", [[0.22, 0.12, 0.30], [0.22, -0.12, 0.30]])],
        gripper_open=grip.get("open", Gripper.open),
        thresholds=Thresholds(**th),
        **{k: rt[k] for k in rt},
    )


def skills(cfg: dict[str, Any]) -> dict[str, Skill]:
    grip = Gripper(**cfg.get("gripper", {}))
    specs = cfg.get("skills") or {
        "pick_flat": {"kind": "scripted_pick"},
        "place_on_stack": {"kind": "scripted_place"},
    }
    out: dict[str, Skill] = {}
    for name, spec in specs.items():
        spec = dict(spec)
        kind = spec.pop("kind")
        if kind == "scripted_pick":
            out[name] = ScriptedPick(gripper=grip, **spec)
        elif kind == "scripted_place":
            out[name] = ScriptedPlace(gripper=grip, **spec)
        elif kind == "lerobot":
            from .policy import LeRobotPolicySkill

            out[name] = LeRobotPolicySkill(**spec)
        else:
            raise ValueError(f"skill {name!r}: unknown kind {kind!r} (scripted_pick, scripted_place, lerobot)")
    return out


def so101(cfg: dict[str, Any]):
    from .hardware.so101 import SO101Config, SO101Robot

    robot, cam, ws = cfg.get("robot", {}), dict(cfg.get("camera", {})), cfg.get("workspace", {})
    camera_extra = {k: cam.pop(k) for k in ("key", "index", "fps", "ee_to_cam_xyz", "ee_to_cam_rpy_deg") if k in cam}
    c = SO101Config(**robot)
    if cam:
        c.camera = Camera(**cam)
    if ws:
        c.workspace = Workspace(tuple(ws["lo"]), tuple(ws["hi"]))
    c.camera_key = camera_extra.get("key", c.camera_key)
    c.camera_index = camera_extra.get("index", c.camera_index)
    c.camera_fps = camera_extra.get("fps", c.camera_fps)
    c.ee_to_cam_xyz = tuple(camera_extra.get("ee_to_cam_xyz", c.ee_to_cam_xyz))
    c.ee_to_cam_rpy_deg = tuple(camera_extra.get("ee_to_cam_rpy_deg", c.ee_to_cam_rpy_deg))
    return SO101Robot(c)
