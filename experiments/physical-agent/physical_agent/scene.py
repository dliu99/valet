"""One benchmark episode behind the tool, and what the agent sees of it each turn."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image

from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import Episode, episode, score

from .tool import Arm

NEAR_CLIP_M = 0.004


@dataclass
class Observation:
    scene: Image.Image
    wrist: Image.Image
    state: str


def open_episode(seed: int) -> tuple[Sim, Arm, Episode]:
    ep = episode(seed)
    sim = Sim(with_box(SimSpec(seed=seed), ep.box, ep.target_xy))
    # MuJoCo's default near plane is 3.2 cm, which clips the box out of the wrist view once it is between the jaws;
    # the real camera sees it. (Found by the jev-policy team; not fixed in sim/ yet.)
    sim.model.vis.map.znear = NEAR_CLIP_M / sim.model.stat.extent
    return sim, Arm(sim), ep


def observe(arm: Arm) -> Observation:
    j = arm.sim.joints()
    e = arm.eef()
    joints = ", ".join(f"{k} {v:.1f}" for k, v in j.items())
    eef = ", ".join(f"{k} {v}" for k, v in e.items())
    return Observation(arm.sim.frame("side"), arm.sim.frame("wrist"), f"joints (deg): {joints}\nend effector: {eef}")


def grade(sim: Sim) -> dict:
    return score(sim.truth())


def jpeg_b64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()


def observation_blocks(obs: Observation, heading: str = "") -> list[dict]:
    blocks: list[dict] = []
    if heading:
        blocks.append({"type": "text", "text": heading})
    blocks += [
        {"type": "text", "text": "Scene camera:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(obs.scene)}},
        {"type": "text", "text": "Wrist camera:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(obs.wrist)}},
        {"type": "text", "text": obs.state},
    ]
    return blocks
