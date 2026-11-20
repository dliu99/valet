"""The scenes programs are debugged and graded on.

`benchmark` is valet-sim's task (`valet_sim.task.episode`): seeds 0-19 are the held-out evaluation set and are never
used for debugging; seeds 100-109 are the debug set. `variant` is the transfer task: thinner, taller boxes that
start on the arm's right, and a target square on its left (the +y side), so every carry crosses in front of the
base. Its debug seeds are 200-209 and its held-out seeds 300-319."""

from __future__ import annotations

import numpy as np
from valet_sim.scene import BoxSpec
from valet_sim.task import MIN_SEPARATION_M, PAN_AXIS_X, Episode, episode

DEBUG_SEEDS = {"benchmark": tuple(range(100, 110)), "variant": tuple(range(200, 210))}
HELDOUT_SEEDS = {"benchmark": tuple(range(20)), "variant": tuple(range(300, 320))}


def _at(r: float, bearing_deg: float) -> tuple[float, float]:
    a = np.radians(bearing_deg)
    return PAN_AXIS_X + r * float(np.cos(a)), r * float(np.sin(a))


def variant_episode(seed: int) -> Episode:
    rng = np.random.default_rng(10_000 + seed)
    side = float(rng.uniform(0.032, 0.042))
    x, y = _at(rng.uniform(0.20, 0.27), rng.uniform(-40.0, 0.0))
    box = BoxSpec(x=x, y=y, yaw_deg=float(rng.uniform(-30, 30)), w=side, d=side, h=float(rng.uniform(0.085, 0.10)))
    for _ in range(100):
        target = _at(rng.uniform(0.19, 0.27), rng.uniform(10.0, 40.0))
        if np.hypot(target[0] - x, target[1] - y) >= MIN_SEPARATION_M:
            return Episode(seed, box, target)
    raise RuntimeError(f"variant seed {seed}: no target far enough from the box")


EPISODES = {"benchmark": episode, "variant": variant_episode}


def make(task: str, seed: int) -> Episode:
    return EPISODES[task](seed)
