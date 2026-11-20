"""The shared benchmark: pick up the white box and set it down on the blue square. Every episode is fixed by its
seed, so teams compare on the same scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import BoxSpec

PAN_AXIS_X = 0.0388
BOX_REACH_M = (0.20, 0.27)
TARGET_REACH_M = (0.19, 0.27)
BEARING_DEG = 40.0
MIN_SEPARATION_M = 0.12
PLACED_WITHIN_M = 0.03
UPRIGHT_WITHIN_DEG = 15.0
LIFTED_AT_LEAST_M = 0.02
BENCHMARK_SEEDS = tuple(range(20))


@dataclass(frozen=True)
class Episode:
    seed: int
    box: BoxSpec
    target_xy: tuple[float, float]


def _on_table(rng: np.random.Generator, reach: tuple[float, float]) -> tuple[float, float]:
    r = rng.uniform(*reach)
    a = np.radians(rng.uniform(-BEARING_DEG, BEARING_DEG))
    return PAN_AXIS_X + r * float(np.cos(a)), r * float(np.sin(a))


def episode(seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    side = float(rng.uniform(0.04, 0.055))
    box_xy = _on_table(rng, BOX_REACH_M)
    box = BoxSpec(x=box_xy[0], y=box_xy[1], yaw_deg=float(rng.uniform(-30, 30)), w=side, d=side, h=float(rng.uniform(0.06, 0.09)))
    for _ in range(100):
        target = _on_table(rng, TARGET_REACH_M)
        if np.hypot(target[0] - box.x, target[1] - box.y) >= MIN_SEPARATION_M:
            return Episode(seed, box, target)
    raise RuntimeError(f"seed {seed}: no target far enough from the box")


def score(truth: dict) -> dict:
    box, target = truth["box"], truth["target"]
    off = float(np.hypot(box["x"] - target[0], box["y"] - target[1]))
    resting = abs(box["z"] - box["h"] / 2) < 0.01
    placed = off <= PLACED_WITHIN_M and box["tilt_deg"] <= UPRIGHT_WITHIN_DEG and resting and not truth["touching_jaws"]
    return {"placed": placed, "lifted": truth["lifted_m"] >= LIFTED_AT_LEAST_M, "off_target_m": round(off, 4),
            "tilt_deg": box["tilt_deg"], "resting": resting, "released": not truth["touching_jaws"]}
