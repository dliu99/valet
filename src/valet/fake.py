"""A tabletop world with a fake arm and a fake Jev, for tests and dry runs without hardware.

The fake judge answers from the world's true state (not from pixels) and can be told to
give unsure or wrong answers, so every branch of the runtime can be exercised.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import camera_looking_down, world_to_pixel
from .judge import Answer, Box, Picked, Question
from .robot import Camera, Move, Observation, Vec3, Workspace


@dataclass
class Thing:
    xyz: Vec3  # centre of its top surface
    size_m: float = 0.08
    held: bool = False
    on_stack: bool = False


@dataclass
class World:
    things: dict[str, Thing]
    board: str = "board"
    stack: list[str] = field(default_factory=list)
    fallen_off: bool = False


class FakeRobot:
    """Moves instantly. The camera sits `cam_up_m` above the grasp point, looking down."""

    def __init__(self, world: World, *, camera: Camera | None = None, cam_up_m: float = 0.05,
                 grasp_radius_m: float = 0.025, drop_on_pick: set[str] | None = None, workspace: Workspace | None = None):
        self.world = world
        self.camera = camera or Camera(640, 480, 500.0, 500.0, 320.0, 240.0)
        self.workspace = workspace or Workspace((-0.5, -0.5, -0.05), (0.5, 0.5, 0.5))
        self.cam_up_m = cam_up_m
        self.grasp_radius_m = grasp_radius_m
        self.drop_on_pick = set(drop_on_pick or ())  # names whose first grasp misses
        self.ee: Vec3 = (0.0, 0.0, 0.25)
        self.gripper = 45.0
        self.moves = 0

    def observe(self) -> Observation:
        cam_at = (self.ee[0], self.ee[1], self.ee[2] + self.cam_up_m)
        return Observation(time.monotonic(), np.zeros((self.camera.height, self.camera.width, 3), np.uint8),
                           camera_looking_down(cam_at), self.ee, self.gripper)

    def move_to(self, xyz: Vec3, *, yaw_deg: float = 0.0) -> Move:
        self.workspace.check(xyz)
        self.moves += 1
        self.ee = xyz
        for t in self.world.things.values():
            if t.held:
                t.xyz = (xyz[0], xyz[1], xyz[2])
        return Move(True, xyz, 0.0)

    def set_gripper(self, opening: float) -> float:
        held = next((n for n, t in self.world.things.items() if t.held), None)
        if opening < self.gripper and held is None:  # closing
            name = self._under_fingers()
            if name and name in self.drop_on_pick:
                self.drop_on_pick.discard(name)
                name = None
            if name:
                self.world.things[name].held = True
                self.gripper = opening + 4.0  # fingers stop on it
                return self.gripper
        if opening > self.gripper and held is not None:  # opening with something held
            self._drop(held)
        self.gripper = opening
        return self.gripper

    def home(self) -> Move:
        return self.move_to((0.0, 0.0, 0.25))

    def hold(self) -> None:
        pass

    def _under_fingers(self) -> str | None:
        for name, t in self.world.things.items():
            if name == self.world.board or t.on_stack:
                continue
            if math.dist(t.xyz[:2], self.ee[:2]) < self.grasp_radius_m and abs(t.xyz[2] - self.ee[2]) < 0.02:
                return name
        return None

    def _drop(self, name: str) -> None:
        t = self.world.things[name]
        t.held = False
        board = self.world.things[self.world.board]
        if math.dist(self.ee[:2], board.xyz[:2]) < 0.03:
            top = board.xyz[2] + sum(0.005 for _ in self.world.stack)
            t.xyz, t.on_stack = (board.xyz[0], board.xyz[1], top), True
            self.world.stack.append(name)
        else:
            t.xyz = (self.ee[0], self.ee[1], 0.003)


class FakeJudge:
    """Answers question ids of the form "<kind>:<item>" from the world.

    `script` overrides answers: {"on_top:cheese": [0.5]} makes the first such question
    unsure and later ones true again. Each call pops one value.
    """

    def __init__(self, world: World, robot: FakeRobot, script: dict[str, list[Answer]] | None = None):
        self.world, self.robot = world, robot
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.asked: list[str] = []

    def ask(self, frame, questions: dict[str, Question], facts=None) -> dict[str, Answer]:
        obs = self.robot.observe()
        out: dict[str, Answer] = {}
        for qid in questions:
            self.asked.append(qid)
            if self.script.get(qid):
                out[qid] = self.script[qid].pop(0)
                continue
            kind, _, name = qid.partition(":")
            out[qid] = self._truth(kind, name, obs)
        return out

    def _truth(self, kind: str, name: str, obs: Observation) -> Answer:
        t = self.world.things.get(name)
        if kind == "present":  # layers count only while loose, as the question says
            return 0.95 if t is not None and not t.held and not t.on_stack and self._box(t, obs) else 0.03
        if kind == "where":
            return (self._box(t, obs) if t is not None else None) or Box(0.4, 0.6, 0.4, 0.6, 0.2)
        if kind == "holding":
            return 0.95 if t is not None and t.held else 0.04
        if kind == "on_top":
            return 0.95 if self.world.stack and self.world.stack[-1] == name else 0.05
        if kind == "stack_ok":
            return 0.05 if self.world.fallen_off else 0.95
        if kind == "after_miss":
            if t is None:
                return Picked("not_visible", 0.9, {})
            choice = "on_top" if t.on_stack else "still_held" if t.held else "beside"
            return Picked(choice, 0.9, {})
        raise KeyError(f"fake judge cannot answer {kind!r}")

    def _box(self, t: Thing, obs: Observation) -> Box | None:
        cam = self.robot.camera
        h = t.size_m / 2
        corners = [(t.xyz[0] + dx, t.xyz[1] + dy, t.xyz[2]) for dx in (-h, h) for dy in (-h, h)]
        px = [world_to_pixel(cam, obs.T_world_cam, c) for c in corners]
        if any(p is None for p in px):
            return None
        us, vs = [p[0] for p in px], [p[1] for p in px]
        left, right, top, bottom = min(us) / cam.width, max(us) / cam.width, min(vs) / cam.height, max(vs) / cam.height
        if right < 0 or left > 1 or bottom < 0 or top > 1:
            return None
        clip = lambda x: min(1.0, max(0.0, x))
        return Box(clip(top), clip(bottom), clip(left), clip(right), 0.9)


def sandwich_world() -> World:
    """Board in front, ingredients spread on the table around it (top-surface heights)."""
    return World(
        things={
            "board": Thing((0.20, 0.00, 0.015), size_m=0.20),
            "bread": Thing((0.12, 0.15, 0.012)),
            "cheese": Thing((0.22, 0.15, 0.003)),
            "ham": Thing((0.32, 0.15, 0.003)),
            "lettuce": Thing((0.12, -0.15, 0.004)),
            "bread_top": Thing((0.22, -0.15, 0.012)),
        }
    )
