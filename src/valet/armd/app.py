from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import ArmdError
from .fake import DEFAULT_BOX, FakeArm
from .fakecam import FakeCameras
from .frames import Cameras, FileCameras, NoFrame, encode, fresh_capture, fresh_frames, turned
from .guard import Clock, Guard, Servos
from .kinematics import GRIPPER, MOTORS, ROLL, URDF_PATH, Kinematics
from .motion import Mover
from .poses import PoseBook
from .segment import DEFAULT_SIZE, SIZES, Segmenter, describe, edge_band
from .settings import Settings

POSES_FILE = Path("runs/armd/poses.json")
FAKE_POSES_FILE = Path("runs/armd/fake/poses.json")
SIM_POSES_FILE = Path("runs/armd/sim/poses.json")
BOX_FIELDS = ("x", "y", "w", "d", "h")


class Armd:
    def __init__(self, settings: Settings, servos: Servos, cameras: Cameras, kind: str, poses_path: Path,
                 clock: Clock | None = None, fake: FakeArm | None = None):
        self.settings = settings
        self.kind = kind
        self.kin = Kinematics()
        self.guard = Guard(servos, clock)
        self.poses = PoseBook(poses_path)
        self.mover = Mover(self.guard, self.kin, self.poses, settings.speed_dps, settings.table_z)
        self._locator = None
        self.cameras = cameras
        self.fake = fake
        self.segmenter = Segmenter()
        self._motion = threading.Lock()

    def state(self) -> dict:
        r = self.guard.read()
        pose = self.kin.pose(r.q)
        return {
            "t": time.time(),
            "kind": self.kind,
            "allow_motion": self.settings.allow_motion,
            "joints": {m: round(float(v), 2) for m, v in zip(MOTORS, r.q)},
            "tip": {k: round(float(v), 4) for k, v in zip("xyz", pose.xyz)},
            "pan_deg": round(pose.pan_deg, 2),
            "reach_m": round(pose.reach_m, 4),
            "height_m": round(float(pose.xyz[2]) - self.settings.table_z, 4),
            "pitch_deg": round(pose.pitch_deg, 2),
            "roll_deg": round(pose.roll_deg, 2),
            "gripper_pct": round(float(r.q[GRIPPER]), 2),
            "grip": self.mover.grip_fact(float(r.q[GRIPPER])),
            "amps": {m: round(float(a), 3) for m, a in zip(MOTORS, r.amps)},
            "peak_amps": round(self.guard.peak_amps, 3),
            "torque": self.guard.torque(),
        }

    def locate(self, body: dict) -> dict:
        try:
            u, v, aspect = (float(body[k]) for k in ("u", "v", "aspect"))
            height = float(body.get("height_m", 0.0))
        except (KeyError, TypeError, ValueError) as e:
            raise ArmdError(400, "bad_request", "body is {u, v, aspect, height_m?}: u, v in 0..1 of the upright wrist image") from e
        if self._locator is None:
            self._locator = self._new_locator()
        return self._locator.locate(self.guard.read().q, u, v, aspect, height)

    def _new_locator(self):
        from .locate import REAL_CAMERA, Locator

        return Locator(self.kin, self.settings.table_z, None if self.kind == "sim" else REAL_CAMERA)

    def triangulate(self, body: dict) -> dict:
        if self._locator is None:
            self._locator = self._new_locator()
        try:
            return self._locator.triangulate(body["a"], body["b"])
        except (KeyError, TypeError, ValueError) as e:
            raise ArmdError(400, "bad_request", "body is {a: ray, b: ray}, each {origin_m, dir} from /locate") from e

    def act(self, motion: object) -> dict:
        if not isinstance(motion, dict):
            raise ArmdError(400, "bad_request", "body is {motion: Motion}")
        moves = motion.get("kind") != "remember"
        if moves and not self.settings.allow_motion:
            raise ArmdError(403, "motion_not_allowed", "allow_motion = false in valet.toml: only remember works")
        with self._motion:
            t0 = time.monotonic()
            before = self.state()
            outcome = self.mover.execute(motion)
            after = self.state()
        result = {"ok": outcome.ok, "before": before, "after": after, "ms": round((time.monotonic() - t0) * 1000.0)}
        if outcome.refused:
            result["refused"] = outcome.refused
        if outcome.stopped:
            result["stopped"] = outcome.stopped
        return result

    def hold(self) -> dict:
        self.guard.interrupt("stopped by a hold request")
        with self._motion:
            self.guard.hold()
        return self.state()

    def relax(self) -> dict:
        self.guard.relax("relaxed by request")
        return self.state()

    def frames(self, cams: object, max_side: object = None) -> dict:
        if not isinstance(cams, list) or not all(isinstance(c, str) for c in cams):
            raise ArmdError(400, "bad_request", 'cams is a list of camera names, e.g. ["wrist", "side"]')
        if max_side is not None and (isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 16):
            raise ArmdError(400, "bad_request", "max_side is an integer of at least 16")
        return fresh_frames(self.cameras, cams, max_side, float(self.guard.read().q[ROLL]))

    def segment(self, body: dict) -> dict:
        """SAM 3 on the current frame of one camera (upright, as /frames serves it), best mask per prompt.
        Runs beside motion, never inside it: it reads the joints once, for the wrist's upright turn."""
        prompts, cam = body.get("prompts"), body.get("cam", "wrist")
        size, max_side = body.get("size", DEFAULT_SIZE), body.get("max_side")
        if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 8
                or not all(isinstance(p, str) and 0 < len(p.strip()) <= 200 for p in prompts)):
            raise ArmdError(400, "bad_request", 'prompts is a list of 1 to 8 short object names, e.g. ["blue bottle cap"]')
        if not isinstance(cam, str):
            raise ArmdError(400, "bad_request", 'cam is a camera name, e.g. "wrist"')
        if type(size) is not int or size not in SIZES:
            raise ArmdError(400, "bad_request", f"size is one of {list(SIZES)} (SAM 3's input side; default {DEFAULT_SIZE})")
        if max_side is not None and (isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 16):
            raise ArmdError(400, "bad_request", "max_side is an integer of at least 16")
        try:
            capture, now = fresh_capture(self.cameras, cam)
        except NoFrame as e:
            raise ArmdError(503, "no_frame", f"no {cam} frame: {e}") from None
        roll = float(self.guard.read().q[ROLL])
        frame = encode(cam, capture, now, max_side, roll)
        image = Image.open(io.BytesIO(base64.b64decode(frame["data_b64"]))).convert("RGB")
        sensor = np.asarray(turned(cam, Image.new("L", capture.image.size, 255), max_side, roll)) > 127
        prompts = list(dict.fromkeys(prompts))
        found, timing = self.segmenter.segment(image, prompts, size)
        band = edge_band(sensor)
        if not body.get("frame"):
            del frame["data_b64"]
        return {"frame": frame, "size": size, **timing,
                "results": {p: describe(found.get(p, []), band, bool(body.get("mask"))) for p in prompts}}

    def pose_book(self) -> dict:
        return self.poses.as_json()

    def fake_truth(self) -> dict:
        return self._fake().truth()

    def fake_reset(self, box: object = None) -> dict:
        fake = self._fake()
        spec = box or {}
        if not isinstance(spec, dict) or not all(isinstance(spec.get(k, 0.0), (int, float)) for k in BOX_FIELDS):
            raise ArmdError(400, "bad_request", "box is {x, y, w, d, h} in metres")
        placed = replace(DEFAULT_BOX, **{k: float(spec[k]) for k in BOX_FIELDS if k in spec})
        placed = replace(placed, z=self.settings.table_z + placed.h / 2)
        self.guard.interrupt("fake reset")
        with self._motion:
            fake.reset(placed)
            self.guard.hold()
            self.mover.forget_grip()
        return fake.truth()

    def close(self) -> None:
        close = getattr(self.guard.servos, "close", None)
        if close:
            close()

    def _fake(self) -> FakeArm:
        if self.fake is None:
            raise ArmdError(404, "not_found", "fake mode only")
        return self.fake


def open_fake(settings: Settings, clock: Clock | None = None, poses_path: Path = FAKE_POSES_FILE) -> Armd:
    from .guard import RealClock

    clock = clock or RealClock()
    kin = Kinematics()
    arm = FakeArm(kin, clock, settings.table_z, replace(DEFAULT_BOX, z=settings.table_z + DEFAULT_BOX.h / 2))
    return Armd(settings, arm, FakeCameras(kin, arm), "fake", poses_path, clock=clock, fake=arm)


def open_sim(settings: Settings, clock: Clock | None = None, poses_path: Path = SIM_POSES_FILE, spec=None) -> Armd:
    from .guard import RealClock
    from .sim import SimSpec, open_world
    from .simscene import SceneSpec

    clock = clock or RealClock()
    spec = spec or SimSpec(scene=SceneSpec(table_z=settings.table_z))
    _, arm, cameras = open_world(spec, clock, Kinematics())
    return Armd(settings, arm, cameras, "sim", poses_path, clock=clock, fake=arm)


def open_so101(settings: Settings, poses_path: Path = POSES_FILE) -> Armd:
    from ..hardware.direct import Arm

    arm = Arm(port=settings.port, robot_id=settings.robot_id, urdf=str(URDF_PATH))
    return Armd(settings, arm, FileCameras(settings.cameras), "so101", poses_path)
