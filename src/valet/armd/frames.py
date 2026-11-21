from __future__ import annotations

import base64
import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from PIL import Image

STALE_AFTER_MS = 300.0
FROZEN_AFTER_S = 1.0
JPEG_QUALITY = 90
UPRIGHT_VERIFIED_AT_ROLL_DEG = -90.0
UPRIGHT_VERIFIED_WITHIN_DEG = 10.0
ROLL_TURN_SIGN_UNVERIFIED = 1.0


def wrist_upright_turn_deg(roll_deg: float) -> float:
    return (180.0 + ROLL_TURN_SIGN_UNVERIFIED * (roll_deg - UPRIGHT_VERIFIED_AT_ROLL_DEG)) % 360.0


def wrist_upright_is_verified(roll_deg: float) -> bool:
    return abs(roll_deg - UPRIGHT_VERIFIED_AT_ROLL_DEG) <= UPRIGHT_VERIFIED_WITHIN_DEG


class NoFrame(Exception):
    pass


@dataclass(frozen=True)
class Capture:
    image: Image.Image
    t_capture: float


class Cameras(Protocol):
    def capture(self, cam: str) -> Capture: ...


class FileCameras:
    def __init__(self, paths: dict[str, Path], wall: Callable[[], float] = time.time):
        self.paths = {cam: Path(p) for cam, p in paths.items()}
        self.wall = wall
        self._unchanged_since: dict[str, tuple[str, float]] = {}

    def capture(self, cam: str) -> Capture:
        path = self.paths.get(cam)
        if path is None:
            raise NoFrame(f"no file configured for camera {cam!r}")
        try:
            mtime = path.stat().st_mtime
            data = path.read_bytes()
        except FileNotFoundError:
            raise NoFrame(f"no frame at {path} (is camfeed running?)") from None
        digest = hashlib.sha256(data).hexdigest()
        now = self.wall()
        seen = self._unchanged_since.get(cam)
        if seen is None or seen[0] != digest:
            self._unchanged_since[cam] = (digest, now)
        elif now - seen[1] > FROZEN_AFTER_S:
            raise NoFrame(f"frozen {now - seen[1]:.1f} s (same bytes)")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except OSError as e:
            raise NoFrame(f"unreadable frame at {path}: {e}") from None
        return Capture(image.convert("RGB"), mtime)


def turned(cam: str, image: Image.Image, max_side: int | None, wrist_roll_deg: float) -> Image.Image:
    """The image as /frames serves it: the wrist frame turned upright, then shrunk to max_side."""
    if cam == "wrist":
        image = image.rotate(wrist_upright_turn_deg(wrist_roll_deg), resample=Image.Resampling.BILINEAR)
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def encode(cam: str, capture: Capture, now: float, max_side: int | None, wrist_roll_deg: float) -> dict:
    image = turned(cam, capture.image, max_side, wrist_roll_deg)
    upright = wrist_upright_is_verified(wrist_roll_deg) if cam == "wrist" else True
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    return {
        "cam": cam,
        "t_capture": capture.t_capture,
        "age_ms": round((now - capture.t_capture) * 1000.0, 1),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": image.width,
        "height": image.height,
        "mime": "image/jpeg",
        "data_b64": base64.b64encode(data).decode("ascii"),
        "upright": upright,
    }


def fresh_capture(cameras: Cameras, cam: str, wall: Callable[[], float] = time.time) -> tuple[Capture, float]:
    capture = cameras.capture(cam)
    now = wall()
    age_ms = (now - capture.t_capture) * 1000.0
    if age_ms > STALE_AFTER_MS:
        raise NoFrame(f"stale {age_ms:.0f} ms")
    return capture, now


def fresh_frames(cameras: Cameras, cams: list[str], max_side: int | None, wrist_roll_deg: float,
                 wall: Callable[[], float] = time.time) -> dict:
    frames, missing = {}, {}
    for cam in cams:
        try:
            capture, now = fresh_capture(cameras, cam, wall)
        except NoFrame as e:
            missing[cam] = str(e)
            continue
        frames[cam] = encode(cam, capture, now, max_side, wrist_roll_deg)
    return {"frames": frames, "missing": missing}
