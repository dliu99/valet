import base64
import hashlib
import io
import os
import time

import numpy as np
import pytest
from PIL import Image

from valet.armd.frames import FileCameras, fresh_frames, wrist_upright_is_verified, wrist_upright_turn_deg


def write_frame_red_at_top_left(path, size=(640, 480)):
    img = Image.new("RGB", size, (0, 0, 0))
    img.paste((255, 0, 0), (0, 0, size[0] // 4, size[1] // 4))
    img.save(path, quality=95)


def decode(frame) -> np.ndarray:
    data = base64.b64decode(frame["data_b64"])
    assert hashlib.sha256(data).hexdigest() == frame["sha256"]
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def red_quadrant(pixels: np.ndarray) -> str:
    h, w, _ = pixels.shape
    red = pixels[..., 0].astype(int) - pixels[..., 1]
    ys, xs = np.nonzero(red > 128)
    return ("top" if ys.mean() < h / 2 else "bottom") + "-" + ("left" if xs.mean() < w / 2 else "right")


@pytest.fixture
def cams(tmp_path):
    paths = {"wrist": tmp_path / "wrist.jpg", "side": tmp_path / "side.jpg"}
    for p in paths.values():
        write_frame_red_at_top_left(p)
    return FileCameras(paths)


def test_the_wrist_frame_turns_upright_at_roll_minus_90(cams):
    out = fresh_frames(cams, ["wrist", "side"], None, wrist_roll_deg=-90.0)
    assert out["missing"] == {}
    wrist, side = out["frames"]["wrist"], out["frames"]["side"]
    assert red_quadrant(decode(wrist)) == "bottom-right"
    assert red_quadrant(decode(side)) == "top-left"
    assert wrist["upright"] and side["upright"]
    assert wrist["mime"] == "image/jpeg" and (wrist["width"], wrist["height"]) == (640, 480)


def test_upright_is_a_function_of_roll_and_only_verified_near_minus_90():
    assert wrist_upright_turn_deg(-90.0) == 180.0
    assert wrist_upright_turn_deg(-60.0) != 180.0
    assert wrist_upright_is_verified(-92.0)
    assert not wrist_upright_is_verified(-45.0)


def test_frames_at_an_unverified_roll_say_so(cams):
    out = fresh_frames(cams, ["wrist"], None, wrist_roll_deg=-30.0)
    assert out["frames"]["wrist"]["upright"] is False


def test_frames_older_than_300_ms_are_refused(cams, tmp_path):
    old = time.time() - 0.8
    os.utime(tmp_path / "wrist.jpg", (old, old))
    out = fresh_frames(cams, ["wrist", "side"], None, wrist_roll_deg=-90.0)
    assert "wrist" not in out["frames"] and "side" in out["frames"]
    assert out["missing"]["wrist"].startswith("stale ") and out["missing"]["wrist"].endswith(" ms")


def test_a_missing_file_is_reported(tmp_path):
    out = fresh_frames(FileCameras({"wrist": tmp_path / "none.jpg"}), ["wrist", "laptop"], None, -90.0)
    assert set(out["missing"]) == {"wrist", "laptop"}


def test_a_frozen_camera_is_refused(tmp_path):
    path = tmp_path / "side.jpg"
    write_frame_red_at_top_left(path)
    now = [1000.0]
    cams = FileCameras({"side": path}, wall=lambda: now[0])
    os.utime(path, (now[0], now[0]))
    assert "side" in fresh_frames(cams, ["side"], None, -90.0, wall=lambda: now[0])["frames"]
    now[0] += 1.5
    os.utime(path, (now[0], now[0]))
    out = fresh_frames(cams, ["side"], None, -90.0, wall=lambda: now[0])
    assert out["missing"]["side"].startswith("frozen")


def test_frames_scale_to_max_side_and_hash_what_they_return(tmp_path):
    path = tmp_path / "wrist.jpg"
    write_frame_red_at_top_left(path, size=(1280, 720))
    frame = fresh_frames(FileCameras({"wrist": path}), ["wrist"], 320, -90.0)["frames"]["wrist"]
    assert (frame["width"], frame["height"]) == (320, 180)
    assert decode(frame).shape == (180, 320, 3)
    assert frame["age_ms"] < 300


def test_fake_frames_come_from_the_fake_geometry(app):
    out = app.frames(["wrist", "side"], 320)
    assert out["missing"] == {}
    for frame in out["frames"].values():
        pixels = decode(frame)
        assert (pixels.min(axis=2) > 240).any(), "the white box is drawn"
        assert (np.abs(pixels.astype(int) - 128).max(axis=2) < 6).any(), "the grey jaw is drawn"
