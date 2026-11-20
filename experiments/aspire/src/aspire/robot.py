"""The primitive API a program drives (see api.md), and the recorder that traces every call.

A program gets one `Robot`. It sees joint readings, wrist-camera frames and the known camera geometry, as on the real
arm (which has only the wrist camera); it never sees the simulator, the box's pose or the target's position. Each
primitive call is recorded with its inputs, outputs, status, the state before and after, and the frames after it with
overlays: the wrist camera's, and an observer side camera's that only the debugging agents see, never the program."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from valet_sim.ik import IK, Unreachable
from valet_sim.scene import MOTORS
from valet_sim.sim import CAMERAS, Sim, develop

from . import perceive
from .camera import Cameras

ARM = MOTORS[:5]
SETTLED_DEG = 0.5
STILL_DEG = 0.03
POLL_S = 0.05


class EpisodeLimit(BaseException):
    """A program ran past its call, perception or time allowance. BaseException so `except Exception` cannot hide it."""


@dataclass(frozen=True)
class Limits:
    calls: int = 60
    localize: int = 10
    sim_s: float = 120.0


@dataclass
class Recorder:
    out: Path | None
    calls: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    keyframes: bool = True

    def save(self, image: Image.Image, name: str) -> str | None:
        if self.out is None or not self.keyframes:
            return None
        self.out.mkdir(parents=True, exist_ok=True)
        image.save(self.out / name, "JPEG", quality=82)
        return name


def _r(v, nd=4):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _r(x, nd) for k, x in v.items()}
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    return v


class Robot:
    def __init__(self, sim: Sim, recorder: Recorder, backend: str = "claude", limits: Limits = Limits()):
        self._sim = sim
        self._rec = recorder
        self._backend = backend
        self._limits = limits
        self._ik = IK(sim.model)
        self._cams = Cameras(sim.model, sim.spec.image_size, sim.spec.optics)
        self._n_localize = 0
        self._frames: dict[str, tuple[np.ndarray, Image.Image]] = {}
        self._capture()
        self._rec.save(self._frames["wrist"][1], "c00_before_wrist.jpg")
        self._rec.save(self._frames["side"][1], "c00_before_side.jpg")

    # ---- primitives -------------------------------------------------------------------------------------------

    def observe(self) -> dict:
        """Joint readings, gripper reading, servo currents, grasp point and jaw pitch, and the wrist frame."""
        def run():
            s = self._state()
            return {**s, "frames": {"wrist": np.asarray(self._frames["wrist"][1]).copy()}}
        return self._call("observe", {}, run, lambda r: {k: v for k, v in r.items() if k != "frames"})

    def localize(self, text: str) -> dict:
        """Where `text` is in the wrist camera's current frame: a pixel bounding box, and where the ray through its
        centre meets the table plane (z = 0)."""
        camera = "wrist"
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must describe what to find")

        def run():
            self._n_localize += 1
            if self._n_localize > self._limits.localize:
                raise EpisodeLimit(f"more than {self._limits.localize} localize calls")
            raw, image = self._frames[camera]
            joints = self._sim.joints()
            f = perceive.locate(image, text, perceive.frame_key(raw), self._backend)
            self._rec.calls[-1]["cached"] = f.cached  # for cost accounting; not shown to the program
            out = {"found": f.found, "bbox": None, "center_px": None, "table_xy": None, "camera": camera,
                   "joints": _r(joints, 2)}
            if f.bbox is not None:
                l, t, r, b = f.bbox
                c = ((l + r) / 2, (t + b) / 2)
                hit = self._cams.view(camera, joints).to_plane(*c, 0.0)
                out.update(bbox=_r([l, t, r, b], 1), center_px=_r(c, 1), table_xy=None if hit is None else _r(hit[:2]))
            if not f.found:
                out["found"] = False
            return out
        return self._call("localize", {"text": text, "camera": camera}, run)

    def move_to_pose(self, xyz, pitch_deg: float, roll_deg: float = -90.0, timeout_s: float = 4.0) -> dict:
        """Move the grasp point (between the jaws) to xyz in metres, jaws pitched `pitch_deg` below horizontal and
        the wrist rolled to `roll_deg`. Waits until the arm settles, stops moving, or `timeout_s` passes."""
        xyz = [float(v) for v in xyz]
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError("xyz must be three finite numbers (metres)")
        pitch_deg, roll_deg, timeout_s = float(pitch_deg), float(roll_deg), float(timeout_s)

        def run():
            sim = self._sim
            now = sim.joints()
            gripper_goal = float(np.degrees(sim.goal[MOTORS.index("gripper")]))
            try:
                q = self._ik.solve(xyz, pitch_deg, roll_deg, seed=now, gripper=gripper_goal)
            except Unreachable as e:
                return {"status": "unreachable", "message": str(e), **self._pose()}
            goals = {j: q[j] for j in ARM}
            sim.set_goals(goals)
            status = self._wait(lambda j: all(abs(j[k] - goals[k]) < SETTLED_DEG for k in ARM), ARM, timeout_s)
            pose = self._pose()
            return {"status": status, "error_m": round(float(np.linalg.norm(np.subtract(pose["grasp_point"], xyz))), 4),
                    **pose, "goal_joints": _r(goals, 2)}
        return self._call("move_to_pose", {"xyz": _r(xyz), "pitch_deg": pitch_deg, "roll_deg": roll_deg, "timeout_s": timeout_s},
                          run, overlay_xyz=xyz)

    def set_gripper(self, pct: float, timeout_s: float = 3.0) -> dict:
        """Drive the jaw to `pct` (0 closed, 100 wide open). Waits until it gets there, stalls on something, or
        `timeout_s` passes; returns the reading and the gripper servo's current."""
        pct, timeout_s = float(pct), float(timeout_s)

        def run():
            self._sim.set_goals({"gripper": pct})
            status = self._wait(lambda j: abs(j["gripper"] - pct) < 1.0, ("gripper",), timeout_s, still="stalled")
            s = self._sim.state()
            return {"status": status, "gripper": round(s["joints"]["gripper"], 2), "amps": s["amps"]["gripper"]}
        return self._call("set_gripper", {"pct": pct, "timeout_s": timeout_s}, run)

    # ---- camera geometry (no motion, not traced) ----------------------------------------------------------------

    def pixel_to_world(self, px, z: float, joints: dict | None = None) -> list[float] | None:
        """Where the ray through pixel (u, v) meets the horizontal plane at height z, for the camera at `joints`
        (default: now). None if the ray does not reach that plane."""
        hit = self._cams.view("wrist", joints or self._sim.joints()).to_plane(float(px[0]), float(px[1]), float(z))
        return None if hit is None else _r(hit)

    def world_to_pixel(self, xyz, joints: dict | None = None) -> list[float] | None:
        """The pixel (u, v) a world point lands on, for the camera at `joints` (default: now). None if behind it."""
        p = self._cams.view("wrist", joints or self._sim.joints()).project([float(v) for v in xyz])
        return None if p is None else _r(p, 1)

    def camera_ray(self, px, joints: dict | None = None) -> dict:
        """Origin and unit direction (world frame) of the ray through pixel (u, v)."""
        o, d = self._cams.view("wrist", joints or self._sim.joints()).ray(float(px[0]), float(px[1]))
        return {"origin": _r(o), "direction": _r(d)}

    def log(self, *parts) -> None:
        """A note for the trace."""
        self._rec.log.append(f"[after call {len(self._rec.calls)}] " + " ".join(str(p) for p in parts)[:500])

    # ---- internals ---------------------------------------------------------------------------------------------

    def _state(self) -> dict:
        s = self._sim.state()
        p = self._pose()
        return {"t": s["t"], "joints": s["joints"], "gripper": s["joints"]["gripper"], "amps": s["amps"],
                "grasp_point": p["grasp_point"], "pitch_deg": p["pitch_deg"]}

    def _pose(self) -> dict:
        point, pitch = self._ik.pose(self._sim.joints())
        return {"grasp_point": _r(point), "pitch_deg": round(pitch, 1)}

    def _wait(self, done, joints, timeout_s: float, still: str = "stopped") -> str:
        sim = self._sim
        t_end = sim.data.time + timeout_s
        last = sim.joints()
        still_s = 0.0
        while sim.data.time < t_end:
            if sim.data.time > self._limits.sim_s:
                raise EpisodeLimit(f"past {self._limits.sim_s:.0f} s of simulated time")
            sim.step(POLL_S)
            now = sim.joints()
            if done(now):
                return "reached"
            moved = max(abs(now[k] - last[k]) for k in joints)
            still_s = still_s + POLL_S if moved < STILL_DEG else 0.0
            if still_s >= 0.25:
                return still
            last = now
        return "timeout"

    def _capture(self) -> None:
        for cam, name in CAMERAS.items():
            raw = self._sim.renderer.render(name)
            image = Image.fromarray(raw)
            if cam == "wrist":
                key = perceive.frame_key(raw)
                image = develop(image, self._sim.spec.optics, np.random.default_rng(int(key[:8], 16)))
            self._frames[cam] = (raw, image)

    def _call(self, name: str, args: dict, run, public=None, overlay_xyz=None):
        if len(self._rec.calls) >= self._limits.calls:
            raise EpisodeLimit(f"more than {self._limits.calls} primitive calls")
        i = len(self._rec.calls) + 1
        before = self._state()
        t0 = time.time()
        record = {"i": i, "prim": name, "args": args, "state_before": _slim(before)}
        self._rec.calls.append(record)
        try:
            result = run()
        except EpisodeLimit:
            record.update(status="limit")
            raise
        except Exception as e:
            record.update(status="error", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            record["wall_s"] = round(time.time() - t0, 2)
        shown = public(result) if public else result
        record.update(status=shown.get("status", "found" if shown.get("found") else ("not_found" if name == "localize" else "ok")),
                      result=_r(shown))
        if name == "localize":
            seen = self._frames[args["camera"]][1]
            record["frames"] = {"seen": self._rec.save(_overlay_bbox(seen, shown, f"#{i} localize({args['text']!r})"), f"c{i:02d}_seen.jpg")}
        else:
            self._capture()
            after = self._state()
            record["state_after"] = _slim(after)
            frames = {}
            for cam in CAMERAS:
                img = self._frames[cam][1]
                caption = f"#{i} {name} -> {record['status']}"
                if overlay_xyz is not None:
                    view = self._cams.view(cam, self._sim.joints())
                    img = _overlay_points(img, view, [(overlay_xyz, (255, 60, 60), "goal"), (after["grasp_point"], (60, 255, 60), "now")], caption)
                else:
                    img = _caption(img, caption)
                frames[cam] = self._rec.save(img, f"c{i:02d}_after_{cam}.jpg")
            record["frames"] = frames
        return result


def _slim(s: dict) -> dict:
    return {"t": s["t"], "joints": s["joints"], "gripper_amps": s["amps"]["gripper"], "grasp_point": s["grasp_point"], "pitch_deg": s["pitch_deg"]}


def _caption(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    d.text((4, 2), text[:100], fill=(255, 255, 255))
    return img


def _overlay_points(img: Image.Image, view, points, caption: str) -> Image.Image:
    img = _caption(img, caption)
    d = ImageDraw.Draw(img)
    for xyz, color, label in points:
        p = view.project(xyz)
        if p is None or not (-50 < p[0] < img.width + 50 and -50 < p[1] < img.height + 50):
            continue
        u, v = p
        d.line([u - 8, v, u + 8, v], fill=color, width=2)
        d.line([u, v - 8, u, v + 8], fill=color, width=2)
        d.text((u + 6, v + 4), label, fill=color)
    return img


def _overlay_bbox(img: Image.Image, r: dict, caption: str) -> Image.Image:
    img = _caption(img, caption + (" found" if r.get("found") else " NOT found"))
    if r.get("bbox"):
        d = ImageDraw.Draw(img)
        d.rectangle(r["bbox"], outline=(255, 220, 0), width=2)
        u, v = r["center_px"]
        d.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(255, 220, 0))
    return img
