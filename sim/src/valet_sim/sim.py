"""The simulated SO-101: joint goals in, physics, joint readings and camera frames out.

Joints are LeRobot calibrated degrees, as the real arm reads them (`so101_new_calib` takes them as radians);
the gripper's 0-100 is taken as degrees, which spans the model's jaw range. Goals are approached at the servo's
speed limit, as a real STS3215 does, and the servos are softer than Menagerie's (see ServoModel)."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from .scene import GRASP_SITE, MOTORS, SIDE_CAMERA, WRIST_CAMERA, BoxSpec, SceneSpec, build, mujoco

START_JOINTS = {"shoulder_pan": 0.0, "shoulder_lift": -27.2, "elbow_flex": -1.0, "wrist_flex": 88.2, "wrist_roll": -90.0, "gripper": 60.0}
CAMERAS = {"wrist": WRIST_CAMERA, "side": SIDE_CAMERA}
MAX_CATCH_UP_S = 2.0
CURRENT_FILTER_S = 0.08


@dataclass(frozen=True)
class ServoModel:
    """Menagerie's kp 998 tracks within 0.03 deg but pulls 1.6-2.7 A on every goal step; the real arm read
    0.1-0.25 A moving at 8-25 deg/s, and kp 80 reads 0.4 A at 30 deg/s and 0.6 A at 60."""

    kp: float = 80.0
    kv: float = 1.5
    max_speed_dps: float = 60.0
    amps_per_nm: float = 0.9
    idle_amps: float = 0.08


@dataclass(frozen=True)
class Optics:
    """What the real wrist camera does that MuJoCo's rasterizer does not."""

    barrel_k1: float = -0.205
    gain: tuple[float, float, float] = (0.96, 0.98, 1.03)
    blur_px: float = 0.6
    noise_sd: float = 0.012
    vignette: float = 0.15


@dataclass(frozen=True)
class SimSpec:
    scene: SceneSpec = field(default_factory=SceneSpec)
    servo: ServoModel = field(default_factory=ServoModel)
    optics: Optics = field(default_factory=Optics)
    start: dict[str, float] = field(default_factory=lambda: dict(START_JOINTS))
    image_size: tuple[int, int] = (640, 360)
    seed: int = 0


class Sim:
    """One arm, one box. Drive it with `step(seconds)` (simulated time), or pass `clock` to have physics keep up
    with that clock on every call (what the HTTP server does). Safe to call from several threads."""

    def __init__(self, spec: SimSpec = SimSpec(), clock: Callable[[], float] | None = None):
        self.spec = spec
        self.clock = clock
        self.lock = threading.RLock()
        self.model = build(spec.scene)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.range = np.degrees(m.jnt_range[jid])
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in MOTORS])
        for a in self.act:
            m.actuator_gainprm[a, 0] = spec.servo.kp
            m.actuator_biasprm[a, 1:3] = [-spec.servo.kp, -spec.servo.kv]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box")]
        self.target_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.jaws = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("gripper", "moving_jaw_so101_v1")}
        self.rng = np.random.default_rng(spec.seed)
        self.renderer = _RenderThread(self, spec.image_size)
        self.reset(spec.scene.box, spec.scene.target_xy)

    def reset(self, box: BoxSpec | None = None, target_xy: tuple[float, float] | None = None) -> None:
        with self.lock:
            m, d = self.model, self.data
            box = box or self.spec.scene.box
            mujoco.mj_resetData(m, d)
            m.geom_size[self.box_geom] = [box.w / 2, box.d / 2, box.h / 2]
            yaw = np.radians(box.yaw_deg)
            d.qpos[self.box_qadr:self.box_qadr + 7] = [box.x, box.y, box.h / 2, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            if target_xy is not None and self.target_geom >= 0:
                m.geom_pos[self.target_geom][:2] = target_xy
            start = np.radians([self.spec.start[j] for j in MOTORS])
            d.qpos[self.qadr] = start
            d.ctrl[self.act] = start
            self.goal = start.copy()
            self.box = box
            self.torque_avg = np.zeros(len(MOTORS))
            self.lifted_m = 0.0
            m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            mujoco.mj_forward(m, d)
            self._t0 = self.clock() if self.clock else 0.0

    def step(self, seconds: float) -> None:
        with self.lock:
            self._run_until(self.data.time + seconds)

    def sync(self) -> None:
        if self.clock is None:
            return
        with self.lock:
            behind = self.clock() - self._t0 - self.data.time
            if behind > MAX_CATCH_UP_S:
                self._t0 += behind - MAX_CATCH_UP_S
            self._run_until(self.clock() - self._t0)

    def set_goals(self, goals: dict[str, float]) -> None:
        with self.lock:
            self.sync()
            for name, deg in goals.items():
                i = MOTORS.index(name)
                lo, hi = self.range[i]
                self.goal[i] = np.radians(np.clip(float(deg), lo, hi))

    def torque(self, on: bool) -> None:
        with self.lock:
            self.sync()
            if on:
                self.goal = self.data.qpos[self.qadr].copy()
                self.data.ctrl[self.act] = self.goal
                self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            else:
                self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    def joints(self) -> dict[str, float]:
        with self.lock:
            self.sync()
            return dict(zip(MOTORS, (float(v) for v in np.degrees(self.data.qpos[self.qadr]))))

    def state(self) -> dict:
        with self.lock:
            self.sync()
            d = self.data
            torque_on = not (self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))
            amps = self.spec.servo.idle_amps + self.spec.servo.amps_per_nm * self.torque_avg if torque_on else np.zeros(len(MOTORS))
            return {
                "t": round(float(d.time), 3),
                "joints": {j: round(float(v), 2) for j, v in zip(MOTORS, np.degrees(d.qpos[self.qadr]))},
                "goals": {j: round(float(v), 2) for j, v in zip(MOTORS, np.degrees(self.goal))},
                "amps": {j: round(float(a), 3) for j, a in zip(MOTORS, amps)},
                "torque": torque_on,
                "grasp_point": [round(float(v), 4) for v in d.site_xpos[self.site]],
            }

    def truth(self) -> dict:
        """What only a simulator knows: for scoring, never for a policy to act on."""
        with self.lock:
            self.sync()
            d = self.data
            pos = d.xpos[self.box_body]
            up = d.xmat[self.box_body].reshape(3, 3)[:, 2]
            touching = self._box_touches()
            target = self.model.geom_pos[self.target_geom][:2] if self.target_geom >= 0 else None
            return {
                "box": {"x": round(float(pos[0]), 4), "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4),
                        "tilt_deg": round(float(np.degrees(np.arccos(np.clip(abs(up[2]), 0, 1)))), 1),
                        "w": self.box.w, "d": self.box.d, "h": self.box.h},
                "held": bool(self.jaws <= touching and self._bottom_m() > 0.003),
                "touching_jaws": bool(self.jaws & touching),
                "lifted_m": round(self.lifted_m, 4),
                "target": None if target is None else [round(float(v), 4) for v in target],
            }

    def frame(self, cam: str = "wrist") -> Image.Image:
        """The upright frame: for the wrist camera, what the real pipeline delivers after turning the raw image."""
        image = Image.fromarray(self.renderer.render(CAMERAS[cam]))
        return develop(image, self.spec.optics, self.rng) if cam == "wrist" else image

    def grasp_point(self) -> np.ndarray:
        with self.lock:
            self.sync()
            return self.data.site_xpos[self.site].copy()

    def _run_until(self, t: float) -> None:
        m, d, servo = self.model, self.data, self.spec.servo
        dt = m.opt.timestep
        max_step = np.radians(servo.max_speed_dps) * dt
        alpha = dt / CURRENT_FILTER_S
        while d.time < t:
            ctrl = d.ctrl[self.act]
            d.ctrl[self.act] = ctrl + np.clip(self.goal - ctrl, -max_step, max_step)
            mujoco.mj_step(m, d)
            self.torque_avg += alpha * (np.abs(d.actuator_force[self.act]) - self.torque_avg)
            if self.jaws <= self._box_touches():
                self.lifted_m = max(self.lifted_m, self._bottom_m())

    def _bottom_m(self) -> float:
        return float(self.data.xpos[self.box_body][2] - self.box.h / 2)

    def _box_touches(self) -> set[int]:
        out = set()
        for c in self.data.contact[: self.data.ncon]:
            if self.box_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.box_geom else c.geom1
                out.add(int(self.model.geom_bodyid[other]))
        return out


class _RenderThread:
    """GL contexts belong to one thread; a context used from a second HTTP thread hangs."""

    def __init__(self, sim: Sim, size: tuple[int, int]):
        self.sim = sim
        self.size = size
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._serve, daemon=True, name="valet-sim-render").start()

    def render(self, camera: str) -> np.ndarray:
        done: queue.Queue = queue.Queue(maxsize=1)
        self.jobs.put((camera, done))
        ok, out = done.get(timeout=10)
        if not ok:
            raise RuntimeError(f"render failed: {out}")
        return out

    def _serve(self) -> None:
        w, h = self.size
        renderer = mujoco.Renderer(self.sim.model, h, w)
        while True:
            camera, done = self.jobs.get()
            try:
                with self.sim.lock:
                    self.sim.sync()
                    renderer.update_scene(self.sim.data, camera=camera)
                done.put((True, renderer.render().copy()))
            except Exception as e:
                done.put((False, f"{type(e).__name__}: {e}"))


def develop(image: Image.Image, optics: Optics, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(image, np.float32) / 255.0
    h, w = a.shape[:2]
    if optics.barrel_k1:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2)
        scale = 1.0 + optics.barrel_k1 * (nx * nx + ny * ny)
        a = a[np.clip(ny * scale * (w / 2) + h / 2, 0, h - 1).astype(int), np.clip(nx * scale * (w / 2) + w / 2, 0, w - 1).astype(int)]
    if optics.vignette:
        yy, xx = np.mgrid[0:h, 0:w]
        a = a * (1.0 - optics.vignette * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (w / 2)) ** 2))[..., None]
    a = np.clip(a * np.array(optics.gain, np.float32), 0, 1)
    if optics.noise_sd:
        a = a + rng.normal(0, optics.noise_sd, a.shape).astype(np.float32)
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(optics.blur_px)) if optics.blur_px else out


def realtime(spec: SimSpec = SimSpec()) -> Sim:
    return Sim(spec, clock=time.monotonic)


def with_box(spec: SimSpec, box: BoxSpec, target_xy: tuple[float, float] | None) -> SimSpec:
    return replace(spec, scene=replace(spec.scene, box=box, target_xy=target_xy))
