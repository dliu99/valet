"""armd's MuJoCo backend: the Servos and Cameras armd drives, backed by physics instead of the serial bus.

Joint values are LeRobot calibrated degrees, as on the real arm; the Menagerie model takes them as radians.
The gripper's 0-100 is taken as degrees too, which spans the model's jaw range."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from PIL import Image, ImageFilter

from .frames import Capture, NoFrame
from .guard import Clock
from .kinematics import ARM, MOTORS, Kinematics
from .simscene import SIDE_CAMERA, WRIST_CAMERA, BoxSpec, SceneSpec, build, mujoco

START_JOINTS = (0.0, -27.2, -1.0, 88.2, -90.0, 60.0)
AMPS_PER_NM = 0.9
IDLE_AMPS = 0.08
SUPPLY_V = 5.4
MAX_CATCH_UP_S = 2.0
CURRENT_FILTER_S = 0.08
RENDER_SIZE = (640, 360)
RENDER_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Optics:
    """What the real camera does to light that MuJoCo's rasterizer does not."""

    turn_raw_deg: float = 180.0
    barrel_k1: float = -0.205
    gain: tuple[float, float, float] = (0.96, 0.98, 1.03)
    gamma: float = 1.0
    blur_px: float = 0.6
    noise_sd: float = 0.012
    vignette: float = 0.15


@dataclass(frozen=True)
class ServoModel:
    """Menagerie's kp 998 tracks within 0.03° but spikes 1.6 A on every 30 Hz goal step; the real arm read
    0.1-0.25 A while moving, which kp near 80 reproduces."""

    kp: float = 80.0
    kv: float = 1.5


@dataclass(frozen=True)
class SimSpec:
    scene: SceneSpec = field(default_factory=SceneSpec)
    optics: Optics = field(default_factory=Optics)
    servo: ServoModel = field(default_factory=ServoModel)
    start_joints: tuple[float, ...] = START_JOINTS


class SimWorld:
    """One MjModel and MjData behind one lock; physics catches up to the clock on every touch."""

    def __init__(self, spec: SimSpec, clock: Clock):
        self.spec = spec
        self.clock = clock
        self.lock = threading.RLock()
        self.model = build(spec.scene)
        self.data = mujoco.MjData(self.model)
        jid = {m: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m) for m in MOTORS}
        self.qadr = np.array([self.model.jnt_qposadr[jid[m]] for m in MOTORS])
        self.act = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, m) for m in MOTORS])
        self.box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.box_qadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box")]
        for a in self.act:
            self.model.actuator_gainprm[a, 0] = spec.servo.kp
            self.model.actuator_biasprm[a, 1:3] = [-spec.servo.kp, -spec.servo.kv]
        self.jaw_bodies = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gripper", "moving_jaw_so101_v1")}
        self.reset(spec.scene.box)

    def reset(self, box: BoxSpec) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.model.geom_size[self.box_geom] = [box.w / 2, box.d / 2, box.h / 2]
            yaw = np.radians(box.yaw_deg)
            self.data.qpos[self.box_qadr:self.box_qadr + 7] = [box.x, box.y, self.spec.scene.table_z + box.h / 2,
                                                                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            q = np.radians(self.spec.start_joints)
            self.data.qpos[self.qadr] = q
            self.data.ctrl[self.act] = q
            self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            mujoco.mj_forward(self.model, self.data)
            self.box = box
            self.torque_avg = np.zeros(len(MOTORS))
            self._t0 = self.clock.now()

    def advance(self) -> None:
        with self.lock:
            behind = self.clock.now() - self._t0 - self.data.time
            if behind > MAX_CATCH_UP_S:
                self._t0 += behind - MAX_CATCH_UP_S
            alpha = self.model.opt.timestep / CURRENT_FILTER_S
            while self.data.time < self.clock.now() - self._t0:
                mujoco.mj_step(self.model, self.data)
                self.torque_avg += alpha * (np.abs(self.data.actuator_force[self.act]) - self.torque_avg)

    def box_pose(self) -> np.ndarray:
        return self.data.xpos[self.box_body].copy()

    def box_touches(self) -> set[int]:
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if self.box_geom in pair:
                other = (pair - {self.box_geom}).pop() if len(pair) == 2 else self.box_geom
                touching.add(int(self.model.geom_bodyid[other]))
        return touching


class SimArm:
    def __init__(self, world: SimWorld, kin: Kinematics):
        self.world = world
        self.kin = kin
        rng = np.degrees(world.model.jnt_range[[mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, m) for m in ARM]])
        self.limit = {m: float(min(-lo, hi)) for m, (lo, hi) in zip(ARM, rng)}

    def q(self) -> np.ndarray:
        with self.world.lock:
            self.world.advance()
            return np.degrees(self.world.data.qpos[self.world.qadr])

    def amps(self) -> np.ndarray:
        with self.world.lock:
            self.world.advance()
            if not self.torque_enabled():
                return np.zeros(len(MOTORS))
            return IDLE_AMPS + AMPS_PER_NM * self.world.torque_avg

    def voltage(self) -> np.ndarray:
        return np.full(len(MOTORS), SUPPLY_V)

    def write_goals(self, goals: dict[str, float]) -> None:
        w = self.world
        with w.lock:
            w.advance()
            for m, v in goals.items():
                i = w.act[MOTORS.index(m)]
                lo, hi = w.model.actuator_ctrlrange[i]
                w.data.ctrl[i] = float(np.clip(np.radians(v), lo, hi))

    def hold(self) -> None:
        w = self.world
        with w.lock:
            w.advance()
            w.data.ctrl[w.act] = w.data.qpos[w.qadr]
            w.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    def relax(self) -> None:
        with self.world.lock:
            self.world.advance()
            self.world.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    def torque_enabled(self) -> bool:
        return not (self.world.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))

    def reset(self, box) -> None:
        spec = self.world.spec.scene.box
        self.world.reset(replace(spec, x=box.x, y=box.y, w=box.w, d=box.d, h=box.h))

    def truth(self) -> dict:
        w = self.world
        with w.lock:
            w.advance()
            p = w.box_pose()
            q = np.degrees(w.data.qpos[w.qadr])
            held = w.jaw_bodies <= w.box_touches() and p[2] - w.box.h / 2 > w.spec.scene.table_z + 0.003
        box = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "w": w.box.w, "d": w.box.d, "h": w.box.h}
        return {"box": box, "held": bool(held), "tip": dict(zip("xyz", map(float, self.kin.tip(q)[:3, 3])))}


class SimCameras:
    """Renders on one thread of its own: a GL context made current on one HTTP request thread hangs on the next."""

    def __init__(self, world: SimWorld, size: tuple[int, int] = RENDER_SIZE, seed: int = 0):
        self.world = world
        self.size = size
        self.rng = np.random.default_rng(seed)
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._serve, daemon=True, name="sim-render").start()

    def capture(self, cam: str) -> Capture:
        names = {"wrist": WRIST_CAMERA, "side": SIDE_CAMERA}
        if cam not in names:
            raise NoFrame(f"the sim has no camera {cam!r}")
        image = Image.fromarray(self.render(names[cam]))
        if cam == "wrist":
            image = develop(image, self.world.spec.optics, self.rng)
        return Capture(image, time.time())

    def render(self, camera: str, segmentation: bool = False) -> np.ndarray:
        done: queue.Queue = queue.Queue(maxsize=1)
        self._jobs.put((camera, segmentation, done))
        ok, out = done.get(timeout=RENDER_TIMEOUT_S)
        if not ok:
            raise NoFrame(f"render failed: {out}")
        return out

    def _serve(self) -> None:
        w, h = self.size
        renderer = mujoco.Renderer(self.world.model, h, w)
        while True:
            camera, segmentation, done = self._jobs.get()
            try:
                if segmentation:
                    renderer.enable_segmentation_rendering()
                with self.world.lock:
                    self.world.advance()
                    renderer.update_scene(self.world.data, camera=camera)
                pixels = renderer.render().copy()
                done.put((True, pixels))
            except Exception as e:
                done.put((False, f"{type(e).__name__}: {e}"))
            finally:
                if segmentation:
                    renderer.disable_segmentation_rendering()


def develop(image: Image.Image, optics: Optics, rng: np.random.Generator) -> Image.Image:
    """Turn a clean upright-as-rendered wrist image into what the real camera sends: barrel distortion,
    colour, blur, noise, vignette, then the raw mounting turn that armd's upright step undoes."""
    a = np.asarray(image, np.float32) / 255.0
    h, w = a.shape[:2]
    if optics.barrel_k1:
        a = _barrel(a, optics.barrel_k1)
    if optics.vignette:
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (w / 2)) ** 2
        a = a * (1.0 - optics.vignette * r2)[..., None]
    a = np.clip(a * np.array(optics.gain, np.float32), 0, 1) ** optics.gamma
    if optics.noise_sd:
        a = a + rng.normal(0, optics.noise_sd, a.shape).astype(np.float32)
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if optics.blur_px:
        out = out.filter(ImageFilter.GaussianBlur(optics.blur_px))
    if optics.turn_raw_deg:
        out = out.rotate(optics.turn_raw_deg)
    return out


def _barrel(a: np.ndarray, k1: float) -> np.ndarray:
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2)
    scale = 1.0 + k1 * (nx * nx + ny * ny)
    sx = np.clip(nx * scale * (w / 2) + w / 2, 0, w - 1).astype(np.int32)
    sy = np.clip(ny * scale * (w / 2) + h / 2, 0, h - 1).astype(np.int32)
    return a[sy, sx]


def open_world(spec: SimSpec, clock: Clock, kin: Kinematics) -> tuple[SimWorld, SimArm, SimCameras]:
    world = SimWorld(spec, clock)
    return world, SimArm(world, kin), SimCameras(world)


def as_json(spec: SimSpec) -> dict:
    return asdict(spec)
