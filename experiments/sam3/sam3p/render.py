"""Per-pixel object ids in the delivered wrist frame, from MuJoCo's segmentation render put through the same lens
resampling as `valet_sim.sim.develop`.

Two uses, kept apart:
- `JawMask`: which pixels the arm's own gripper covers. The camera rides on the gripper, so this is a fixed
  calibration of the arm (rendered from the arm model alone, no scene), which the real arm would get from one photo.
- `true_masks`: the box and square pixels of a live sim. Grading only; no policy may call it."""

from __future__ import annotations

import mujoco
import numpy as np
from valet_sim.scene import MOTORS, WRIST_CAMERA
from valet_sim.sim import Optics

from .camera import H, W, arm_model


def _lens_index(k1: float = Optics().barrel_k1) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (W / 2)
    s = 1.0 + k1 * (nx * nx + ny * ny)
    return (np.clip(ny * s * (W / 2) + H / 2, 0, H - 1).astype(int), np.clip(nx * s * (W / 2) + W / 2, 0, W - 1).astype(int))


class SegRenderer:
    """Geom ids per delivered pixel (-1 for none). Use from one thread: GL contexts belong to the thread that made them."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.renderer = mujoco.Renderer(model, H, W)
        self.renderer.enable_segmentation_rendering()
        self.index = _lens_index()

    def geoms(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=WRIST_CAMERA)
        seg = self.renderer.render()
        ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        return ids[self.index]


class JawMask:
    """The gripper's own pixels in the wrist frame, for a gripper opening (cached by whole percent)."""

    def __init__(self):
        self.model = arm_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.seg = SegRenderer(m)
        bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("gripper", "moving_jaw_so101_v1")}
        self.jaw_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies])
        self.cache: dict[int, np.ndarray] = {}

    def __call__(self, gripper: float) -> np.ndarray:
        key = int(round(gripper))
        if key not in self.cache:
            q = {"shoulder_pan": 0.0, "shoulder_lift": -27.2, "elbow_flex": -1.0, "wrist_flex": 88.2, "wrist_roll": -90.0, "gripper": float(key)}
            self.data.qpos[self.qadr] = np.radians([q[j] for j in MOTORS])
            mujoco.mj_forward(self.model, self.data)
            self.cache[key] = np.isin(self.seg.geoms(self.data), self.jaw_geoms)
        return self.cache[key]


def true_masks(seg: SegRenderer, sim) -> dict[str, np.ndarray]:
    """Visible box and square pixels in the sim's current wrist frame. Grading only."""
    with sim.lock:
        ids = seg.geoms(sim.data)
    return {"box": ids == sim.box_geom, "square": ids == sim.target_geom}


class OracleSegmenter:
    """Stands in for SAM 3 with the sim's true masks, to debug control code and bound what perfect masks give.
    It reads the sim's state, so it is never a policy's perception; results from it are labelled oracle."""

    def __init__(self):
        self.sim = None
        self.seg = None
        self.calls: list[dict] = []

    def attach(self, sim) -> None:
        self.sim, self.seg = sim, SegRenderer(sim.model)

    def segment(self, image, prompts: list[str]) -> dict:
        from .seg import Instance

        tm = true_masks(self.seg, self.sim)
        self.calls.append({"s": 0.0, "prompts": len(prompts)})
        out = {}
        for p in prompts:
            m = tm["box"] if "box" in p else tm["square"]
            if m.sum() < 30:
                out[p] = []
                continue
            ys, xs = np.nonzero(m)
            out[p] = [Instance(m, 1.0, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))]
        return out
