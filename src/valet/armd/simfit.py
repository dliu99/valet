"""Fit the simulated wrist camera to the real one from the fixed jaw's outline.

The camera and the fixed jaw are both bolted to the gripper, so where the jaw falls in the image depends only
on the camera's mount and lens, never on the arm's joints: any real upright frame with the gripper open is a
calibration target."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

from .kinematics import MOTORS, Kinematics
from .simscene import WRIST_CAMERA, Look, SceneSpec, build, mujoco

FIT_SIZE = (160, 90)
ALONG_JAWS_WITHIN_DEG = 25.0
LEVEL_WITHIN_DEG = 2.0
WORKING_POSE = (0.0, -2.65, 73.62, -40.97, -90.0, 60.0)
JAW_BODIES = ("gripper",)


def real_jaw_mask(image: Image.Image, size: tuple[int, int] = FIT_SIZE) -> np.ndarray:
    a = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), np.float32) / 255.0
    w, h = size
    v = a.max(2)
    s = (a.max(2) - a.min(2)) / np.maximum(v, 1e-3)
    desk = np.percentile(v[int(0.65 * h):int(0.95 * h), int(0.62 * w):int(0.94 * w)], 50)
    jaw = np.percentile(v[int(0.83 * h):, int(0.22 * w):int(0.4 * w)], 90)
    bright = ndimage.binary_closing((v > (desk + jaw) / 2) & (s < 0.2), iterations=1)
    lab, n = ndimage.label(bright)
    keep = np.zeros_like(bright)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if ys.max() >= h - 1 - h // 30 and 0.1 * w < xs.mean() < 0.6 * w and len(xs) > w * h // 60:
            keep |= lab == k
    keep[: int(0.55 * h)] = False
    return ndimage.binary_fill_holes(keep)


@dataclass(frozen=True)
class CameraFit:
    dpos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fovy_deg: float = 61.0
    barrel_k1: float = 0.0

    def look(self, base: Look) -> Look:
        quat = _compose(base.wrist_quat, self.drot_deg)
        pos = tuple(float(p + d) for p, d in zip(base.wrist_pos, self.dpos))
        return replace(base, wrist_pos=pos, wrist_quat=quat, wrist_fovy_deg=self.fovy_deg)

    @staticmethod
    def from_vector(x: np.ndarray) -> "CameraFit":
        return CameraFit(tuple(x[0:3] / 100.0), tuple(x[3:6]), float(x[6]), float(x[7]))

    def vector(self) -> np.ndarray:
        return np.array([*np.array(self.dpos) * 100.0, *self.drot_deg, self.fovy_deg, self.barrel_k1])


def _compose(quat: tuple[float, ...], rot_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    r = np.radians(rot_deg)
    angle = float(np.linalg.norm(r))
    if angle < 1e-9:
        return tuple(quat)
    axis = r / angle
    dq = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.array(quat, float), dq)
    return tuple(float(v) for v in out / np.linalg.norm(out))


class JawRenderer:
    """Renders which pixels of the wrist image the fixed jaw covers, for a candidate camera fit."""

    def __init__(self, gripper_pct: float = 60.0, size: tuple[int, int] = FIT_SIZE):
        self.model = build(SceneSpec())
        self.data = mujoco.MjData(self.model)
        self.size = size
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.base = Look()
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self.data.qpos[self.model.jnt_qposadr[gid]] = np.radians(gripper_pct)
        box = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box")
        self.data.qpos[self.model.jnt_qposadr[box]:self.model.jnt_qposadr[box] + 3] = [5.0, 5.0, -5.0]
        mujoco.mj_forward(self.model, self.data)
        bodies = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in JAW_BODIES}
        self.jaw_geoms = np.array([g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies
                                   and self.model.geom_group[g] != 3 and self.model.geom_contype[g] == 0])
        self.renderer = mujoco.Renderer(self.model, size[1], size[0])
        self.renderer.enable_segmentation_rendering()
        q = np.zeros(6)
        q[5] = gripper_pct
        self.jaw_axis = Kinematics().tip(q)[:3, 2]
        self.working = mujoco.MjData(self.model)
        adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in MOTORS]
        self.working.qpos[adr] = np.radians(WORKING_POSE)

    def tilt_deg(self) -> float:
        mujoco.mj_forward(self.model, self.working)
        right = self.working.cam_xmat[self.cam].reshape(3, 3)[:, 0]
        return float(np.degrees(np.arcsin(np.clip(right[2], -1, 1))))

    def off_jaw_deg(self) -> float:
        optical = -self.data.cam_xmat[self.cam].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(optical @ self.jaw_axis, -1, 1))))

    def mask(self, fit: CameraFit) -> np.ndarray:
        look = fit.look(self.base)
        self.model.cam_pos[self.cam] = look.wrist_pos
        self.model.cam_quat[self.cam] = look.wrist_quat
        self.model.cam_fovy[self.cam] = look.wrist_fovy_deg
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=WRIST_CAMERA)
        seg = self.renderer.render()
        ids, kinds = seg[..., 0], seg[..., 1]
        m = (kinds == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids, self.jaw_geoms)
        return barrel_mask(m, fit.barrel_k1)


def barrel_mask(m: np.ndarray, k1: float) -> np.ndarray:
    if not k1:
        return m
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2)
    scale = 1.0 + k1 * (nx * nx + ny * ny)
    sx = np.clip(nx * scale * (w / 2) + w / 2, 0, w - 1).astype(np.int32)
    sy = np.clip(ny * scale * (w / 2) + h / 2, 0, h - 1).astype(np.int32)
    return m[sy, sx]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


BOUNDS = [(-3, 3), (-3, 3), (-3, 3), (-40, 40), (-30, 30), (-30, 30), (45, 90), (-0.4, 0.1)]


def fit_camera(targets: list[np.ndarray], seed: int = 0, maxiter: int = 60) -> tuple[CameraFit, float]:
    jaw = JawRenderer()

    def loss(x: np.ndarray) -> float:
        m = jaw.mask(CameraFit.from_vector(x))
        off = max(0.0, jaw.off_jaw_deg() - ALONG_JAWS_WITHIN_DEG)
        tilt = max(0.0, abs(jaw.tilt_deg()) - LEVEL_WITHIN_DEG)
        return 1.0 - float(np.mean([iou(m, t) for t in targets])) + off / 20.0 + tilt / 5.0

    result = optimize.differential_evolution(loss, BOUNDS, seed=seed, maxiter=maxiter, popsize=12, tol=1e-4, polish=False,
                                             x0=CameraFit().vector())
    polished = optimize.minimize(loss, result.x, method="Powell", bounds=BOUNDS, options={"xtol": 1e-3, "ftol": 1e-4})
    best = polished if polished.fun < result.fun else result
    return CameraFit.from_vector(best.x), 1.0 - float(best.fun)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m valet.armd.simfit", description=__doc__)
    p.add_argument("frames", nargs="+", type=Path, help="real upright wrist frames with the gripper open")
    p.add_argument("--out", type=Path, required=True, help="where to write the fit as JSON")
    p.add_argument("--maxiter", type=int, default=60)
    a = p.parse_args(argv)
    targets = [real_jaw_mask(Image.open(f)) for f in a.frames]
    for f, t in zip(a.frames, targets):
        print(f"{f.name}: jaw covers {t.mean():.1%} of the frame")
    jaw = JawRenderer()
    baseline = jaw.mask(CameraFit())
    print(f"current camera: jaw IoU {np.mean([iou(baseline, t) for t in targets]):.3f}, {jaw.off_jaw_deg():.0f}° off the jaw axis, {jaw.tilt_deg():.0f}° tilt at the working pose")
    fit, score = fit_camera(targets, maxiter=a.maxiter)
    final = jaw.mask(fit)
    print(f"fitted camera: jaw IoU {np.mean([iou(final, t) for t in targets]):.3f}, {jaw.off_jaw_deg():.0f}° off the jaw axis, {jaw.tilt_deg():.0f}° tilt  {asdict(fit)}")
    a.out.write_text(json.dumps({"camera": asdict(fit), "jaw_iou": score, "frames": [str(f) for f in a.frames]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
