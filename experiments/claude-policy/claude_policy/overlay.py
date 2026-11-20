"""A table grid in base-frame metres and the grasp point, drawn onto the camera frames.

Uses only what a calibrated real arm also has: the joint state, the arm's kinematic model, each camera's pose
and lens (the wrist camera's mount and barrel distortion). Nothing about the box or the square."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from valet_sim.scene import MOTORS, mujoco
from valet_sim.sim import CAMERAS, Optics

GRID_M = 0.05
X_RANGE = (-0.05, 0.45)
Y_RANGE = (-0.35, 0.35)
LINE = (200, 30, 30, 110)
MARK = (230, 20, 20, 255)


class Overlay:
    def __init__(self, model: mujoco.MjModel, optics: Optics = Optics()):
        self.model = model
        self.data = mujoco.MjData(model)
        self.k1 = optics.barrel_k1
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in MOTORS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.cam = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for name, c in CAMERAS.items()}

    def draw(self, image: Image.Image, cam: str, joints: dict[str, float], grasp_point: list[float]) -> Image.Image:
        self.data.qpos[self.qadr] = np.radians([joints[j] for j in MOTORS])
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        w, h = image.size
        project = lambda pts: self._project(pts, cam, w, h)  # noqa: E731
        layer = Image.new("RGBA", image.size)
        d = ImageDraw.Draw(layer)
        step = GRID_M if cam == "wrist" else 2 * GRID_M
        xs = np.arange(np.ceil(X_RANGE[0] / step) * step, X_RANGE[1] + 1e-9, step)
        ys = np.arange(np.ceil(Y_RANGE[0] / step) * step, Y_RANGE[1] + 1e-9, step)
        n = 141
        fine = np.linspace(0, 1, n)
        for x in xs:
            self._polyline(d, project(np.c_[np.full(n, x), Y_RANGE[0] + fine * np.ptp(Y_RANGE), np.zeros(n)]), w, h)
        for y in ys:
            self._polyline(d, project(np.c_[X_RANGE[0] + fine * np.ptp(X_RANGE), np.full(n, y), np.zeros(n)]), w, h)
        placed: list[tuple[float, float]] = []
        for x in xs:
            for y in ys:
                (u, v), = project(np.array([[x, y, 0.0]]))
                if 0 <= u < w - 40 and 10 <= v < h and all(np.hypot(u - a, v - b) > 48 for a, b in placed):
                    placed.append((u, v))
                    d.text((u + 2, v - 11), f"{round(x * 100) + 0},{round(y * 100) + 0}", fill=LINE[:3] + (220,))
        g = np.array(grasp_point, float)
        (gu, gv), (tu, tv) = project(np.array([g, [g[0], g[1], 0.0]]))
        if np.isfinite(gu):
            d.ellipse((gu - 6, gv - 6, gu + 6, gv + 6), outline=MARK, width=2)
        if np.isfinite(tu):
            d.line((tu - 6, tv, tu + 6, tv), fill=MARK, width=2)
            d.line((tu, tv - 6, tu, tv + 6), fill=MARK, width=2)
        return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")

    def _project(self, pts: np.ndarray, cam: str, w: int, h: int) -> np.ndarray:
        """Pixel coordinates of world points (NaN behind the camera or outside the lens model)."""
        i = self.cam[cam]
        rot = self.data.cam_xmat[i].reshape(3, 3)
        pc = (pts - self.data.cam_xpos[i]) @ rot  # camera frame: x right, y up, looking down -z
        depth = -pc[:, 2]
        f = (h / 2) / np.tan(np.radians(self.model.cam_fovy[i]) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sx, sy = f * pc[:, 0] / depth, -f * pc[:, 1] / depth  # pixels from the centre, y down
        sx[depth <= 1e-3] = np.nan
        if cam == "wrist" and self.k1:
            # The developed frame samples the pinhole image at r_src = r_out * (1 + k1 r_out^2), r in units of w/2;
            # invert by bisection on the rising part of that curve, and drop points beyond it (outside the frame).
            rs = np.hypot(sx, sy) / (w / 2)
            top = np.sqrt(-1 / (3 * self.k1)) if self.k1 < 0 else 2.0
            lo, hi = np.zeros_like(rs), np.full_like(rs, top)
            for _ in range(40):
                mid = (lo + hi) / 2
                below = mid * (1 + self.k1 * mid**2) < rs
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            ok = np.isfinite(rs) & (rs < top * (1 + self.k1 * top**2) - 1e-3)
            scale = np.where(ok & (rs > 0), lo / np.where(rs > 0, rs, 1), np.where(rs == 0, 1.0, np.nan))
            sx, sy = sx * scale, sy * scale
        return np.c_[sx + w / 2, sy + h / 2]

    @staticmethod
    def _polyline(d: ImageDraw.ImageDraw, uv: np.ndarray, w: int, h: int) -> None:
        inside = lambda u, v: -20 <= u <= w + 20 and -20 <= v <= h + 20  # noqa: E731
        for (u0, v0), (u1, v1) in zip(uv[:-1], uv[1:]):
            if np.isfinite([u0, v0, u1, v1]).all() and inside(u0, v0) and inside(u1, v1) and np.hypot(u1 - u0, v1 - v0) < 60:
                d.line((u0, v0, u1, v1), fill=LINE, width=1)
