"""From a mask in one wrist frame to a pose on the table, in code: the pixel-to-metre step.

The camera's pose comes from the joints, so a shape of known kind sitting on the table has only a few unknowns.
`fit_box` finds the upright square-based box (x, y, yaw, side, height), resting on the table, whose silhouette
best overlaps the mask (soft IoU, anti-aliased raster, Nelder-Mead from a few yaw starts). `fit_square` does the
same for the flat target square of known size. Pixels the gripper covers, or another object hides, are ignored."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from .camera import H, W, Camera

SHIFT = 4  # fixed-point bits for sub-pixel polygon vertices
EDGE_SAMPLES = 6  # points along each edge, so the lens's bent edges are followed
SIDE_M = (0.025, 0.08)
HEIGHT_M = (0.025, 0.12)
SQUARE_Z = 0.0005


@dataclass
class BoxFit:
    x: float
    y: float
    yaw_deg: float  # in [-45, 45): a square box looks the same every 90 degrees
    side: float
    h: float
    iou: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class SquareFit:
    x: float
    y: float
    yaw_deg: float
    iou: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


def _edges(corners: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    t = np.linspace(0, 1, EDGE_SAMPLES)[:, None]
    return np.concatenate([corners[a] + t * (corners[b] - corners[a]) for a, b in pairs])


def cuboid_points(x, y, yaw_deg, side, h) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    r = side / 2
    base = np.array([[-r, -r], [r, -r], [r, r], [-r, r]]) @ np.array([[c, s], [-s, c]]) + [x, y]
    corners = np.concatenate([np.c_[base, np.zeros(4)], np.c_[base, np.full(4, h)]])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return _edges(corners, pairs)


def square_points(x, y, yaw_deg, size, z=SQUARE_Z) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    r = size / 2
    base = np.array([[-r, -r], [r, -r], [r, r], [-r, r]]) @ np.array([[c, s], [-s, c]]) + [x, y]
    return _edges(np.c_[base, np.full(4, z)], [(0, 1), (1, 2), (2, 3), (3, 0)])


class Silhouette:
    """Soft IoU between a projected convex shape and a mask, inside a window around the mask."""

    def __init__(self, cam: Camera, joints: dict[str, float], mask: np.ndarray, ignore: np.ndarray | None, margin: int = 60):
        self.cam = cam
        self.pos, self.rot = cam.pose(joints)
        ys, xs = np.nonzero(mask)
        self.x0, self.y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
        self.x1, self.y1 = min(W, xs.max() + margin + 1), min(H, ys.max() + margin + 1)
        m = mask[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        care = np.ones_like(m) if ignore is None else (~ignore[self.y0:self.y1, self.x0:self.x1]).astype(np.float32)
        self.m, self.care = m * care, care
        self.canvas = np.zeros(m.shape, np.uint8)

    def raster(self, pts3: np.ndarray) -> np.ndarray | None:
        px = self.cam.project_from(self.pos, self.rot, pts3)
        if np.isnan(px).any():
            return None
        hull = cv2.convexHull(((px - [self.x0, self.y0]) * (1 << SHIFT)).astype(np.int32))
        self.canvas[:] = 0
        cv2.fillConvexPoly(self.canvas, hull, 255, lineType=cv2.LINE_AA, shift=SHIFT)
        return self.canvas.astype(np.float32) / 255.0

    def iou(self, pts3: np.ndarray) -> float:
        p = self.raster(pts3)
        if p is None:
            return 0.0
        p = p * self.care
        u = np.maximum(p, self.m).sum()
        return float(np.minimum(p, self.m).sum() / u) if u else 0.0


def mask_contour(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)[:, 0, :].astype(float) if cs else np.zeros((0, 2))


def plane_centroid(cam: Camera, joints: dict[str, float], mask: np.ndarray, z: float) -> np.ndarray | None:
    """Centroid of the mask's outline back-projected onto the plane at height z (shoelace on the table)."""
    c = mask_contour(mask)
    if len(c) < 3:
        return None
    p = cam.on_plane(joints, c + 0.5, z)[:, :2]
    if np.isnan(p).any():
        return None
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return p.mean(0)
    return np.array([((x + xn) * cross).sum() / (6 * a), ((y + yn) * cross).sum() / (6 * a)])


def pixel_on_plane(cam: Camera, joints: dict[str, float], px, z: float) -> np.ndarray | None:
    p = cam.on_plane(joints, np.asarray(px, float), z)
    return None if np.isnan(p).any() else p[:2]


def _wrap45(yaw: float) -> float:
    return (yaw + 45) % 90 - 45


def fit_box(cam: Camera, joints: dict[str, float], mask: np.ndarray, ignore: np.ndarray | None = None,
            hw: tuple[float, float] | None = None) -> BoxFit | None:
    """The resting square-based box whose silhouette best matches `mask`. With `hw` (side, height) known from an
    earlier, oblique look, only x, y and yaw are fitted: from straight above, height and size trade off."""
    if mask.sum() < 30:
        return None
    sil = Silhouette(cam, joints, mask, ignore)
    side0, h0 = hw or (0.0475, 0.075)
    start = plane_centroid(cam, joints, mask, h0 / 2)
    if start is None:
        return None

    def cost(v):
        x, y, yaw, side, h = v if hw is None else (*v, *hw)
        if not (SIDE_M[0] <= side <= SIDE_M[1] and HEIGHT_M[0] <= h <= HEIGHT_M[1]):
            return 1.0 + abs(side - np.clip(side, *SIDE_M)) * 100 + abs(h - np.clip(h, *HEIGHT_M)) * 100
        return 1.0 - sil.iou(cuboid_points(x, y, yaw, side, h))

    steps = [[0.01, 0, 0, 0, 0], [0, 0.01, 0, 0, 0], [0, 0, 15, 0, 0], [0, 0, 0, 0.006, 0], [0, 0, 0, 0, 0.015]]
    n = 5 if hw is None else 3
    best = None
    for yaw0 in (-30.0, 0.0, 30.0):
        v0 = np.array([start[0], start[1], yaw0, side0, h0])[:n]
        simplex = np.array([v0] + [v0 + np.array(s[:n]) for s in steps[:n]])
        r = minimize(cost, v0, method="Nelder-Mead", options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw, side, h = best.x if hw is None else (*best.x, *hw)
    return BoxFit(float(x), float(y), _wrap45(float(yaw)), float(side), float(h), 1.0 - float(best.fun))


def fit_box_views(cam: Camera, views: list[tuple[dict[str, float], np.ndarray, np.ndarray | None]]) -> BoxFit | None:
    """One resting box fitted to masks from several camera poses at once (mean soft IoU). Two looks from
    different directions fix side and height, which one look from above cannot."""
    views = [v for v in views if v[1].sum() >= 30]
    if not views:
        return None
    sils = [Silhouette(cam, j, m, ign) for j, m, ign in views]
    j, m, _ = views[-1]
    start = plane_centroid(cam, j, m, 0.0375)
    if start is None:
        return None

    def cost(v):
        x, y, yaw, side, h = v
        if not (SIDE_M[0] <= side <= SIDE_M[1] and HEIGHT_M[0] <= h <= HEIGHT_M[1]):
            return 1.0 + abs(side - np.clip(side, *SIDE_M)) * 100 + abs(h - np.clip(h, *HEIGHT_M)) * 100
        pts = cuboid_points(x, y, yaw, side, h)
        return 1.0 - float(np.mean([s.iou(pts) for s in sils]))

    steps = np.diag([0.01, 0.01, 15, 0.006, 0.015])
    best = None
    for yaw0 in (-30.0, 0.0, 30.0):
        v0 = np.array([start[0], start[1], yaw0, 0.0475, 0.075])
        r = minimize(cost, v0, method="Nelder-Mead", options={"initial_simplex": np.vstack([v0, v0 + steps]), "xatol": 1e-4,
                                                              "fatol": 1e-4, "maxiter": 800})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw, side, h = best.x
    return BoxFit(float(x), float(y), _wrap45(float(yaw)), float(side), float(h), 1.0 - float(best.fun))


def fit_square(cam: Camera, joints: dict[str, float], mask: np.ndarray, size: float, ignore: np.ndarray | None = None) -> SquareFit | None:
    """The flat square of known `size` on the table whose outline best matches `mask`."""
    if mask.sum() < 30:
        return None
    sil = Silhouette(cam, joints, mask, ignore)
    start = plane_centroid(cam, joints, mask, SQUARE_Z)
    if start is None:
        return None

    def cost(v):
        return 1.0 - sil.iou(square_points(v[0], v[1], v[2], size))

    best = None
    for yaw0 in (-30.0, 0.0, 30.0):
        v0 = np.array([start[0], start[1], yaw0])
        simplex = np.array([v0, v0 + [0.01, 0, 0], v0 + [0, 0.01, 0], v0 + [0, 0, 15]])
        r = minimize(cost, v0, method="Nelder-Mead", options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw = best.x
    return SquareFit(float(x), float(y), _wrap45(float(yaw)), 1.0 - float(best.fun))
