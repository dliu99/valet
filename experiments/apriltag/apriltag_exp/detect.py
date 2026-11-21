"""Colour detectors for the white box and the blue square in a wrist frame. No model: deterministic and free, so
every camera-pose method in the comparison sees the same pixels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Lens, on_plane


@dataclass(frozen=True)
class Blob:
    centroid_px: np.ndarray
    pixels: np.ndarray  # Nx2 (u, v)
    touches_edge: bool


def _largest(mask: np.ndarray, min_area: int) -> Blob | None:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    v, u = np.nonzero(labels == k)
    x, y, w, h = stats[k, :4]
    edge = x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1 or y + h >= mask.shape[0] - 1
    return Blob(np.array([u.mean(), v.mean()]), np.column_stack([u, v]).astype(float), bool(edge))


def white(frame: np.ndarray, exclude: np.ndarray | None = None) -> Blob | None:
    """The biggest white or light-grey patch: the box's lit and shaded faces. The box is a cool white and the
    table a warm beige, so the cue is blue at least red, without the blue square's saturation."""
    a = np.asarray(frame).astype(np.int16)
    spread = a.max(2) - a.min(2)
    mask = (a[..., 2] - a[..., 0] >= 4) & (spread < 45) & (a.mean(2) > 80)
    if exclude is not None:
        mask &= ~exclude
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    return _largest(mask, 200)


def blue(frame: np.ndarray, exclude: np.ndarray | None = None) -> Blob | None:
    hsv = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 90) & (hsv[..., 2] > 15)
    if exclude is not None:
        mask &= ~exclude
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return _largest(mask, 150)


def on_table(blob: Blob, T_world_cam: np.ndarray, z: float, lens: Lens = Lens()) -> np.ndarray | None:
    """The blob's area centroid on the plane at height z: right for a flat shape seen whole, whatever the view."""
    px = blob.pixels[:: max(1, len(blob.pixels) // 1500)]
    pts = []
    for uv in px:
        p = on_plane(*lens.ray(T_world_cam, uv), z)
        if p is not None:
            pts.append(p)
    return np.mean(pts, axis=0) if pts else None


def arm_mask(sim, grow_px: int = 8) -> np.ndarray:
    """Where the gripper itself shows in the frame (it rides with the camera, so the place is fixed for a given
    jaw opening). Taken once from the sim's segmentation of the current pose, as one would mask a real frame of an
    empty table; then bent by the same lens resampling as the frame."""
    import mujoco

    m = sim.model
    r = mujoco.Renderer(m, sim.spec.image_size[1], sim.spec.image_size[0])
    r.enable_segmentation_rendering()
    r.update_scene(sim.data, camera="wrist_cam")
    seg = r.render()[..., 0]
    r.close()
    arm_bodies = set(range(1, m.nbody)) - {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")}
    geom_body = np.full(m.ngeom + 1, -1)
    geom_body[: m.ngeom] = m.geom_bodyid
    mask = np.isin(geom_body[seg.clip(-1, m.ngeom)], list(arm_bodies)) & (seg >= 0)
    h, w = mask.shape
    k1 = sim.spec.optics.barrel_k1
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2)
    s = 1.0 + k1 * (nx * nx + ny * ny)
    mask = mask[np.clip(ny * s * (w / 2) + h / 2, 0, h - 1).astype(int), np.clip(nx * s * (w / 2) + w / 2, 0, w - 1).astype(int)]
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow_px + 1,) * 2, np.uint8)) > 0
