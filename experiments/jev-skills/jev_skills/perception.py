"""Fast perception: Jev draws a box around the white box and the blue square in the wrist frame, and code refines
inside it (the box's bright top face, the square's blue pixels) and back-projects through the camera model.

The top-face triangulation and rectangle fit follow the aspire team's validated skills S01 and S02
(branch claude/aspire-skills-1325c3, experiments/aspire/skills/)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arm import Arm, View, jaw_mask

TARGET_SIDE_M = 0.07
BOX_Q = {"type": "box", "instructions": "Draw a box around the white box standing on the table in `wrist`."}
SQUARE_Q = {"type": "box", "instructions": "Draw a box around the blue square taped on the table in `wrist`."}


@dataclass
class Sighting:
    """One look at the box or the square: the Jev box (pixels), the refined pixels, and the frame's joints."""

    what: str
    bbox: tuple[float, float, float, float] | None
    confidence: float | None
    us: np.ndarray
    vs: np.ndarray
    joints: dict
    note: str = ""

    @property
    def ok(self) -> bool:
        return len(self.us) >= 30

    @property
    def centre_px(self) -> tuple[float, float]:
        return float(self.us.mean()), float(self.vs.mean())


def ask_boxes(jev, frame: np.ndarray, want: tuple[str, ...], use: str) -> dict:
    qs = {}
    if "box" in want:
        qs["where:box"] = BOX_Q
    if "square" in want:
        qs["where:square"] = SQUARE_Q
    return jev.ask({}, frame, qs, use)


def _bbox(answer: dict | None, w: int, h: int) -> tuple[tuple[float, float, float, float] | None, float | None]:
    if not answer:
        return None, None
    b = answer.get("box")
    conf = answer.get("confidence")
    if not b:
        return None, conf
    box = (b["left"] * w, b["top"] * h, b["right"] * w, b["bottom"] * h)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None, conf
    return box, None if conf is None else float(conf)


def _window(bbox, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    l, t, r, b = bbox
    return max(0, int(l - pad)), max(0, int(t - pad)), min(w, int(r + pad)), min(h, int(b + pad))


def box_pixels(frame: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray]:
    """The box's top face inside the Jev box: the brightest, least coloured pixels, minus the arm's own jaw."""
    h, w = frame.shape[:2]
    l, t, r, b = _window(bbox, 12, w, h)
    crop = frame[t:b, l:r].astype(float)
    if crop.size == 0:
        return np.array([]), np.array([])
    gray, sat = crop.mean(2), crop.max(2) - crop.min(2)
    ok = ~jaw_mask(w, h)[t:b, l:r]
    # measured on seeds 0 and 3 (start and hover views): the top face reads 209-250 even under the arm's shadow,
    # the side faces 95-165 and the table's 90th percentile under 185
    mask = ok & (gray > TOP_FACE_GRAY) & (sat < 35)
    vs, us = np.nonzero(mask)
    return us + l, vs + t


TOP_FACE_GRAY = 196


def triangulate(views: list[tuple[View, tuple[float, float]]]) -> np.ndarray | None:
    """The point nearest to every ray in the least-squares sense (the top face's centre seen from several poses)."""
    if len(views) < 2:
        return None
    A, b = np.zeros((3, 3)), np.zeros(3)
    for v, px in views:
        o, d = v.ray(*px)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    if np.linalg.cond(A) > 1e6:
        return None
    return np.linalg.solve(A, b)


def square_pixels(frame: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    l, t, r, b = _window(bbox, 30, w, h)
    c = frame[t:b, l:r].astype(int)
    mask = (c[..., 2] - c[..., 0] > 50) & (c[..., 2] - c[..., 1] > 25)
    vs, us = np.nonzero(mask)
    return us + l, vs + t


def look(arm: Arm, jev, want: tuple[str, ...], use: str) -> dict[str, Sighting]:
    frame, joints = arm.frame()
    answers = ask_boxes(jev, frame, want, use)
    h, w = frame.shape[:2]
    out = {}
    for what in want:
        bbox, conf = _bbox(answers.get(f"where:{what}"), w, h)
        if bbox is None:
            out[what] = Sighting(what, None, conf, np.array([]), np.array([]), joints, "Jev found no " + what)
            continue
        us, vs = (box_pixels if what == "box" else square_pixels)(frame, bbox)
        s = Sighting(what, bbox, conf, us, vs, joints)
        if not s.ok:
            s.note = f"Jev boxed the {what} but too few {what} pixels inside ({len(us)})"
        out[what] = s
    out["_frame"] = frame  # type: ignore[assignment]
    return out


# ---- geometry -------------------------------------------------------------------------------------------------

def ray_meet(a: View, pa, b: View, pb) -> np.ndarray | None:
    o1, d1 = a.ray(*pa)
    o2, d2 = b.ray(*pb)
    w0 = o1 - o2
    bb, dd, ee = d1 @ d2, d1 @ w0, d2 @ w0
    den = 1 - bb * bb
    if abs(den) < 1e-6:
        return None
    s, t = (bb * ee - dd) / den, (ee - bb * dd) / den
    return 0.5 * (o1 + s * d1 + o2 + t * d2)


def baseline(a: View, b: View) -> float:
    return float(np.linalg.norm(a.pos - b.pos))


def top_rect(view: View, us: np.ndarray, vs: np.ndarray, z: float) -> dict | None:
    """Back-project top-face pixels to the plane at the box's top and fit the smallest rectangle around them."""
    n = len(us)
    if n < 30:
        return None
    step = max(1, n // 1500)
    pts = [view.to_plane(float(u), float(v), z) for u, v in zip(us[::step], vs[::step])]
    pts = np.array([p[:2] for p in pts if p is not None])
    if len(pts) < 30:
        return None
    best = None
    for deg in range(0, 90):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        x = pts[:, 0] * c + pts[:, 1] * s
        y = -pts[:, 0] * s + pts[:, 1] * c
        x0, x1 = np.percentile(x, [1, 99])
        y0, y1 = np.percentile(y, [1, 99])
        area = (x1 - x0) * (y1 - y0)
        if best is None or area < best["area"]:
            xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            best = {"area": area, "yaw": float(deg), "da": float(x1 - x0), "db": float(y1 - y0),
                    "x": float(xm * c - ym * s), "y": float(xm * s + ym * c)}
    return best


def square_centre(view: View, s: Sighting) -> tuple[np.ndarray | None, str]:
    """Centre of the square from its blue pixels, or None with the reason when the view is cut off or covered."""
    if not s.ok:
        return None, s.note or "square not seen"
    if s.us.min() <= 2 or s.vs.min() <= 2 or s.us.max() >= view.width - 3 or s.vs.max() >= view.height - 3:
        return None, "the square runs off the edge of the view"
    r = top_rect(view, s.us, s.vs, 0.0005)
    if r is None:
        return None, "square pixels do not back-project"
    if min(r["da"], r["db"]) < 0.8 * TARGET_SIDE_M:
        return None, f"only part of the square is visible ({100 * min(r['da'], r['db']):.1f} cm of 7)"
    return np.array([r["x"], r["y"]]), "ok"
